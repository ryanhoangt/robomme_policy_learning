"""Offline MRMG diagnostics. Heavy entrypoints require a Slurm job AND step.

This measures representation availability and teacher-forced/action sensitivity,
not robot success. Raw episode IDs are matched to cache IDs using RGB and state.
Probe splits hold out episodes from each other, NOT from the released policy's
original training set. See docs/memory_grounding_diagnostic.md.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import logging
import os
from pathlib import Path
import pickle
import re
import subprocess

import numpy as np

LOG = logging.getLogger(__name__)
TASKS = ("ButtonUnmask", "ButtonUnmaskSwap", "PickHighlight", "SwingXtimes")
DEFAULT_CHECKPOINT = "/mnt/data/vhoangth2/pretrained-weights/RoboMME-perceptual-framesamp-modul/79999"


def require_slurm():
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_STEP_ID"):
        raise SystemExit("Run this entrypoint with srun inside a Slurm allocation; do not compute on the login node.")


def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def text(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def digest(image):
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def uniform_indices(step, frames):
    if step < 0 or frames < 1:
        raise ValueError("Invalid history size")
    return list(range(step + 1)) if step < frames else np.linspace(0, step, frames, dtype=np.int32).tolist()


def interval_mask(steps, intervals):
    """Intervals are half-open [start,end); only already-observed frames exist."""
    steps = np.asarray(steps)
    result = np.zeros(steps.shape, dtype=bool)
    for start, end in intervals:
        if end <= start:
            raise ValueError("Cue intervals must have positive length")
        result |= (steps >= start) & (steps < end)
    return result


def corrupt_mask(mask, token_steps, intervals, *, control=False):
    """Keep token features/positions intact, masking cue or equally many other frames.

    The control is deterministic and avoids the current frame. It controls the
    number of removed frames, not all distribution shift caused by an ablation.
    """
    mask = np.asarray(mask, dtype=bool).copy()
    token_steps = np.asarray(token_steps)
    cue = interval_mask(token_steps, intervals) & mask
    removed = sorted(set(token_steps[cue].tolist()))
    if control:
        candidates = sorted(set(token_steps[mask & ~cue].tolist()))
        if candidates:
            candidates = [s for s in candidates if s != int(token_steps[mask].max())]
        if len(candidates) < len(removed):
            raise ValueError("Insufficient non-cue historical frames for a matched removal control")
        removed = candidates[: len(removed)]
    mask[np.isin(token_steps, removed)] = False
    if not mask.any():
        raise ValueError("Refusing all-masked softmax; use explicit memory bypass instead")
    return mask, removed


def paired_bootstrap(differences, seed=7, repeats=2000):
    """Each input must already be an independent episode-level mean."""
    values = np.asarray(differences, dtype=np.float64)
    if len(values) < 2:
        return {"n_episodes": len(values), "mean": float(values.mean()) if len(values) else None, "ci95": None}
    rng = np.random.default_rng(seed)
    means = values[rng.integers(len(values), size=(repeats, len(values)))].mean(axis=1)
    return {
        "n_episodes": len(values),
        "mean": float(values.mean()),
        "ci95": np.quantile(means, [0.025, 0.975]).tolist(),
    }


def physical_mse(prediction, target, horizon=16):
    """Exclude 24 padded action dimensions and unexecuted chunk tail."""
    error = np.asarray(prediction)[..., :horizon, :8] - np.asarray(target)[..., :horizon, :8]
    return np.mean(np.square(error), axis=(-2, -1))


def aggregate_flow_effects(rows, seed=7):
    """Pair interventions before averaging draws/timestamps within each episode."""
    points = {}
    for row in rows:
        key = (row["task"], int(row["raw_id"]), int(row["step"]), float(row["tau"]), int(row["noise_draw"]))
        points.setdefault(key, {})[row["variant"]] = row
    grouped = {}
    for (task, episode, step, tau, _draw), conditions in points.items():
        if "correct" not in conditions:
            raise ValueError("Missing paired intact-memory observation")
        correct = conditions["correct"]
        contrasts = []
        for name, row in conditions.items():
            if name == "correct":
                continue
            if name in ("cue_mask", "noncue_mask", "cue_replace") and int(row["cue_tokens_available"]) == 0:
                continue
            contrasts.append((name, "correct", float(row["flow_mse"]) - float(correct["flow_mse"])))
        if {"cue_mask", "noncue_mask"} <= conditions.keys() and int(correct["cue_tokens_available"]) > 0:
            contrasts.append(
                (
                    "cue_mask",
                    "noncue_mask",
                    float(conditions["cue_mask"]["flow_mse"]) - float(conditions["noncue_mask"]["flow_mse"]),
                )
            )
        for variant, reference, delta in contrasts:
            for region in ("all", correct["region"]):
                key = (task, variant, reference, tau, region)
                grouped.setdefault(key, {}).setdefault(episode, {}).setdefault(step, []).append(delta)
    effects = []
    for (task, variant, reference, tau, region), episodes in sorted(grouped.items()):
        # Equal timestamp weight inside episodes, then equal episode weight.
        means = [np.mean([np.mean(draws) for draws in timestamps.values()]) for timestamps in episodes.values()]
        effect = paired_bootstrap(means, seed)
        effects.append(dict(effect, task=task, variant=variant, reference=reference, tau=tau, region=region))
    return effects


def feature(root, cache_id, step):
    import ml_dtypes  # noqa: F401 -- native bfloat16 pickle registration

    result = np.load(root / "features" / f"episode_{cache_id}" / f"token_emb_{step}.npy", allow_pickle=True).item()
    for key, patches in (("image_emb_4x4", 16), ("image_emb_8x8", 64)):
        values = result[key]
        if values.shape != (1, patches, 2048) or not np.isfinite(values.astype(np.float32)).all():
            raise ValueError(f"Invalid native cache {cache_id}/{step}/{key}: {values.shape}")
    return result


def cache_length(directory):
    if not (directory / "token_emb_0.npy").exists():
        raise FileNotFoundError(directory)
    lo, hi = 0, 1
    while (directory / f"token_emb_{hi}.npy").exists():
        lo, hi = hi, hi * 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if (directory / f"token_emb_{mid}.npy").exists():
            lo = mid
        else:
            hi = mid
    return hi


def decision_from_stages(task, stages):
    """A recorded subgoal transition is a stage proxy, not exact state aliasing."""
    for step, simple, grounded in stages:
        if task == "SwingXtimes":
            matches = "press" in simple and "button" in simple
        else:
            matches = "pick up" in simple and ("container" in simple or "highlight" in simple)
        coordinates = re.findall(r"<\s*(\d+)\s*,\s*(\d+)\s*>", grounded)
        if matches and coordinates:
            y, x = map(int, coordinates[-1])
            if not (0 <= y <= 255 and 0 <= x <= 255):
                raise ValueError("Grounded coordinate outside raw 256x256 image")
            return {"step": step, "label_yx": [y, x], "simple": simple, "grounded": grounded}
    raise ValueError(f"Cannot identify a grounded target decision for {task}: {stages}")


def prepare(args):
    import h5py
    from PIL import Image
    from PIL import ImageDraw

    wanted, episodes = {}, []
    annotations = json.loads(args.cue_annotations.read_text()) if args.cue_annotations else {}
    for task in args.tasks:
        with h5py.File(args.raw_dir / f"record_dataset_{task}.h5") as raw:
            split_ids = [(i, "probe_train") for i in range(args.train_episodes)]
            split_ids += [(60 + i, "probe_val") for i in range(args.val_episodes)]
            split_ids += [(80 + i, "probe_test") for i in range(args.test_episodes)]
            for raw_id, split in split_ids:
                group = raw[f"episode_{raw_id}"]
                length = sum(k.startswith("timestep_") for k in group)
                start = 0
                while bool(group[f"timestep_{start}/info/is_video_demo"][()]):
                    start += 1
                stages, last = [], None
                for step in range(start, length):
                    info = group[f"timestep_{step}/info"]
                    simple = text(info["simple_subgoal"][()]).lower()
                    grounded = text(info["grounded_subgoal"][()]).lower()
                    # Grounded coordinates may vary within a stage: use its first value.
                    if simple != last and "complete" not in simple:
                        stages.append((step, simple, grounded))
                        last = simple
                decision = decision_from_stages(task, stages)
                override = annotations.get(f"{task}/{raw_id}")
                if override:
                    intervals = override["cue_intervals"]
                    cue_source = "provided_annotation"
                elif task.startswith("ButtonUnmask"):
                    from scripts.test_surprise_detector import visibility_proxy

                    images = np.stack([group[f"timestep_{t}/obs/front_rgb"][()] for t in range(min(80, length))])
                    _, end = visibility_proxy(images)
                    if end is None:
                        raise ValueError(f"No initial closure proxy for {task}/{raw_id}; provide annotation")
                    intervals, cue_source = [[0, end]], "initial_RGB_overlap_proxy_not_all_reveals"
                elif task == "PickHighlight":
                    intervals, cue_source = [[10, 100]], "simulator_schedule_proxy_requires_RGB_inspection"
                else:
                    intervals, cue_source = [[0, min(32, decision["step"])]], "early_window_control_not_cue_annotation"
                cue_step = (intervals[0][0] + intervals[0][1] - 1) // 2
                if cue_step > decision["step"] or intervals[0][1] > length:
                    raise ValueError(f"Cue is not in decision history for {task}/{raw_id}")
                prompt = text(group["setup/task_goal"][()][0]).lower()
                # Query information must come from the policy-visible instruction,
                # never from a privileged subgoal annotation.
                color = re.search(r"(?:hiding|hides) the (red|green|blue) cube", prompt)
                if color is None and task.startswith("ButtonUnmask"):
                    raise ValueError(f"Cannot parse instruction color for {task}/{raw_id}")
                query = color.group(1) if color else "highlight" if task == "PickHighlight" else "count"
                ep = {
                    "task": task,
                    "raw_id": raw_id,
                    "split": split,
                    "length": length,
                    "exec_start": start,
                    "prompt": prompt,
                    "decision": decision,
                    "stages": stages,
                    "cue_intervals": intervals,
                    "cue_step": cue_step,
                    "cue_source": cue_source,
                    "query": query,
                }
                key = digest(group[f"timestep_{start}/obs/front_rgb"][()])
                if key in wanted:
                    raise ValueError("Ambiguous duplicate initial RGB")
                wanted[key] = ep
                # A contact sheet is an audit artifact, not an automatic validation.
                if split == "probe_test" and raw_id < 80 + args.policy_episodes:
                    times = sorted(
                        {0, cue_step, intervals[0][1] - 1, min(intervals[0][1] + 2, length - 1), decision["step"]}
                    )
                    sheet = Image.new("RGB", (256 * len(times), 280), "white")
                    draw = ImageDraw.Draw(sheet)
                    for column, t in enumerate(times):
                        sheet.paste(Image.fromarray(group[f"timestep_{t}/obs/front_rgb"][()]), (column * 256, 24))
                        draw.text((column * 256 + 4, 4), f"t={t}; decision={decision['step']}", fill="black")
                    sheet.save(args.output_dir / f"{task}_{raw_id}_audit.png")
                    # Also expose later reveals/swaps; the initial-closure proxy
                    # deliberately does not claim to annotate these events.
                    history_sheet = Image.new("RGB", (1024, 1120), "white")
                    history_draw = ImageDraw.Draw(history_sheet)
                    for column, t in enumerate(np.linspace(0, decision["step"], 16, dtype=int)):
                        x, y = (column % 4) * 256, (column // 4) * 280
                        history_sheet.paste(Image.fromarray(group[f"timestep_{t}/obs/front_rgb"][()]), (x, y + 24))
                        history_draw.text((x + 4, y + 4), f"t={t}", fill="black")
                    history_sheet.save(args.output_dir / f"{task}_{raw_id}_history_audit.png")
    stats = json.loads((args.dataset_dir / "meta/stats.json").read_text())
    index = 0
    while index < stats["execution_samples"] and wanted:
        with (args.dataset_dir / "data" / f"{index}.pkl").open("rb") as f:
            sample = pickle.load(f)
        cache_id = int(np.asarray(sample["epis_idx"]).item())
        step = int(np.asarray(sample["step_idx"]).item())
        execution_start = int(np.asarray(sample["exec_start_idx"]).item())
        if step != execution_start:
            raise ValueError(f"Sample {index} is not an episode start")
        length = cache_length(args.dataset_dir / "features" / f"episode_{cache_id}")
        ep = wanted.get(digest(sample["image"]))
        if ep is not None:
            if length != ep["length"] or execution_start != ep["exec_start"] or sample["prompt"] != ep["prompt"]:
                raise ValueError("Raw/cache episode metadata mismatch")
            with h5py.File(args.raw_dir / f"record_dataset_{ep['task']}.h5") as raw:
                for t in sorted({step, length // 2, length - 1}):
                    obs = raw[f"episode_{ep['raw_id']}/timestep_{t}/obs"]
                    state = np.r_[obs["joint_state"][()], obs["gripper_state"][()][:1]]
                    np.testing.assert_allclose(
                        feature(args.dataset_dir, cache_id, t)["state_emb"], state, atol=1e-5, rtol=0
                    )
            ep.update(cache_id=cache_id, sample_start=index)
            episodes.append(ep)
            del wanted[digest(sample["image"])]
        index += length - step
    if wanted:
        raise ValueError(f"Unmatched raw episodes: {[(e['task'], e['raw_id']) for e in wanted.values()]}")
    episodes.sort(key=lambda e: (e["task"], e["raw_id"]))
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        diff = subprocess.check_output(["git", "diff", "--", "src/mme_vla_suite/models/integration"], text=True)
    except subprocess.CalledProcessError:
        revision, diff = "unknown", "unknown"
    (args.output_dir / "model_changes.patch").write_text(diff)
    dump_json(
        args.output_dir / "manifest.json",
        {
            "episodes": episodes,
            "args": vars_as_json(args),
            "git_revision": revision,
            "source_sha256": {
                str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (
                    Path(__file__),
                    Path("scripts/memory_grounding_policy.py"),
                    Path("src/mme_vla_suite/models/integration/history_gemma.py"),
                    Path("src/mme_vla_suite/models/integration/history_pi0.py"),
                )
            },
            "slurm_job": os.environ.get("SLURM_JOB_ID"),
            "scope": "Raw training episodes; held out only from diagnostic probes, not from the pretrained policy.",
        },
    )
    LOG.info("Verified %s raw/cache episodes", len(episodes))


def normalized_features(values):
    values = np.asarray(values, dtype=np.float32).reshape(len(values), -1)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)


def ridge_prediction(train_kernel, cross_kernel, targets, alpha):
    # Explicit intercept by centering the kernel using training statistics only.
    row_mean, grand = train_kernel.mean(axis=0), train_kernel.mean()
    centered = train_kernel - row_mean[None] - row_mean[:, None] + grand
    cross = cross_kernel - row_mean[None] - cross_kernel.mean(axis=1, keepdims=True) + grand
    mean_y = targets.mean(axis=0)
    weights = np.linalg.solve(centered + alpha * np.eye(len(targets)), targets - mean_y)
    return cross @ weights + mean_y


def run_probes(args):
    episodes = json.loads((args.output_dir / "manifest.json").read_text())["episodes"]
    results = []
    for task in args.tasks:
        if task == "SwingXtimes":
            continue  # There is no comparable transient object-location query in this control.
        eps = [e for e in episodes if e["task"] == task]
        train = np.array([i for i, e in enumerate(eps) if e["split"] == "probe_train"])
        val = np.array([i for i, e in enumerate(eps) if e["split"] == "probe_val"])
        test = np.array([i for i, e in enumerate(eps) if e["split"] == "probe_test"])
        labels = np.array([e["decision"]["label_yx"] for e in eps], dtype=np.float64)
        queries = np.array([e["query"] for e in eps])
        # Feature x one-hot(query) interactions prevent an unconditioned probe
        # from having to guess which color the instruction asks to localize.
        query_kernel = (queries[:, None] == queries[None, :]).astype(np.float64)
        predictions = {}
        for resolution in (4, 8):
            cue, current = [], []
            for ep in eps:
                cue.append(
                    feature(args.dataset_dir, ep["cache_id"], ep["cue_step"])[f"image_emb_{resolution}x{resolution}"]
                )
                current.append(
                    feature(args.dataset_dir, ep["cache_id"], ep["decision"]["step"])[
                        f"image_emb_{resolution}x{resolution}"
                    ]
                )
            cue, current = normalized_features(cue), normalized_features(current)
            kernels = {
                f"cue_{resolution}": cue @ cue.T,
                f"current_{resolution}": current @ current.T,
                f"cue_current_{resolution}": (cue @ cue.T + current @ current.T) / 2,
            }
            if resolution == 4:
                kernels["query_only"] = np.ones((len(eps), len(eps)))
            for name, raw_kernel in kernels.items():
                kernel = raw_kernel.astype(np.float64) * query_kernel
                alphas = (0.001, 0.01, 0.1, 1.0, 10.0)
                val_errors = [
                    np.mean(
                        np.square(
                            ridge_prediction(kernel[np.ix_(train, train)], kernel[np.ix_(val, train)], labels[train], a)
                            - labels[val]
                        )
                    )
                    for a in alphas
                ]
                alpha = alphas[int(np.argmin(val_errors))]
                pred = ridge_prediction(kernel[np.ix_(train, train)], kernel[np.ix_(test, train)], labels[train], alpha)
                predictions[name] = pred
                error = np.linalg.norm(pred - labels[test], axis=1)
                result = {
                    "task": task,
                    "representation": name,
                    "alpha_selected_on_val": alpha,
                    "test_episodes": [eps[i]["raw_id"] for i in test],
                    "n_train": len(train),
                    "n_val": len(val),
                    "n_test": len(test),
                    "query_counts_train": dict(Counter(queries[train])),
                    "mean_localization_error_px": float(error.mean()),
                    "within_16px_fraction_not_robot_success": float(np.mean(error <= 16)),
                    "episode_error_px": error.tolist(),
                }
                # Shuffle labels within the instruction query to preserve its marginal.
                rng = np.random.default_rng(args.seed)
                null_labels = labels[train].copy()
                for query in np.unique(queries[train]):
                    where = np.flatnonzero(queries[train] == query)
                    null_labels[where] = null_labels[rng.permutation(where)]
                null_pred = ridge_prediction(
                    kernel[np.ix_(train, train)], kernel[np.ix_(test, train)], null_labels, alpha
                )
                result["shuffled_train_label_error_px"] = float(np.linalg.norm(null_pred - labels[test], axis=1).mean())
                results.append(result)
                LOG.info("Probe %s/%s test error %.2f px", task, name, error.mean())
            del cue, current
        for resolution in (4, 8):
            name = f"cue_current_{resolution}"
            comparison = np.linalg.norm(predictions[name] - labels[test], axis=1) - np.linalg.norm(
                predictions[f"current_{resolution}"] - labels[test], axis=1
            )
            next(r for r in results if r["task"] == task and r["representation"] == name)[
                "paired_delta_vs_current_px"
            ] = paired_bootstrap(comparison, args.seed)
    dump_json(args.output_dir / "probes.json", results)
    write_report(args.output_dir)


def vars_as_json(args):
    return {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}


def write_report(output):
    lines = [
        "# Memory grounding diagnostic",
        "",
        "Offline diagnostic on raw training episodes. Probe train/validation/test episodes are disjoint; the released VLA may have trained on all of them. No robot-success claim.",
        "",
    ]
    path = output / "probes.json"
    if path.exists():
        lines += [
            "## Episode-held-out location probes",
            "",
            "Cue-frame probes use privileged frame selection for diagnosis. They do not measure uniform-buffer retrieval or establish policy grounding. Coordinates are recorded subgoal targets, not object-tracking ground truth.",
            "",
            "| Task | Features | Test N | Error (px) | Shuffled-label error (px) |",
            "|---|---|---:|---:|---:|",
        ]
        for r in json.loads(path.read_text()):
            lines.append(
                f"| {r['task']} | {r['representation']} | {r['n_test']} | {r['mean_localization_error_px']:.2f} | {r['shuffled_train_label_error_px']:.2f} |"
            )
    policy = output / "policy_summary.json"
    if policy.exists():
        summary = json.loads(policy.read_text())
        lines += [
            "",
            "## Paired flow error contrasts",
            "",
            "Errors use the first 16 steps and 8 physical action dimensions in normalized delta-action space. Positive deltas indicate that correct memory helps. Confidence intervals resample episodes, not timesteps/noise draws. These are teacher-forced errors, not rollout success.",
            "",
            "| Task | Intervention minus reference | Region | Tau | Episodes | Mean delta MSE | 95% episode bootstrap CI |",
            "|---|---|---|---:|---:|---:|---|",
        ]
        for row in summary["paired_effects"]:
            lines.append(
                f"| {row['task']} | {row['variant']} minus {row['reference']} | {row['region']} | {row['tau']} | {row['n_episodes']} | {row['mean']:.6f} | {row['ci95']} |"
            )
        lines += [
            "",
            "The bypassed branch belongs to a memory-trained checkpoint and may be out of distribution. Its error is not a calibrated non-Markovian sensor. Cue intervals are explicitly recorded proxies; masking the initial cue does not erase later reveals. Resolution interventions change the checkpoint's input distribution.",
        ]
    else:
        lines += ["", "Policy sensitivity and flow-error experiments have not completed in this directory."]
    lines += [
        "",
        "## Interpretation",
        "",
        "- Evidence for readable cached cues: held-out cue+current localization improves over current-only, query-only and shuffled-label controls.",
        "- Evidence for useful policy dependence: cue corruption worsens prediction relative to both intact memory and equal-count non-cue removal, with paired noise held fixed.",
        "- A null cue effect is inconclusive if cue frames are absent, the proxy misses later reveals, or representation quality is inadequate.",
        "- A negative flow-error delta is not proof that memory hurts rollout success; multimodal demonstrations and a memory-trained bypass are confounders.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "probes", "policy", "all", "report"))
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/robomme_preprocessed_data"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/robomme_data_h5"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/memory_grounding/pilot"))
    parser.add_argument("--checkpoint", type=Path, default=Path(DEFAULT_CHECKPOINT))
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--train-episodes", type=int, default=30)
    parser.add_argument("--val-episodes", type=int, default=10)
    parser.add_argument("--test-episodes", type=int, default=10)
    parser.add_argument("--policy-episodes", type=int, default=5)
    parser.add_argument("--noise-draws", type=int, default=4)
    parser.add_argument("--taus", type=float, nargs="+", default=[0.2, 0.6, 0.95])
    parser.add_argument("--uniform-points", type=int, default=6)
    parser.add_argument("--action-samples", type=int, default=2)
    parser.add_argument("--denoising-steps", type=int, default=10)
    parser.add_argument("--cue-annotations", type=Path)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if not (1 <= args.train_episodes <= 60 and 1 <= args.val_episodes <= 20 and 1 <= args.test_episodes <= 20):
        parser.error("Split ranges must be disjoint: train 0:60, validation 60:80, test 80:100")
    if not 1 <= args.policy_episodes <= args.test_episodes:
        parser.error("policy-episodes must be between 1 and test-episodes")
    if args.noise_draws < 1 or args.action_samples < 0 or args.uniform_points < 2 or args.denoising_steps < 1:
        parser.error("Invalid sampling counts")
    if not all(0 < tau <= 1 for tau in args.taus):
        parser.error("Noise times must be in (0,1]")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.stage != "report":
        require_slurm()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage in ("prepare", "all"):
        prepare(args)
    if args.stage in ("probes", "all"):
        run_probes(args)
    if args.stage in ("policy", "all"):
        from scripts.memory_grounding_policy import run_policy

        run_policy(args)
    write_report(args.output_dir)


if __name__ == "__main__":
    main()
