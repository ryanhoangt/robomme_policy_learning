"""Standalone offline BeliefVLA diagnostic on cached native SigLIP tokens.

No VLA training. Compare persistence, linear extrapolation, pixel change, and a
small, optionally trained action-conditioned predictor on held-out episodes.
Run from the repository root; see docs/beliefvla_surprise_diagnostic.md.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import logging
import os
from pathlib import Path
import pickle

import h5py
import matplotlib as mpl
import ml_dtypes  # noqa: F401 -- registers the bfloat16 dtype used by cached arrays
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw

TASKS = ("ButtonUnmask", "ButtonUnmaskSwap")
LOG = logging.getLogger(__name__)


@dataclasses.dataclass
class Episode:
    task: str
    raw_id: int
    cache_id: int
    length: int
    split: str
    prompt: str


def read_feature(root: Path, cache_id: int, step: int) -> dict:
    path = root / "features" / f"episode_{cache_id}" / f"token_emb_{step}.npy"
    return np.load(path, allow_pickle=True).item()


def native_tokens(root: Path, cache_id: int, step: int) -> np.ndarray:
    values = np.asarray(read_feature(root, cache_id, step)["image_emb_8x8"], dtype=np.float32)
    if values.ndim != 3 or values.shape[:2] != (1, 64) or not np.isfinite(values).all():
        raise ValueError(f"Expected finite [1,64,D] front-view cache, got {values.shape}")
    return values[0]


def cache_length(directory: Path) -> int:
    """Length of a contiguous sequence, using O(log T) filesystem probes."""
    if not (directory / "token_emb_0.npy").exists():
        raise FileNotFoundError(f"Missing first frame in {directory}")
    lo, hi = 0, 1
    while (directory / f"token_emb_{hi}.npy").exists():
        lo, hi = hi, hi * 2
    while hi - lo > 1:
        mid = (hi + lo) // 2
        if (directory / f"token_emb_{mid}.npy").exists():
            lo = mid
        else:
            hi = mid
    return hi


def image_digest(image: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def discover_episodes(args) -> list[Episode]:
    """Match raw RGB and cached sample starts; never assume filesystem order."""
    wanted = {}
    for task in TASKS:
        with h5py.File(args.raw_dir / f"record_dataset_{task}.h5", "r") as raw:
            for raw_id in range(args.episodes_per_task + args.train_episodes_per_task):
                group = raw[f"episode_{raw_id}"]
                if bool(group["timestep_0/info/is_video_demo"][()]):
                    raise ValueError("Expected execution-only ButtonUnmask episodes")
                key = image_digest(group["timestep_0/obs/front_rgb"][()])
                if key in wanted:
                    raise ValueError("Duplicate initial RGB; raw/cache mapping is ambiguous")
                length = sum(k.startswith("timestep_") for k in group)
                wanted[key] = (task, raw_id, length)
    stats = json.loads((args.dataset_dir / "meta/stats.json").read_text())
    total = stats.get("execution_samples", stats.get("total_samples"))
    sample_index, result = 0, []
    while sample_index < total and wanted:
        with (args.dataset_dir / "data" / f"{sample_index}.pkl").open("rb") as handle:
            sample = pickle.load(handle)
        cache_id = int(np.asarray(sample["epis_idx"]).item())
        step = int(np.asarray(sample["step_idx"]).item())
        if step != int(np.asarray(sample["exec_start_idx"]).item()):
            raise ValueError(f"Sample {sample_index} is not an episode start; cannot safely map cache")
        length = cache_length(args.dataset_dir / "features" / f"episode_{cache_id}")
        match = wanted.get(image_digest(sample["image"])) if step == 0 else None
        if match:
            task, raw_id, raw_length = match
            if length != raw_length:
                raise ValueError(f"Raw/cache length mismatch for {task}/{raw_id}")
            with h5py.File(args.raw_dir / f"record_dataset_{task}.h5", "r") as raw:
                for t in sorted({0, length // 2, length - 1}):
                    feature = read_feature(args.dataset_dir, cache_id, t)
                    obs = raw[f"episode_{raw_id}/timestep_{t}/obs"]
                    state = np.r_[obs["joint_state"][()], obs["gripper_state"][()][:1]]
                    if not np.allclose(state, feature["state_emb"], rtol=0, atol=1e-5):
                        raise ValueError(f"Raw/cache state mismatch for {task}/{raw_id}, t={t}")
            split = "diagnostic" if raw_id < args.episodes_per_task else "predictor_train"
            result.append(Episode(task, raw_id, cache_id, length, split, sample["prompt"]))
            del wanted[image_digest(sample["image"])]
            LOG.info("Matched %s/%s -> cache episode_%s (%s)", task, raw_id, cache_id, split)
        sample_index += length - step
    if wanted:
        raise ValueError(f"No verified cache match for {list(wanted.values())}")
    return sorted(result, key=lambda e: (e.task, e.raw_id))


def read_raw(args, episode: Episode):
    with h5py.File(args.raw_dir / f"record_dataset_{episode.task}.h5", "r") as raw:
        group = raw[f"episode_{episode.raw_id}"]
        actions, images, boundaries = [], [], []
        for t in range(episode.length):
            step = group[f"timestep_{t}"]
            actions.append(step["action/joint_action"][()])
            images.append(step["obs/front_rgb"][()])
            if bool(step["info/is_subgoal_boundary"][()]):
                boundaries.append({"t": t, "subgoal": step["info/simple_subgoal"][()].decode()})
    return np.asarray(images), np.asarray(actions, dtype=np.float32), boundaries


def color_masks(images: np.ndarray) -> np.ndarray:
    """Independent RGB visibility proxy, never used for fitting or peak selection."""
    channels = images.astype(np.float32)
    maximum = channels.max(axis=-1)
    second = np.partition(channels, -2, axis=-1)[..., -2]
    saturated = (maximum >= 100) & (second < 0.35 * maximum)
    return saturated[..., None] & (channels == maximum[..., None])


def visibility_proxy(images: np.ndarray) -> tuple[np.ndarray, int | None]:
    """Initial colored-pixel overlap at fixed coordinates; NOT ground truth/tracking."""
    initial = color_masks(images[:1])[0]
    size = np.count_nonzero(initial)
    if size < 8:
        raise ValueError("Cannot identify initial colored cue; inspect RGB")
    visibility = np.array([np.count_nonzero(color_masks(im[None])[0] & initial) / size for im in images])
    hidden = np.flatnonzero(np.convolve((visibility < 0.25).astype(int), np.ones(3, dtype=int), "valid") == 3)
    return visibility, int(hidden[0]) if len(hidden) else None


def minmax(values: np.ndarray) -> np.ndarray:
    result = np.full_like(values, np.nan, dtype=np.float32)
    valid = np.isfinite(values)
    if valid.any():
        lo, hi = values[valid].min(), values[valid].max()
        result[valid] = (values[valid] - lo) / (hi - lo) if hi > lo else 0
    return result


def select_nms(scores: np.ndarray, count: int, radius: int) -> list[int]:
    """Retrospective whole-episode top-K, not an online commit simulation."""
    valid = np.flatnonzero(np.isfinite(scores))
    if not len(valid) or np.ptp(scores[valid]) == 0:
        return []
    order = valid[np.argsort(-scores[valid], kind="stable")]
    selected = []
    for step in order:
        if all(abs(int(step) - previous) > radius for previous in selected):
            selected.append(int(step))
            if len(selected) == count:
                break
    return selected


def raised_cosine_targets(scores: np.ndarray, peaks: list[int], radius: int) -> np.ndarray:
    """Offline max-composition of raised cosines, preserving [0,1] and peak heights."""
    normalized = minmax(scores)
    target = np.zeros(len(scores), dtype=np.float32)
    for peak in peaks:
        offsets = np.arange(max(0, peak - radius), min(len(scores), peak + radius + 1))
        weights = 0.5 * (1 + np.cos(np.pi * (offsets - peak) / radius)) if radius else np.ones(1)
        target[offsets] = np.maximum(target[offsets], normalized[peak] * weights)
    return target


def patch_reductions(errors: np.ndarray, prefix: str) -> dict[str, np.ndarray]:
    return {
        f"{prefix}_max": np.max(errors, axis=-1),
        f"{prefix}_top4": np.mean(np.sort(errors, axis=-1)[:, -4:], axis=-1),
        f"{prefix}_mean": np.mean(errors, axis=-1),
    }


def baseline_scores(tokens, images, gaps):
    """Persistence is a zero-order predictor, distinct from a learned world model."""
    curves, patches = {}, {}
    normed = tokens / np.maximum(np.linalg.norm(tokens, axis=-1, keepdims=True), 1e-8)
    pixels = images.astype(np.float32) / 255
    for gap in gaps:
        error = np.full(tokens.shape[:2], np.nan, dtype=np.float32)
        error[gap:] = np.sum(np.square(tokens[gap:] - tokens[:-gap]), axis=-1)
        curves.update(patch_reductions(error, f"persist_d{gap}"))
        patches[f"persist_d{gap}"] = error
        cosine = np.full_like(error, np.nan)
        cosine[gap:] = np.maximum(0, 1 - np.sum(normed[gap:] * normed[:-gap], axis=-1))
        curves[f"cosine_d{gap}_max"] = np.max(cosine, axis=-1)
        linear = np.full_like(error, np.nan)
        linear[2 * gap :] = np.sum(np.square(tokens[2 * gap :] - 2 * tokens[gap:-gap] + tokens[: -2 * gap]), axis=-1)
        curves[f"linear_d{gap}_max"] = np.max(linear, axis=-1)
        difference = np.abs(pixels[gap:] - pixels[:-gap])
        height, width = images.shape[1:3]
        difference = difference.reshape(-1, 8, height // 8, 8, width // 8, 3).mean(axis=(2, 4, 5))
        pixel_curve = np.full(len(tokens), np.nan, dtype=np.float32)
        pixel_curve[gap:] = difference.max(axis=(1, 2))
        curves[f"pixel_d{gap}_max"] = pixel_curve
    return curves, patches


def make_predictor(dim: int, context: int, gap: int, hidden: int):
    """Three causal layers per patch; no cross-patch attention in this small control."""
    import torch
    from torch import nn

    class PatchPredictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = nn.Linear(dim, hidden)
            self.action = nn.Linear(gap * 8, hidden)
            self.position = nn.Embedding(64, hidden)
            self.time = nn.Parameter(torch.zeros(1, context, hidden))
            layer = nn.TransformerEncoderLayer(hidden, 4, hidden * 2, dropout=0, batch_first=True)
            self.temporal = nn.TransformerEncoder(layer, 3, enable_nested_tensor=False)
            self.out = nn.Linear(hidden, dim)
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)
            self.register_buffer("mask", torch.triu(torch.ones(context, context, dtype=torch.bool), diagonal=1))

        def forward(self, history, actions, patch):
            x = self.visual(history) + self.position(patch)[:, None] + self.time
            x = x.clone()
            x[:, -1] = x[:, -1] + self.action(actions)
            return history[:, -1] + self.out(self.temporal(x, mask=self.mask)[:, -1])

    return PatchPredictor()


def train_predictor(args, episodes):
    import torch

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    histories, targets, paths, patch_ids = [], [], [], []
    minimum = args.predict_gap + (args.context - 1) * args.context_stride
    for episode in episodes:
        if episode.split != "predictor_train":
            continue
        with h5py.File(args.raw_dir / f"record_dataset_{episode.task}.h5", "r") as raw:
            actions = np.stack(
                [raw[f"episode_{episode.raw_id}/timestep_{t}/action/joint_action"][()] for t in range(episode.length)]
            ).astype(np.float32)
        for t in rng.integers(minimum, episode.length, size=args.train_windows_per_episode):
            patch = rng.choice(64, size=8, replace=False)
            indices = t - args.predict_gap - np.arange(args.context - 1, -1, -1) * args.context_stride
            history = np.stack([native_tokens(args.dataset_dir, episode.cache_id, int(i))[patch] for i in indices])
            histories.append(history.transpose(1, 0, 2))
            targets.append(native_tokens(args.dataset_dir, episode.cache_id, int(t))[patch])
            paths.append(np.repeat(actions[t - args.predict_gap : t].reshape(1, -1), len(patch), axis=0))
            patch_ids.append(patch)
        LOG.info("Prepared predictor samples: %s/%s", episode.task, episode.raw_id)
    x, y, actions = [np.concatenate(values) for values in (histories, targets, paths)]
    ids = np.concatenate(patch_ids)
    scale = float(np.sqrt(np.mean(np.square(x), dtype=np.float64)))
    action_mean, action_std = actions.mean(axis=0), np.maximum(actions.std(axis=0), 1e-6)
    x, y = torch.from_numpy(x / scale), torch.from_numpy(y / scale)
    actions, ids = torch.from_numpy((actions - action_mean) / action_std), torch.from_numpy(ids)
    model = make_predictor(y.shape[-1], args.context, args.predict_gap, args.hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.predictor_lr)
    history_log = []
    for step in range(args.predictor_steps):
        indices = torch.from_numpy(rng.integers(0, len(x), size=args.batch_size))
        loss = torch.square(model(x[indices], actions[indices], ids[indices]) - y[indices]).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        if step % 50 == 0 or step == args.predictor_steps - 1:
            history_log.append({"step": step, "normalized_mse": float(loss.detach())})
            LOG.info("Predictor %d/%d loss=%.6f", step + 1, args.predictor_steps, float(loss.detach()))
    model.eval()
    calibration = {"scale": scale, "action_mean": action_mean, "action_std": action_std}
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    torch.save(
        {"model": model.state_dict(), "calibration": calibration, "config": config},
        args.output_dir / "surprise_predictor.pt",
    )
    return model, calibration, history_log


def learned_errors(args, model, calibration, tokens, actions):
    import torch

    minimum = args.predict_gap + (args.context - 1) * args.context_stride
    error = np.full(tokens.shape[:2], np.nan, dtype=np.float32)
    scale = calibration["scale"]
    with torch.inference_mode():
        for start in range(minimum, len(tokens), 16):
            times = np.arange(start, min(start + 16, len(tokens)))
            indices = times[:, None] - args.predict_gap - np.arange(args.context - 1, -1, -1) * args.context_stride
            x = tokens[indices].transpose(0, 2, 1, 3).reshape(-1, args.context, tokens.shape[-1]) / scale
            a = np.stack([actions[t - args.predict_gap : t].reshape(-1) for t in times])
            a = np.repeat((a - calibration["action_mean"]) / calibration["action_std"], 64, axis=0)
            prediction = model(torch.from_numpy(x), torch.from_numpy(a), torch.arange(64).repeat(len(times)))
            prediction = prediction.numpy().reshape(len(times), 64, -1) * scale
            error[times] = np.square(tokens[times] - prediction).sum(axis=-1)
    return error


def selection_record(selected: list[int], cue_end: int | None, radius: int) -> dict:
    return {
        "selected": selected,
        "initial_cue_hit": any(t < cue_end for t in selected) if cue_end is not None else None,
        "closure_hit": any(abs(t - cue_end) <= radius for t in selected) if cue_end is not None else None,
        "cue_frames_selected": sum(t < cue_end for t in selected) if cue_end is not None else None,
    }


def evaluate_selection(curves, length, cue_end, args):
    result = {}
    for name, values in curves.items():
        selected = select_nms(values, args.slots, args.nms_radius)
        result[name] = {
            **selection_record(selected, cue_end, args.label_radius),
            "valid_frames": int(np.isfinite(values).sum()),
        }
    for count in sorted({args.slots, 32}):
        selected = np.linspace(0, length - 1, count, dtype=int).tolist()
        result[f"uniform_{count}"] = selection_record(selected, cue_end, args.label_radius)
    result["initial_anchor"] = selection_record([0], cue_end, args.label_radius)
    return result


def save_episode_plots(args, episode, images, curves, patches, visibility, cue_end, boundaries):
    key = f"{episode.task}_ep{episode.raw_id}"
    fig, axes = plt.subplots(2, 1, figsize=(13, 5), sharex=True, constrained_layout=True)
    displayed = [
        f"persist_d{args.predict_gap}_max",
        f"cosine_d{args.predict_gap}_max",
        f"pixel_d{args.predict_gap}_max",
        "wm_max",
        "wm_top4",
    ]
    for name in displayed:
        if name in curves:
            axes[0].plot(minmax(curves[name]), label=name, linewidth=0.9)
    if cue_end is not None:
        axes[0].axvspan(0, cue_end - 1, color="green", alpha=0.12, label="initial visible-cube proxy")
        axes[0].axvline(cue_end, color="red", linestyle="--", linewidth=1, label="closure proxy")
    for boundary in boundaries:
        if boundary["t"]:
            axes[0].axvline(boundary["t"], color="gray", alpha=0.35, linestyle=":")
    axes[0].set(title=f"{key} | min-max normalization for display only", ylabel="Normalized score")
    axes[0].legend(fontsize=7, ncol=3)
    patch_key = "wm" if "wm" in patches else f"persist_d{args.predict_gap}"
    heat = patches[patch_key].T
    axes[1].imshow(
        heat,
        origin="lower",
        aspect="auto",
        extent=(-0.5, len(images) - 0.5, -0.5, 63.5),
        vmin=0,
        vmax=np.quantile(heat[np.isfinite(heat)], 0.99),
        cmap="magma",
    )
    axes[1].set(xlabel="Raw timestep (score aligned to target observation)", ylabel=f"Patch ID ({patch_key})")
    fig.savefig(args.output_dir / f"{key}.png", dpi=130)
    plt.close(fig)

    chosen = []
    for source in (f"persist_d{args.predict_gap}", "wm"):
        if source in patches:
            chosen.extend((source, t) for t in select_nms(curves[f"{source}_max"], args.slots, args.nms_radius))
    frames = [
        ("anchor", 0),
        ("last visible proxy", max(0, (cue_end or 1) - 1)),
        ("closure proxy", cue_end or 0),
        *chosen,
    ]
    canvas = Image.new("RGB", (len(frames) * 192, 222), "white")
    draw = ImageDraw.Draw(canvas)
    for col, (source, t) in enumerate(frames):
        canvas.paste(Image.fromarray(images[t]).resize((192, 192)), (col * 192, 30))
        draw.text((col * 192 + 2, 2), f"t={t}: {source}", fill="black")
        draw.text((col * 192 + 2, 15), f"initial color overlap={visibility[t]:.2f}", fill="black")
        if source in patches:
            patch = int(np.nanargmax(patches[source][t]))
            x, y = (patch % 8) * 24 + col * 192, (patch // 8) * 24 + 30
            draw.rectangle((x, y, x + 24, y + 24), outline="yellow", width=2)
    canvas.save(args.output_dir / f"{key}_frames.png")


def write_summary(args, results, training_log):
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    summary = {
        "config": config,
        "annotation": "Initial saturated-color overlap proxy; inspect saved RGB, not ground truth.",
        "selection": "Retrospective whole-episode top-K NMS, not online thresholded commits.",
        "predictor": "3-layer patch-wise causal Transformer, no cross-patch attention, native latent output.",
        "training_log": training_log,
        "episodes": results,
    }
    (args.output_dir / "surprise_metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    lines = [
        "# Surprise signal diagnostic",
        "",
        "Initial-cue capture under retrospective selection; RGB proxy labels.",
        "",
        "| Method | Task | Cue captured | Closure selected |",
        "|---|---|---:|---:|",
    ]
    for method in results[0]["selection"]:
        for task in TASKS:
            rows = [r["selection"][method] for r in results if r["task"] == task]
            valid = [r for r in rows if r["initial_cue_hit"] is not None]
            hit, closure = sum(r["initial_cue_hit"] for r in valid), sum(r["closure_hit"] for r in valid)
            lines.append(f"| {method} | {task} | {hit}/{len(valid)} | {closure}/{len(valid)} |")
    lines += [
        "",
        "Closure is information loss, not successful evidence capture. Uniform controls include frame 0.",
        "Warm-up frames are NaN, not zero-surprise observations. Min-max normalization and raised-cosine",
        "labels use the complete offline episode; they are not calibrated online probabilities.",
        "Only predictor training episodes fit model weights and action/feature normalization.",
        "Diagnostic episodes are held out from fitting, but are not a benchmark evaluation split.",
        "No VLA training, memory integration, or benchmark rollouts have been performed.",
    ]
    (args.output_dir / "surprise_summary.md").write_text("\n".join(lines) + "\n")


def main(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "episode_manifest.json"
    source = {
        "dataset_dir": str(args.dataset_dir.resolve()),
        "raw_dir": str(args.raw_dir.resolve()),
        "diagnostic_per_task": args.episodes_per_task,
        "train_per_task": args.train_episodes_per_task,
    }
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["source"] != source:
            raise ValueError("Existing manifest uses different inputs/splits; use a new output directory")
        episodes = [Episode(**row) for row in manifest["episodes"]]
    else:
        episodes = discover_episodes(args)
        manifest_path.write_text(
            json.dumps({"source": source, "episodes": [dataclasses.asdict(e) for e in episodes]}, indent=2)
        )
    model, calibration, training_log = None, None, []
    if args.predictor_steps:
        model, calibration, training_log = train_predictor(args, episodes)
    results, overview = [], []
    with (args.output_dir / "surprise_scores.csv").open("w", newline="") as csv_handle:
        writer = None
        for episode in episodes:
            if episode.split != "diagnostic":
                continue
            LOG.info("Scoring %s/%s (%d frames)", episode.task, episode.raw_id, episode.length)
            images, actions, boundaries = read_raw(args, episode)
            visibility, cue_end = visibility_proxy(images)
            tokens = np.stack([native_tokens(args.dataset_dir, episode.cache_id, t) for t in range(episode.length)])
            curves, patches = baseline_scores(tokens, images, sorted({*args.gaps, args.predict_gap}))
            motion = np.full(episode.length, np.nan, dtype=np.float32)
            motion[1:] = np.linalg.norm(np.diff(actions[:, :7], axis=0), axis=-1)
            curves["joint_target_change"] = motion
            quality = {}
            if model is not None:
                error = learned_errors(args, model, calibration, tokens, actions)
                curves.update(patch_reductions(error, "wm"))
                patches["wm"] = error
                valid = np.isfinite(error[:, 0])
                persistence = patches[f"persist_d{args.predict_gap}"][valid]
                quality = {
                    "wm_patch_squared_l2": float(error[valid].mean()),
                    "persistence_patch_squared_l2_same_frames": float(persistence.mean()),
                    "wm_to_persistence_error_ratio": float(error[valid].mean() / persistence.mean()),
                }
            selection = evaluate_selection(curves, episode.length, cue_end, args)
            primary = "wm_max" if model is not None else f"persist_d{args.predict_gap}_max"
            soft = raised_cosine_targets(curves[primary], selection[primary]["selected"], args.label_radius)
            arrays = {
                **curves,
                **{f"patch_{k}": v for k, v in patches.items()},
                "soft_labels": soft,
                "initial_color_overlap": visibility,
            }
            np.savez_compressed(args.output_dir / f"{episode.task}_ep{episode.raw_id}_scores.npz", **arrays)
            if writer is None:
                writer = csv.writer(csv_handle)
                writer.writerow(["task", "raw_episode", "t", "initial_color_overlap", *curves])
            for t in range(episode.length):
                writer.writerow(
                    [
                        episode.task,
                        episode.raw_id,
                        t,
                        float(visibility[t]),
                        *[float(v[t]) if np.isfinite(v[t]) else "" for v in curves.values()],
                    ]
                )
            save_episode_plots(args, episode, images, curves, patches, visibility, cue_end, boundaries)
            results.append(
                {
                    **dataclasses.asdict(episode),
                    "token_shape": list(tokens.shape),
                    "initial_cue_end_proxy": cue_end,
                    "subgoal_boundaries": boundaries,
                    "predictor_quality": quality,
                    "selection": selection,
                }
            )
            overview.append((episode, curves, cue_end))
    fig, axes = plt.subplots(
        args.episodes_per_task, 2, figsize=(17, 3 * args.episodes_per_task), squeeze=False, constrained_layout=True
    )
    for episode, curves, cue_end in overview:
        ax = axes[episode.raw_id, TASKS.index(episode.task)]
        for name in (f"persist_d{args.predict_gap}_max", f"pixel_d{args.predict_gap}_max", "wm_max"):
            if name in curves:
                ax.plot(minmax(curves[name]), linewidth=0.8, label=name)
        if cue_end is not None:
            ax.axvspan(0, cue_end - 1, color="green", alpha=0.15)
            ax.axvline(cue_end, color="red", linestyle="--", linewidth=0.8)
        ax.set(
            title=f"{episode.task} ep{episode.raw_id}: visible-cue proxy [0,{cue_end})",
            xlabel="Raw timestep",
            ylabel="Normalized surprise",
        )
        ax.legend(fontsize=7, loc="upper right")
    fig.savefig(args.output_dir / "surprise_curve_sanity.png", dpi=150)
    plt.close(fig)
    write_summary(args, results, training_log)
    LOG.info("Saved diagnostic to %s", args.output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/robomme_preprocessed_data"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/robomme_data_h5"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--episodes-per-task", type=int, default=5)
    parser.add_argument("--train-episodes-per-task", type=int, default=10)
    parser.add_argument("--gaps", type=int, nargs="+", default=[1, 8, 16])
    parser.add_argument("--predict-gap", type=int, default=1)
    parser.add_argument("--context", type=int, default=4)
    parser.add_argument("--context-stride", type=int, default=1)
    parser.add_argument("--predictor-steps", type=int, default=400, help="0 runs only non-learned controls")
    parser.add_argument("--train-windows-per-episode", type=int, default=24)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--predictor-lr", type=float, default=3e-4)
    parser.add_argument("--slots", type=int, default=4)
    parser.add_argument("--nms-radius", type=int, default=8)
    parser.add_argument("--label-radius", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    positive = [
        args.episodes_per_task,
        args.context,
        args.context_stride,
        args.predict_gap,
        args.train_windows_per_episode,
        args.hidden_dim,
        args.batch_size,
        args.slots,
        args.threads,
        *args.gaps,
    ]
    if (
        min(positive) <= 0
        or min(args.train_episodes_per_task, args.predictor_steps, args.nms_radius, args.label_radius) < 0
    ):
        parser.error("Invalid nonpositive size or negative step/radius")
    if args.predictor_steps and not args.train_episodes_per_task:
        parser.error("A learned predictor requires separate training episodes")
    if args.hidden_dim % 4:
        parser.error("--hidden-dim must be divisible by 4 attention heads")
    return args


if __name__ == "__main__":
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_STEP_ID"):
        raise SystemExit("Run on a compute node with srun, or submit scripts/surprise_diagnostic.slurm.")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main(parse_args())
