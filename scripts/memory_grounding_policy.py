"""GPU backend for memory_grounding_diagnostic; imported only on compute nodes."""

from __future__ import annotations

import copy
import csv
import dataclasses
import hashlib
import json
import logging
import pickle

import einops
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from mme_vla_suite.models.config.utils import get_history_config
from mme_vla_suite.models.integration.history_observation import HistAugObservation
from mme_vla_suite.models.integration.history_observation import preprocess_observation
from mme_vla_suite.models.integration.history_pi0 import make_attn_mask
from mme_vla_suite.shared.mem_buffer import MemoryBuffer
from mme_vla_suite.training import config as training_config
from openpi import transforms
from openpi.models import model as base_model
from openpi.shared import nnx_utils
from openpi.training import checkpoints
from scripts.memory_grounding_diagnostic import aggregate_flow_effects
from scripts.memory_grounding_diagnostic import corrupt_mask
from scripts.memory_grounding_diagnostic import dump_json
from scripts.memory_grounding_diagnostic import feature
from scripts.memory_grounding_diagnostic import interval_mask
from scripts.memory_grounding_diagnostic import physical_mse
from scripts.memory_grounding_diagnostic import require_slurm
from scripts.memory_grounding_diagnostic import uniform_indices
from scripts.memory_grounding_diagnostic import vars_as_json
from scripts.memory_grounding_diagnostic import write_report

LOG = logging.getLogger(__name__)


class DiagnosticRunner(nnx.Module):
    def __init__(self, model):
        self.model = model

    def prefix(self, observation):
        obs = preprocess_observation(None, observation, train=False)
        tokens, mask, ar_mask, _, _ = self.model.embed_prefix(obs)
        positions = jnp.cumsum(mask, axis=1) - 1
        _, cache = self.model.PaliGemma.llm([tokens, None], positions=positions, mask=make_attn_mask(mask, ar_mask))
        return mask, cache

    def memory(self, observation):
        tokens, mask, _, _, _ = self.model.embed_memory(observation)
        return tokens, mask

    def velocity(self, observation, prefix_mask, cache, memory, memory_mask, noisy_actions, time, enabled):
        tokens, mask, ar_mask, _, cond = self.model.embed_suffix(observation, noisy_actions, time)
        suffix_attention = make_attn_mask(mask, ar_mask)
        prefix_attention = einops.repeat(prefix_mask, "b p -> b s p", s=tokens.shape[1])
        attention = jnp.concatenate([prefix_attention, suffix_attention], axis=-1)
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(mask, axis=-1) - 1
        (_, suffix), _ = self.model.PaliGemma.llm(
            [None, tokens],
            positions=positions,
            mask=attention,
            kv_cache=cache,
            adarms_cond=[None, cond],
            mem_seq=[None, memory],
            mem_mask=[None, memory_mask],
            memory_enabled=enabled,
        )
        return self.model.action_out_proj(suffix[:, -self.model.action_horizon :])

    def rollout(self, observation, prefix_mask, cache, memory, memory_mask, noise, enabled, *, steps=10):
        dt = -1.0 / steps

        def step(index, actions):
            time = jnp.full((actions.shape[0],), 1.0 + index * dt)
            velocity = self.velocity(observation, prefix_mask, cache, memory, memory_mask, actions, time, enabled)
            return actions + dt * velocity

        return jax.lax.fori_loop(0, steps, step, noise)


def load_model(args):
    config_file = args.checkpoint.parent / "history_config.txt"
    if not config_file.exists() or config_file.read_text().strip() != "perceptual-framesamp-modul.yaml":
        raise ValueError("This diagnostic requires a released perceptual-framesamp-modul checkpoint")
    cfg = training_config.get_config("mme_vla_suite")
    cfg = dataclasses.replace(
        cfg, model=dataclasses.replace(cfg.model, use_history=True, history_config=config_file.read_text().strip())
    )
    LOG.info("Loading trained checkpoint %s", args.checkpoint)
    model = cfg.model.load(base_model.restore_params(args.checkpoint / "params", dtype=jnp.bfloat16))
    model.eval()
    norm_stats = checkpoints.load_norm_stats(args.checkpoint / "assets", "robomme")
    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    transform = transforms.compose(
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ]
    )
    return model, transform, norm_stats, get_history_config(config_file.read_text().strip())


def build_memories(args, episode, step, history_config, norm_stats):
    buffer = MemoryBuffer(num_views=1, img_emb_dim=2048, pos_emb_dim=768, state_emb_dim=8)
    cached = {}

    def gather(indices):
        for t in indices:
            if t not in cached:
                cached[t] = feature(args.dataset_dir, episode["cache_id"], t)
        return {t: cached[t] for t in indices}

    def construct(tokens_per_frame):
        images, positions, states, mask = buffer.prepare_frame_sampling(step, 512, tokens_per_frame, gather)
        states = (states - norm_stats["state"].q01) / (norm_stats["state"].q99 - norm_stats["state"].q01 + 1e-6) * 2 - 1
        indices = uniform_indices(step, 512 // tokens_per_frame)
        times = np.repeat(indices + [-1] * (512 // tokens_per_frame - len(indices)), tokens_per_frame)
        return {
            "static_image_emb": images.astype(np.float32),
            "static_pos_emb": positions.astype(np.float32),
            "static_state_emb": states.astype(np.float32),
            "static_mask": mask,
        }, times

    correct, times = construct(history_config.token_per_image)
    result = {"correct": (correct, True), "bypass": (correct, False)}
    metadata = {
        "sampled_frames": sorted(set(times[times >= 0].tolist())),
        "cue_frames_removed": [],
        "noncue_frames_removed": [],
    }
    metadata["cue_tokens_available"] = int(
        np.sum(interval_mask(times, episode["cue_intervals"]) & correct["static_mask"])
    )
    for name, control in (("cue_mask", False), ("noncue_mask", True)):
        try:
            mask, removed = corrupt_mask(correct["static_mask"], times, episode["cue_intervals"], control=control)
        except ValueError as error:
            metadata[f"{name}_unavailable"] = str(error)
            continue
        modified = dict(correct, static_mask=mask)
        result[name] = (modified, True)
        metadata["noncue_frames_removed" if control else "cue_frames_removed"] = removed
    # A content intervention preserves token count, original masks, and original
    # positional embeddings. Its donor is from this episode and never the future.
    cue_positions = interval_mask(times, episode["cue_intervals"]) & correct["static_mask"]
    donors = [t for t in range(step + 1) if not interval_mask([t], episode["cue_intervals"])[0]]
    if donors and cue_positions.any():
        donor = min(donors, key=lambda t: abs(t - episode["cue_intervals"][0][1]))
        values = feature(args.dataset_dir, episode["cache_id"], donor)["image_emb_4x4"][0].astype(np.float32)
        modified = dict(correct, static_image_emb=correct["static_image_emb"].copy())
        for position in np.flatnonzero(cue_positions):
            modified["static_image_emb"][position] = values[position % 16]
        result["cue_replace"] = (modified, True)
        metadata["cue_replacement_donor"] = donor
    fine, _ = construct(64)
    result["uniform_8x64"] = (fine, True)
    return result, metadata


def make_observation(sample, memory, transform):
    payload = copy.deepcopy(sample)
    for key in (
        "static_image_emb",
        "static_pos_emb",
        "static_state_emb",
        "static_mask",
        "recur_image_emb",
        "recur_pos_emb",
        "recur_state_emb",
        "recur_mask",
    ):
        payload[key] = memory.get(key)
    transformed = transform(payload)
    target = np.asarray(transformed.pop("actions"), dtype=np.float32)[None]
    batch = jax.tree.map(lambda value: jnp.asarray(value)[None], transformed)
    return HistAugObservation.from_dict(batch), target


def validate_backend(runner, observation, prefix, memory, target, args):
    """Compare cached evaluation to the production full-forward FM objective.

    Also prove the bypass is invariant to memory content, and preserves default
    checkpoint parameter shapes. No parameter training is performed.
    """
    from openpi.shared import nnx_utils

    key = jax.random.key(args.seed)
    _, noise_key, time_key = jax.random.split(key, 3)
    noise = jax.random.normal(noise_key, target.shape)
    time = jax.random.beta(time_key, 1.5, 1, target.shape[:-2]) * 0.999 + 0.001
    noisy = time[:, None, None] * noise + (1 - time[:, None, None]) * target
    expected_loss, _ = nnx_utils.module_jit(runner.model.compute_loss, static_argnames=("train",))(
        key, observation, jnp.asarray(target), train=False
    )
    velocity = nnx_utils.module_jit(runner.velocity)
    prediction = velocity(observation, *prefix, *memory, noisy, time, enabled=True)
    actual_loss = np.mean(np.square(np.asarray(prediction) - (np.asarray(noise) - target)), axis=-1)
    np.testing.assert_allclose(actual_loss, np.asarray(expected_loss), rtol=0.03, atol=0.01)
    disabled = velocity(observation, *prefix, *memory, noisy, time, enabled=False)
    altered = velocity(observation, *prefix, jnp.zeros_like(memory[0]), memory[1], noisy, time, enabled=False)
    np.testing.assert_array_equal(np.asarray(disabled), np.asarray(altered))
    expected_actions = nnx_utils.module_jit(runner.model.sample_actions)(
        key, observation, noise=noise, num_steps=args.denoising_steps
    )
    actual_actions = nnx_utils.module_jit(runner.rollout, static_argnames=("steps",))(
        observation, *prefix, *memory, noise, enabled=True, steps=args.denoising_steps
    )
    np.testing.assert_allclose(np.asarray(actual_actions), np.asarray(expected_actions), rtol=0.03, atol=0.03)
    return {
        "cached_flow_max_abs_loss_difference": float(np.max(np.abs(actual_loss - np.asarray(expected_loss)))),
        "cached_sampling_max_abs_action_difference": float(
            np.max(np.abs(np.asarray(actual_actions) - np.asarray(expected_actions)))
        ),
        "bypass_memory_invariance": True,
        "jax_devices": [str(d) for d in jax.devices()],
    }


def sample_points(episode, uniform_points):
    start, end = episode["exec_start"], episode["length"] - 1
    decision = episode["decision"]["step"]
    points = set(np.linspace(start, end, uniform_points, dtype=int).tolist())
    points.update(min(end, max(start, decision + offset)) for offset in (-16, -8, 0, 8, 16))
    return sorted(points)


def run_policy(args):
    require_slurm()
    if not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError("Policy diagnostic requires a Slurm GPU allocation")
    episodes = json.loads((args.output_dir / "manifest.json").read_text())["episodes"]
    episodes = [e for e in episodes if e["split"] == "probe_test" and e["raw_id"] < 80 + args.policy_episodes]
    provenance = vars_as_json(args)
    provenance.pop("stage")
    provenance["manifest_sha256"] = hashlib.sha256((args.output_dir / "manifest.json").read_bytes()).hexdigest()
    config_path = args.output_dir / "policy_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != provenance:
        raise ValueError("Output directory contains a different policy configuration; choose a new output-dir")
    dump_json(config_path, provenance)
    model, transform, norm, history = load_model(args)
    runner = DiagnosticRunner(model)
    prefix_fn = nnx_utils.module_jit(runner.prefix)
    memory_fn = nnx_utils.module_jit(runner.memory)
    velocity_fn = nnx_utils.module_jit(runner.velocity)
    rollout_fn = nnx_utils.module_jit(runner.rollout, static_argnames=("steps",))
    validated = False
    points_dir = args.output_dir / "points"
    points_dir.mkdir(exist_ok=True)
    for ep in episodes:
        for step in sample_points(ep, args.uniform_points):
            stem = f"{ep['task']}_ep{ep['raw_id']}_t{step}"
            done = points_dir / f"{stem}.csv"
            if done.exists():
                LOG.info("Resume: skipping complete %s", stem)
                continue
            sample_index = ep["sample_start"] + step - ep["exec_start"]
            with (args.dataset_dir / "data" / f"{sample_index}.pkl").open("rb") as f:
                sample = pickle.load(f)
            if (
                int(np.asarray(sample["step_idx"]).item()) != step
                or int(np.asarray(sample["epis_idx"]).item()) != ep["cache_id"]
            ):
                raise ValueError("Sample index does not refer to the verified cache episode/timestep")
            memories, meta = build_memories(args, ep, step, history, norm)
            obs, target = make_observation(sample, memories["correct"][0], transform)
            prefix = prefix_fn(obs)
            encoded = memory_fn(obs)
            if not validated:
                dump_json(
                    args.output_dir / "backend_validation.json",
                    validate_backend(runner, obs, prefix, encoded, target, args),
                )
                validated = True
            # Seeds are independent of iteration order, intervention, noise time,
            # and resuming. Every intervention sees the identical noisy action.
            seed = int.from_bytes(hashlib.sha256(f"{args.seed}/{stem}".encode()).digest()[:8], "little")
            rng = np.random.default_rng(seed)
            noises = rng.standard_normal((args.noise_draws, *target.shape)).astype(np.float32)
            sampling_noises = rng.standard_normal((args.action_samples, *target.shape)).astype(np.float32)
            rows, outputs, action_outputs = [], {}, {}
            for variant, (raw_memory, enabled) in memories.items():
                observation, variant_target = make_observation(sample, raw_memory, transform)
                np.testing.assert_array_equal(target, variant_target)
                mem = memory_fn(observation) if variant not in ("correct", "bypass") else encoded
                velocities = []
                for tau in args.taus:
                    per_noise = []
                    for draw, noise in enumerate(noises):
                        noisy = (1 - tau) * target + tau * noise
                        prediction = np.asarray(
                            velocity_fn(
                                observation,
                                *prefix,
                                *mem,
                                jnp.asarray(noisy),
                                jnp.array([tau], dtype=jnp.float32),
                                enabled,
                            )
                        )
                        if not np.isfinite(prediction).all():
                            raise ValueError(f"Nonfinite flow prediction in {stem}/{variant}")
                        flow_target = noise - target
                        rows.append(
                            {
                                "task": ep["task"],
                                "raw_id": ep["raw_id"],
                                "step": step,
                                "region": "decision_window_proxy"
                                if abs(step - ep["decision"]["step"]) <= 16
                                else "other",
                                "variant": variant,
                                "tau": tau,
                                "noise_draw": draw,
                                "flow_mse": float(physical_mse(prediction, flow_target)[0]),
                                "flow_mse_full32": float(np.square(prediction - flow_target).mean()),
                                "cue_tokens_available": meta["cue_tokens_available"],
                            }
                        )
                        per_noise.append(prediction[0, :16, :8])
                    velocities.append(per_noise)
                outputs[variant] = np.array(velocities)
                generated = []
                for noise in sampling_noises:
                    actions = rollout_fn(
                        observation, *prefix, *mem, jnp.asarray(noise), enabled, steps=args.denoising_steps
                    )
                    generated.append(np.asarray(actions)[0, :16, :8])
                action_outputs[variant] = np.asarray(generated)
            arrays = {f"velocity_{name}": value for name, value in outputs.items()}
            arrays.update({f"actions_{name}": value for name, value in action_outputs.items()})
            arrays["gt_normalized_delta_actions"] = target[0, :16, :8]
            for name, generated in action_outputs.items():
                if name == "correct" or not len(generated):
                    continue
                meta[f"{name}_action_rms_change"] = float(
                    np.sqrt(np.mean(np.square(generated - action_outputs["correct"])))
                )
                meta[f"{name}_action_error_minus_correct"] = float(
                    physical_mse(generated, target[0]).mean()
                    - physical_mse(action_outputs["correct"], target[0]).mean()
                )
            if not all(np.isfinite(value).all() for value in arrays.values()):
                raise ValueError("Nonfinite diagnostic output")
            np.savez_compressed(points_dir / f"{stem}.npz", **arrays)
            dump_json(points_dir / f"{stem}.json", dict(meta, task=ep["task"], raw_id=ep["raw_id"], step=step))
            temporary = done.with_suffix(".csv.tmp")
            with temporary.open("w") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            temporary.replace(done)  # completion marker written last
            LOG.info("Completed %s: %s conditions, %s cue tokens", stem, len(memories), meta["cue_tokens_available"])
    summarize_policy(args.output_dir, args.seed)
    write_report(args.output_dir)


def summarize_policy(output, seed):
    rows, metadata = [], []
    for path in sorted((output / "points").glob("*.csv")):
        with path.open() as f:
            rows.extend(csv.DictReader(f))
        metadata.append(json.loads(path.with_suffix(".json").read_text()))
    correct = {
        (r["task"], r["raw_id"], r["step"], r["tau"], r["noise_draw"]): float(r["flow_mse"])
        for r in rows
        if r["variant"] == "correct"
    }
    effects = aggregate_flow_effects(rows, seed)
    # Measure association rather than declaring high error to be an ambiguity label.
    associations = []
    for task in sorted({r["task"] for r in rows}):
        for tau in sorted({r["tau"] for r in rows}, key=float):
            pairs = {}
            for row in rows:
                if row["task"] == task and row["tau"] == tau and row["variant"] == "bypass":
                    sample = (row["task"], row["raw_id"], row["step"], row["tau"], row["noise_draw"])
                    pairs.setdefault((row["raw_id"], row["step"]), []).append(
                        (float(row["flow_mse"]), float(row["flow_mse"]) - correct[sample])
                    )
            means = np.array([np.mean(p, axis=0) for p in pairs.values()])
            corr = (
                float(np.corrcoef(means.T)[0, 1]) if len(means) > 2 and (np.std(means, axis=0) > 1e-10).all() else None
            )
            associations.append(
                {
                    "task": task,
                    "tau": float(tau),
                    "n_timestamps": len(means),
                    "base_error_vs_memory_gain_pearson_descriptive_only": corr,
                    "warning": "Shared base-error term mechanically induces correlation; not sensor validation.",
                }
            )
    dump_json(
        output / "policy_summary.json",
        {
            "paired_effects": effects,
            "point_metadata": metadata,
            "sensor_associations": associations,
            "scope": "Offline training-data diagnostic; no robot rollouts or success rates; bypass may be out of distribution.",
        },
    )
