"""Research-integrity checks for episode alignment, temporal causality, and labels."""

import argparse
import json
import pickle

import h5py
import numpy as np
import pytest

from scripts import test_surprise_detector as detector


def test_scores_align_to_target_frame_and_preserve_warmup():
    tokens = np.zeros((8, 64, 3), dtype=np.float32)
    tokens[4:, 17] = 2
    images = np.zeros((8, 16, 16, 3), dtype=np.uint8)
    curves, patches = detector.baseline_scores(tokens, images, [1, 2])
    assert np.isnan(curves["persist_d2_max"][:2]).all()
    assert np.isnan(curves["linear_d2_max"][:4]).all()
    assert curves["persist_d1_max"][4] == 12
    assert np.argmax(patches["persist_d1"][4]) == 17
    assert curves["persist_d1_max"][5] == 0
    assert curves["persist_d1_top4"][4] == 3


def test_nms_skips_warmup_and_soft_labels_remain_bounded():
    scores = np.array([np.nan, 0, 10, 9, 0, 0, 8, 0], dtype=np.float32)
    selected = detector.select_nms(scores, 2, 2)
    assert selected == [2, 6]
    labels = detector.raised_cosine_targets(scores, selected, 2)
    assert labels[2] == 1
    assert np.isclose(labels[6], 0.8)
    assert labels[4] == 0
    assert np.isfinite(labels).all()
    assert labels.min() >= 0
    assert labels.max() <= 1
    assert detector.select_nms(np.ones(10), 4, 2) == []
    assert np.array_equal(detector.minmax(np.ones(10)), np.zeros(10))


def test_closure_is_not_counted_as_information_capture():
    images = np.zeros((10, 16, 16, 3), dtype=np.uint8)
    images[:4, 4:8, 4:8, 0] = 255
    images[7:, 4:8, 4:8, 0] = 255  # later re-reveal must not extend the initial cue
    visibility, end = detector.visibility_proxy(images)
    assert end == 4
    assert visibility[0] == 1
    assert visibility[4] == 0
    record = detector.selection_record([4, 8], end, 1)
    assert record["closure_hit"]
    assert not record["initial_cue_hit"]
    assert detector.selection_record([0], end, 1)["initial_cue_hit"]


@pytest.fixture
def shuffled_cache(tmp_path):
    dataset, raw_dir = tmp_path / "dataset", tmp_path / "raw"
    (dataset / "data").mkdir(parents=True)
    (dataset / "meta").mkdir()
    raw_dir.mkdir()
    # Cached global IDs deliberately reverse raw task/episode order.
    order = [(detector.TASKS[1], 1), (detector.TASKS[0], 0), (detector.TASKS[1], 0), (detector.TASKS[0], 1)]
    samples = 0
    for cache_id, (task, raw_id) in enumerate(order):
        image = np.full((16, 16, 3), cache_id + 1, dtype=np.uint8)
        directory = dataset / "features" / f"episode_{cache_id}"
        directory.mkdir(parents=True)
        length = 5 + raw_id
        with h5py.File(raw_dir / f"record_dataset_{task}.h5", "a") as raw:
            group = raw.create_group(f"episode_{raw_id}")
            for t in range(length):
                step = group.create_group(f"timestep_{t}")
                step["info/is_video_demo"] = False
                step["obs/front_rgb"] = image
                step["obs/joint_state"] = np.full(7, cache_id + t, dtype=np.float32)
                step["obs/gripper_state"] = np.zeros(2, dtype=np.float32)
                state = np.r_[np.full(7, cache_id + t, dtype=np.float32), 0]
                np.save(directory / f"token_emb_{t}.npy", {"state_emb": state})
                with (dataset / "data" / f"{samples}.pkl").open("wb") as handle:
                    pickle.dump(
                        {"epis_idx": cache_id, "step_idx": t, "exec_start_idx": 0, "image": image, "prompt": task},
                        handle,
                    )
                samples += 1
    (dataset / "meta/stats.json").write_text(json.dumps({"execution_samples": samples}))
    return argparse.Namespace(dataset_dir=dataset, raw_dir=raw_dir, episodes_per_task=1, train_episodes_per_task=1)


def test_mapping_matches_rgb_and_state_instead_of_directory_order(shuffled_cache):
    episodes = detector.discover_episodes(shuffled_cache)
    actual = {(e.task, e.raw_id): (e.cache_id, e.split) for e in episodes}
    assert actual[(detector.TASKS[0], 0)] == (1, "diagnostic")
    assert actual[(detector.TASKS[0], 1)] == (3, "predictor_train")
    assert actual[(detector.TASKS[1], 0)] == (2, "diagnostic")
    assert actual[(detector.TASKS[1], 1)] == (0, "predictor_train")


def test_mapping_rejects_misaligned_state(shuffled_cache):
    path = shuffled_cache.dataset_dir / "features/episode_0/token_emb_3.npy"
    np.save(path, {"state_emb": np.full(8, -999)})
    with pytest.raises(ValueError, match="state mismatch"):
        detector.discover_episodes(shuffled_cache)


def test_prediction_is_causal_and_warmup_is_excluded():
    import torch

    torch.set_num_threads(1)
    torch.manual_seed(1)
    args = argparse.Namespace(predict_gap=2, context=3, context_stride=1)
    model = detector.make_predictor(8, args.context, args.predict_gap, 8).eval()
    # Exercise a nonzero learned residual rather than just the persistence initialization.
    torch.nn.init.normal_(model.out.weight, std=0.01)
    rng = np.random.default_rng(7)
    tokens = rng.normal(size=(16, 64, 8)).astype(np.float32)
    actions = rng.normal(size=(16, 8)).astype(np.float32)
    calibration = {
        "scale": 1,
        "action_mean": np.zeros(16, dtype=np.float32),
        "action_std": np.ones(16, dtype=np.float32),
    }
    original = detector.learned_errors(args, model, calibration, tokens, actions)
    tokens[12:] += 100
    actions[12:] -= 100
    perturbed = detector.learned_errors(args, model, calibration, tokens, actions)
    assert np.isnan(original[:4]).all()
    np.testing.assert_allclose(original[4:12], perturbed[4:12])
    assert not np.allclose(original[12:], perturbed[12:])
