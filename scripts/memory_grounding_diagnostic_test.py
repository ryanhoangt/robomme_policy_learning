"""Small CPU-only tests; no model imports or real dataset traversal."""

import numpy as np
import pytest

from scripts import memory_grounding_diagnostic as diagnostic


def test_uniform_history_includes_anchor_and_never_uses_future():
    assert diagnostic.uniform_indices(0, 32) == [0]
    assert diagnostic.uniform_indices(3, 32) == [0, 1, 2, 3]
    indices = diagnostic.uniform_indices(300, 32)
    assert len(indices) == 32
    assert indices[0] == 0
    assert indices[-1] == 300
    assert indices == sorted(set(indices))


def test_cue_ablation_preserves_other_frames_and_has_count_matched_control():
    times = np.repeat([0, 10, 20, 30, 40, -1], 2)
    mask = times >= 0
    cue_mask, cue_removed = diagnostic.corrupt_mask(mask, times, [[0, 11]])
    control_mask, control_removed = diagnostic.corrupt_mask(mask, times, [[0, 11]], control=True)
    assert cue_removed == [0, 10]
    assert control_removed == [20, 30]
    assert np.sum(mask & ~cue_mask) == np.sum(mask & ~control_mask) == 4
    assert np.array_equal(mask, times >= 0)
    assert not cue_mask[times == -1].any()
    assert control_mask[times == 40].all()


def test_all_masked_attention_is_rejected_and_no_cue_is_a_noop():
    with pytest.raises(ValueError, match="all-masked"):
        diagnostic.corrupt_mask([True, True], [0, 1], [[0, 2]])
    mask, removed = diagnostic.corrupt_mask([True, True, False], [10, 20, -1], [[0, 2]])
    assert np.array_equal(mask, [True, True, False])
    assert removed == []


def test_error_excludes_padding_and_unexecuted_action_tail():
    target = np.zeros((2, 20, 32))
    prediction = target.copy()
    prediction[:, :, 8:] = 1000
    prediction[:, 16:, :8] = 1000
    assert np.array_equal(diagnostic.physical_mse(prediction, target), [0, 0])
    prediction[:, :16, :8] = 2
    assert np.array_equal(diagnostic.physical_mse(prediction, target), [4, 4])


def test_decision_labels_are_stage_proxies_and_coordinates_are_yx():
    stages = [
        (0, "press the button", "press the button at <44, 55>"),
        (80, "pick up the container hiding the red cube", "pick up the container at <123, 72>"),
    ]
    decision = diagnostic.decision_from_stages("ButtonUnmask", stages)
    assert decision["step"] == 80
    assert decision["label_yx"] == [123, 72]
    with pytest.raises(ValueError, match="Cannot identify"):
        diagnostic.decision_from_stages("PickHighlight", stages[:1])


def test_probe_kernel_centers_using_train_statistics_and_fits_intercept():
    train = np.array([[0.0, 0], [1, 0], [0, 1], [1, 1]])
    heldout = np.array([[0.25, 0.75], [0.75, 0.25]])
    targets = train @ np.array([[2.0], [3.0]]) + 7
    prediction = diagnostic.ridge_prediction(train @ train.T, heldout @ train.T, targets, 1e-8)
    np.testing.assert_allclose(prediction, heldout @ np.array([[2.0], [3.0]]) + 7, atol=1e-6)


def test_bootstrap_unit_is_episode_and_is_reproducible():
    result = diagnostic.paired_bootstrap([1, 1, 1], seed=5)
    assert result == {"n_episodes": 3, "mean": 1.0, "ci95": [1.0, 1.0]}
    assert diagnostic.paired_bootstrap([1])["ci95"] is None


def test_heavy_entrypoint_requires_a_slurm_step(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_STEP_ID", raising=False)
    with pytest.raises(SystemExit, match="login node"):
        diagnostic.require_slurm()
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    with pytest.raises(SystemExit):
        diagnostic.require_slurm()
    monkeypatch.setenv("SLURM_STEP_ID", "0")
    diagnostic.require_slurm()


def test_flow_contrasts_pair_noise_and_bootstrap_episodes_not_individual_rows():
    rows = []
    for episode in (80, 81):
        for draw in (0, 1):
            for variant, error in (("correct", 1.0), ("cue_mask", 4.0), ("noncue_mask", 2.0)):
                rows.append(
                    {
                        "task": "ButtonUnmask",
                        "raw_id": str(episode),
                        "step": "100",
                        "tau": "0.95",
                        "noise_draw": str(draw),
                        "variant": variant,
                        "flow_mse": error + draw,
                        "cue_tokens_available": "16",
                        "region": "decision_window_proxy",
                    }
                )
    effects = diagnostic.aggregate_flow_effects(rows)
    contrast = next(
        r for r in effects if r["variant"] == "cue_mask" and r["reference"] == "noncue_mask" and r["region"] == "all"
    )
    assert contrast["n_episodes"] == 2
    assert contrast["mean"] == 2
    assert contrast["ci95"] == [2, 2]
