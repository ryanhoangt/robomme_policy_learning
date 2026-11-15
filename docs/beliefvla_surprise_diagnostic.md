# BeliefVLA: initial surprise diagnostic

This is the first feasibility check, not an implementation of the full BeliefVLA
policy. It does not change the VLA model, its training pipeline, or its memory
buffer. No policy training or benchmark rollouts are scheduled.

## Observed run: Slurm 61181, 2026-10-20

The job completed on `worker-0` in 2 minutes 13 seconds; all six integrity tests
passed. The wrapper requested eight CPUs and 64 GB RAM. Slurm accounting records
eight CPUs and 120 GB for this job (the scheduler's recorded allocation differs
from the requested memory); no GPU was allocated.

All ten RGB closure proxies were at timestep 32. The frame sheets were inspected:
frame 0 and frame 31 show colored cubes, while frame 32 shows covering containers.
Example sheets are `outputs/ButtonUnmask_ep0_frames.png` and
`outputs/ButtonUnmaskSwap_ep0_frames.png`. Several other high-error patches are
on the robot/gripper rather than the hidden-object evidence.

| Retrospective selection, four slots unless noted | ButtonUnmask cue capture | ButtonUnmaskSwap cue capture | Closure selected across both tasks |
|---|---:|---:|---:|
| Persistence patch-max, gap 1 | 0/5 | 0/5 | 10/10 |
| Learned predictor patch-max, gap 1 | 0/5 | 0/5 | 10/10 |
| Learned predictor top-4-patch mean | 0/5 | 0/5 | 10/10 |
| Pixel-change patch-max, gap 1 | 0/5 | 0/5 | 0/10 |
| Uniform 4 frames | 5/5 | 5/5 | 0/10 |
| Uniform 32 frames | 5/5 | 5/5 | 10/10 |
| Initial anchor only, one frame | 5/5 | 5/5 | 0/10 |

The small learned predictor's held-out mean patch error was **2.2–3.4% higher**
than persistence on the same timestamps. It has not demonstrated useful world
modeling, so its failure is not evidence against stronger predictors. Persistence,
cosine change and the small predictor nevertheless expose the same concrete
problem: a peak at *information disappearance* is not a useful evidence frame.

These are initial-cue capture numbers under a visually checked RGB proxy, **not
robot success rates**. There are only five held-out training episodes per task.
The head, online thresholding, and spatial propagator have not been evaluated.
Symmetric raised-cosine targets can extend backwards into the visible interval;
that does not mean an argmax/NMS commit centered at frame 32 captures the cue.
Whether a trained anticipatory head exploits the earlier soft labels is untested.

The proposed explanation that uniform sampling misses this initial cue is not
supported here: the repository's `even_sampling_indices` includes frame 0. A
more plausible alternative to test is spatial information loss: the official
FrameSamp variant pools each frame to 4×4, whereas this proposal retains 8×8.
Compare uniform 8×64 tokens to uniform 32×16 tokens at the same 512-token budget
before attributing an improvement to surprise-based selection.

Recommended next diagnostic: preserve an initial anchor, compare writing the
pre-transition frame (`peak-1`) with writing the peak itself, and inspect whether
the evidence survives 4×4 versus 8×8 pooling. Treat these as new controls, not as
evidence that the proposed surprise gate already works. Defer VLA training and
Module B until the write target is defined as useful evidence rather than a
large visual change.

## Run on Slurm

From the repository root:

```bash
mkdir -p outputs/slurm
sbatch scripts/surprise_diagnostic.slurm
```

The job requests one task, eight CPUs, 64 GB RAM and 30 minutes. No GPU is
requested. Tests and computation both run in `srun` steps on a compute node. The
diagnostic entry point refuses to run without Slurm job and step identifiers.
Logs are `outputs/slurm/surprise_<job_id>.out` and `.err`.

For an allocated interactive session, start the computation using `srun`:

```bash
salloc --ntasks=1 --cpus-per-task=8 --mem=64G --time=00:30:00
srun .venv/bin/python scripts/test_surprise_detector.py --threads 8
```

To run only the non-learned controls in a separate output directory:

```bash
sbatch scripts/surprise_diagnostic.slurm \
    --predictor-steps 0 --train-episodes-per-task 0 \
    --output-dir outputs/surprise_controls
```

## What is measured

Only `ButtonUnmask` and `ButtonUnmaskSwap` are selected. By default, raw episodes
0–4 per task are diagnostic holdouts and episodes 5–14 fit the small predictor.
These are all from the available training HDF5 files, not the benchmark test
split. The holdouts must not be used to tune and then claim an unbiased result.

Raw episode IDs and global cache IDs are different namespaces. The script walks
contiguous preprocessed sample runs, matches initial RGB by SHA-256, checks
sequence length, and independently checks robot state at three timestamps.
It does not assume the original dataset builder's filesystem traversal order.
The mapping and split are saved in `outputs/episode_manifest.json`.

The available native `image_emb_8x8` cache is `[1,64,2048]` with `bfloat16`
features; computation uses float32. `SigLipTokenizer` constructs the OpenPi
SigLIP `So400m/14` encoder. There is no assumed 1024-dimensional SigLIP2 space
and no random projection in this diagnostic.

Controls:

- Persistence prediction: predict the current patch from the patch at `t-gap`.
  Report squared-L2 errors reduced by patch maximum, mean of the four largest
  errors, and mean of all patches. Gaps are 1, 8 and 16 raw timesteps.
- Cosine distance after per-token L2 normalization, reduced by maximum.
- Linear extrapolation: `prediction[t] = 2*o[t-gap] - o[t-2*gap]`.
- Pixel change: largest mean absolute RGB difference on an 8×8 image grid.
- Joint-target change as a motion confound control. This is a change in commanded
  joint targets, not measured robot speed.
- Uniform 4-frame and 32-frame selection, and an initial-frame anchor.

The optional predictor is a three-layer, action-conditioned causal temporal
Transformer **per spatial patch**, with hidden width 64 and a residual output in
the original 2048-dimensional latent space. It has no cross-patch attention and
is a small control, not a validated spatial world model. Training samples 24
windows per training episode and eight patches per window, then fits 400 steps.
All feature scaling and action normalization use only those training samples.
By default it uses four past frames to predict one step ahead. At target `t`,
the last context frame is `t-1` and the action is `a[t-1]`. General gap `D` uses
actions `a[t-D:t]`, excluding the action at target `t`.

Prediction scores are aligned to the **observed target frame**, not the last
context frame. Warm-up frames have undefined surprise (`NaN`), not zero error.
Predictor error is compared to persistence on identical valid frames; a weak
predictor cannot justify rejecting more capable world models.

## Event annotation and a critical task detail

In the ten initially inspected raw episodes, colored cubes are already visible
at frame 0 and become covered around frame 32. This differs from interpreting
the task as a later button press causing the first reveal. The checked-in
simulator calls `lift_and_drop_objects_back_to_original` during the initial
0–64-step window; the helper teleports containers away during its first half.
Exact RGB must take priority over an assumed event schedule.

The script measures overlap with initially visible, strongly saturated RGB
pixels at their original coordinates. The first three consecutive frames with
less than 25% overlap define a **closure proxy**. This is an independently
computed visualization aid, not an object segmentation label, tracking signal,
or exact simulator ground truth. Inspect the contact sheets before interpreting
the numbers. Subgoal boundaries are drawn separately and are never relabeled as
reveal ground truth. A later cube reveal does not extend the initial cue window.

Report two different outcomes:

1. **Initial cue captured:** at least one selected frame lies before the closure
   proxy, while the color-to-container information is still available.
2. **Closure selected:** at least one selected frame lies within ±5 frames of
   closure. This is information loss and must not be scored as a successful write.

Selection is retrospective whole-episode top-4 with greedy temporal suppression
of radius 8. This is an offline ranking diagnostic, not a simulation of an online
thresholded commit policy. Uniform controls include frame 0. They therefore test
whether surprise provides anything beyond the initial anchor, rather than
assuming uniform sampling must miss this task's initial cue.

Per-episode min-max scaling is for plots and provisional offline targets only.
Raised-cosine soft labels use radius 5 and max-composition around selected peaks.
This preserves peak heights and stays in `[0,1]`. Both episode normalization and
symmetric smoothing depend on future observations; neither is an online
calibration procedure. Thresholds for deployment would need training-only
calibration and a separate streaming evaluation.

## Artifacts

- `outputs/surprise_curve_sanity.png`: ten episode curves, with initial visibility
  windows shaded green and closure proxies marked red.
- `outputs/<task>_ep<id>.png`: curves and patch-error heatmap.
- `outputs/<task>_ep<id>_frames.png`: initial cue, closure, and selected frames;
  yellow rectangles locate maximum-error patches.
- `outputs/<task>_ep<id>_scores.npz`: original unnormalized scores, patch errors,
  provisional soft labels, and initial-color-overlap proxy.
- `outputs/surprise_scores.csv`: target-aligned scores, with blank warm-up values.
- `outputs/surprise_metrics.json`: configuration, per-episode metrics and selected
  timestamps, prediction quality, and training log.
- `outputs/surprise_summary.md`: capture/closure comparison table.
- `outputs/surprise_predictor.pt`: small predictor weights and training calibration.

## Implications for the proposed next steps

Surprise measures predictability, not task relevance. A predictable cue can still
be necessary to remember, while a large robot movement or closure can be highly
surprising but provide no new task evidence. Keep the initial-anchor baseline
and inspect *what* a peak contains, not just whether it marks a visual transition.

The neighboring V-JEPA2 checkout constructs its AC predictor with its own
encoder's latent dimension, spatial grid and learned feature space, plus
7-dimensional action/state encoders. Its frozen predictor is not interchangeable
with cached 8×8 SigLIP features and Panda joint-action vectors. Reusing it would
require its own encoding/action convention or a separately validated adapter.

The neighboring EventVLA checkout provides an action-token keyframe head and
raised-cosine label construction. These are useful references for a later head;
the current OpenPi/MME-VLA training path uses JAX/Flax, so a production memory
module should integrate with that path rather than inserting PyTorch parameters
into the JAX optimizer.

Before Module B, resolve the transport convention: a row-softmax destination
query defines a distribution over source patches, not automatically a
source-to-destination physical transition probability. A scalar or row-constant
action bias cancels in softmax. Visually identical containers under occlusion
also make correspondence ambiguous. A whole-scene reveal reconstruction loss
can reward background reconstruction without proving preservation of the hidden
cube's identity. Those are separate hypotheses from the current surprise check.
