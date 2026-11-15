# MRMG: memory grounding diagnostics

This experiment distinguishes (1) availability of target information in native
cached visual features, (2) a trained policy's dependence on historical evidence,
and (3) the usefulness and confounders of memoryless flow error. It trains small
kernel-ridge probes only; it does not train or alter policy weights.

## Run on the cluster

Submit from the repository root:

```bash
mkdir -p outputs/slurm
sbatch scripts/memory_grounding_diagnostic.slurm
```

The job requests one GPU, eight CPUs, 64 GB and at most two hours. Python heavy
entrypoints refuse to run without both a Slurm job ID and an `srun` step ID. GPU
evaluation additionally checks that JAX sees an actual GPU. BLAS/OpenMP threads
are bounded. Model checkpoint loading and dataset scans take place on compute
nodes, never the login node.

Default checkpoint (already present on this cluster):
`/mnt/data/vhoangth2/pretrained-weights/RoboMME-perceptual-framesamp-modul/79999`.
Override it with `--checkpoint`. It must have a sibling `history_config.txt`
containing `perceptual-framesamp-modul.yaml`, and its own normalization assets.

For a smaller first pilot:

```bash
sbatch scripts/memory_grounding_diagnostic.slurm \
  --train-episodes 30 --val-episodes 10 --test-episodes 10 \
  --policy-episodes 3 --noise-draws 2 --uniform-points 4 --action-samples 1
```

To run only data preparation and representation probes while GPUs are busy:

```bash
sbatch scripts/memory_grounding_probes.slurm
```

This separate CPU job requests four CPUs and 16 GB (the scheduler may adjust the
recorded memory allocation). Results go to `outputs/memory_grounding/probes_<id>`.

Stages can also run separately **inside an allocation**:

```bash
srun .venv/bin/python -m scripts.memory_grounding_diagnostic prepare --output-dir outputs/memory_grounding/custom
srun .venv/bin/python -m scripts.memory_grounding_diagnostic probes --output-dir outputs/memory_grounding/custom
srun .venv/bin/python -m scripts.memory_grounding_diagnostic policy --output-dir outputs/memory_grounding/custom
```

Use the same arguments for resumed policy runs. Fully completed timestamps have
an atomic CSV completion marker and are skipped. The manifest hash and policy
configuration must match. Choose another output directory to change parameters.
The lightweight `report` stage can regenerate Markdown without loading a model.

## Data alignment and splits

Tasks are ButtonUnmask, ButtonUnmaskSwap, PickHighlight and SwingXtimes (the
simulator's spelling). Raw episode IDs and global feature cache IDs are different
namespaces. Initial RGB SHA-256, exact sequence length, instruction and three
robot-state timestamps verify the mapping; filesystem task order is never assumed.

Probe training uses raw episodes 0 through N-1 (default 30), validation starts at
60 (default 10), and diagnostic test starts at 80 (default 10). Hyperparameters
are selected on validation. Split ranges cannot overlap. All these files are
from the original **policy training dataset**: this is an episode-held-out probe
evaluation, not policy test-set generalization. Prior surprise diagnostics used
episodes 0-14; these do not overlap the new diagnostic test episodes.

Decision timestamps are first target-pick subgoal transitions, or the final
button-press transition for SwingXtimes. They are **stage proxies**, not labels
proving that observations are state-aliased or memory is necessary at that instant.
Uniform timestamps and +/-16, +/-8, 0 around the decision are evaluated.

## Cue annotations need visual audit

For ButtonUnmask tasks, the default intervention masks the initially visible
interval determined by the existing RGB overlap closure proxy. It does NOT mask
all subsequent reveals or certify target tracking after swaps. PickHighlight
defaults to the simulator's [10,100) highlight schedule, explicitly marked as an
unverified RGB proxy. SwingXtimes uses an early-window negative/control interval,
not a semantic cue annotation.

Inspect the generated `*_audit.png` contact sheets. To supply reviewed half-open
cue intervals, use `--cue-annotations annotations.json`, for example:

```json
{"ButtonUnmask/80": {"cue_intervals": [[0, 32], [120, 145]]}}
```

This example illustrates the format; its later interval is not an actual label.
Annotations are diagnostic labels only and never policy training supervision.

## Representation probes

Native features are [1,16,2048] (4x4) or [1,64,2048] (8x8). A query-conditioned
kernel ridge probe regresses recorded target subgoal coordinates in the raw
256x256 front-camera image. Instruction color interactions are included for the
ButtonUnmask queries. Models use cue-only, current-only, cue+current and
query-only representations, with within-query shuffled training-label controls.
Training-feature kernel centering is reused for validation/test; regularization
is selected only on validation. There is one example per independent episode.

These are intentionally **privileged cue-frame probes**, not an end-to-end
uniform-buffer retrieval benchmark. Positive results establish that information
can be extracted, not that the policy uses it. Report localization error in
pixels and paired episode bootstrap intervals; the within-16-pixel statistic is
not a robot success rate. Grounded subgoal coordinates can contain annotation
noise and are not exact object segmentation or tracking labels.
In ButtonUnmaskSwap, a single initial frame also omits subsequent swaps; failure
of this probe alone cannot distinguish missing tracking from lost visual detail.

## Policy and flow diagnostics

The released model is loaded without training. The current-observation VLM
prefix is cached once per timestamp and reused across interventions. The backend
first compares cached predictions against the production full-forward FM loss
and production action sampling, and checks that bypass is invariant to memory
content. Failure aborts the run. Validation tolerances allow bfloat16 numerical
differences from differently shaped attention computations.

Conditions are intact memory; explicit modulation bypass; cue token masking;
equal-count non-cue frame masking; replacing cue visual features with a causal,
same-episode non-cue donor while preserving original positions/masks; and uniform
8x64-token history versus the checkpoint's native uniform 32x16-token history.
All conditions keep the current observation and instruction unchanged.

`memory_enabled=False` bypasses the **entire** extra memory RMS normalization,
not just scale/shift. All-masked softmax is forbidden. The production defaults
are unchanged and no checkpoint parameters are added. This memoryless branch
belongs to a memory-trained policy: it may be out of distribution and is NOT a
calibrated memoryless expert or automatic non-Markovian sensor.

Paired flow losses share exactly the same action target, noise draw and noise
time (default 0.2, 0.6, 0.95). These stratified diagnostic times are deliberately
different from the training Beta(1.5,1) sampling; they isolate noise confounding.
Scores use only the first 16 executed action steps and eight physical dimensions,
excluding the 24 padded dimensions. Full-32-dimensional scores are retained as
a confound check. Optional generated action chunks use matched initial noise and
the normal ten denoising steps; changes and demonstration errors are not robot
rollout success. No simulator is launched.

Cue and non-cue ablations preserve the original token slots and positional
encodings; no-op cue removals are excluded from aggregate cue-effect estimates.
The 8x64 condition changes the checkpoint's training input distribution and is
an intervention, not a fair separately trained resolution baseline.

## Artifacts and interpretation

Each job writes `outputs/memory_grounding/job_<id>/`:

- `manifest.json`, `model_changes.patch`: verified mapping, labels, proxy
  provenance, split definitions, arguments and Git revision.
- `*_audit.png`: cue/closure/decision frames for inspection.
- `probes.json`: held-out localization, shuffled controls and bootstrap deltas.
- `backend_validation.json`: cached/full-forward checks and GPU device.
- `points/*.csv`, `*.npz`, `*.json`: raw paired errors, velocity/action arrays,
  exact memory selection and intervention metadata for every timestamp.
- `policy_summary.json`, `report.md`: episode-level aggregate results and limits.

Look for consistent cue+current probe improvements over current-only, query-only
and shuffled controls. For grounding, require cue interventions to worsen
predictions more than equal-count non-cue interventions. High error or action
sensitivity alone is insufficient: improvement must concern correct decisions.
Bootstrap over episodes, never over correlated timestamps or Monte Carlo draws.
An association between base error and (base error minus memory error) is partly
mathematically induced; the descriptive correlation is not evidence that the
sensor detects memory necessity. Inspect results by noise level and task.
