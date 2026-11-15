#!/bin/bash
#SBATCH --job-name=mrmg-probes
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/slurm/mrmg_probes_%j.out
#SBATCH --error=outputs/slurm/mrmg_probes_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mrmg-probes-${SLURM_JOB_ID}"

srun .venv/bin/python -m pytest -q scripts/memory_grounding_diagnostic_test.py
srun .venv/bin/python -m scripts.memory_grounding_diagnostic prepare \
    --output-dir "outputs/memory_grounding/probes_${SLURM_JOB_ID}" "$@"
srun .venv/bin/python -m scripts.memory_grounding_diagnostic probes \
    --output-dir "outputs/memory_grounding/probes_${SLURM_JOB_ID}" "$@"
