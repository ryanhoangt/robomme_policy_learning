#!/bin/bash
#SBATCH --job-name=beliefvla-surprise
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/slurm/surprise_%j.out
#SBATCH --error=outputs/slurm/surprise_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the robomme_policy_learning repository root}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/beliefvla-matplotlib-${SLURM_JOB_ID}"
export PYTHONUNBUFFERED=1

printf 'Slurm job %s on %s; CPUs=%s\n' "$SLURM_JOB_ID" "$(hostname)" "$SLURM_CPUS_PER_TASK"
srun .venv/bin/python -m pytest -q scripts/surprise_detector_test.py
srun .venv/bin/python -u scripts/test_surprise_detector.py --threads "$SLURM_CPUS_PER_TASK" "$@"
