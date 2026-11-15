#!/bin/bash
#SBATCH --job-name=mrmg-policy-resume
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=outputs/slurm/mrmg_policy_%j.out
#SBATCH --error=outputs/slurm/mrmg_policy_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mrmg-matplotlib-${SLURM_JOB_ID}"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export JAX_COMPILATION_CACHE_DIR="${SLURM_SUBMIT_DIR}/outputs/memory_grounding/jax_cache"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/home/vhoangth2/.cache/openpi}"

srun --gres=gpu:1 --gpu-bind=none .venv/bin/python scripts/memory_grounding_gpu_check.py
srun --gres=gpu:1 --gpu-bind=none .venv/bin/python -m scripts.memory_grounding_diagnostic policy "$@"
