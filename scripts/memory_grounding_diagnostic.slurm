#!/bin/bash
#SBATCH --job-name=mrmg-diagnostic
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=outputs/slurm/mrmg_%j.out
#SBATCH --error=outputs/slurm/mrmg_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mrmg-matplotlib-${SLURM_JOB_ID}"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
# Use the allocated GPU(s) provided by Slurm; never override CUDA_VISIBLE_DEVICES.
export JAX_COMPILATION_CACHE_DIR="${SLURM_SUBMIT_DIR}/outputs/memory_grounding/jax_cache"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/home/vhoangth2/.cache/openpi}"

srun .venv/bin/python -m pytest -q scripts/memory_grounding_diagnostic_test.py
srun --gres=gpu:1 --gpu-bind=none .venv/bin/python scripts/memory_grounding_gpu_check.py
srun --gres=gpu:1 --gpu-bind=none .venv/bin/python -m scripts.memory_grounding_diagnostic all \
    --output-dir "outputs/memory_grounding/job_${SLURM_JOB_ID}" "$@"
