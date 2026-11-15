"""GPU visibility preflight. Run only in a GPU Slurm srun step."""

import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_STEP_ID"):
        raise SystemExit("Run this GPU preflight with srun on a compute node.")
    keys = (
        "SLURMD_NODENAME",
        "SLURM_JOB_GPUS",
        "SLURM_STEP_GPUS",
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "CUDA_DEVICE_ORDER",
        "JAX_PLATFORMS",
        "LD_LIBRARY_PATH",
    )
    report = {
        "environment": {k: os.environ.get(k) for k in keys},
        "device_files": [str(p) for p in Path("/dev").glob("nvidia*")],
    }
    for name, command in (
        ("nvidia_smi", ["nvidia-smi", "--query-gpu=index,uuid,name,driver_version", "--format=csv"]),
        (
            "jax",
            [sys.executable, "-c", "import jax; print(jax.devices()); print(jax.numpy.ones(1).block_until_ready())"],
        ),
        (
            "torch",
            [
                sys.executable,
                "-c",
                "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())",
            ],
        ),
    ):
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=90, check=False)
            report[name] = {"exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
        except (OSError, subprocess.TimeoutExpired) as error:
            report[name] = {"error": str(error)}
    try:
        cuda = ctypes.CDLL("libcuda.so.1")
        count = ctypes.c_int()
        report["driver"] = {
            "cuInit": cuda.cuInit(0),
            "cuDeviceGetCount": cuda.cuDeviceGetCount(ctypes.byref(count)),
            "count": count.value,
        }
    except OSError as error:
        report["driver"] = {"error": str(error)}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
