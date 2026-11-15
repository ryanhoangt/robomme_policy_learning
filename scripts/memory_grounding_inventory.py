"""Small compute-node inventory for the MRMG diagnostic, not a model run."""

import importlib.metadata
import json
import os
from pathlib import Path


def main():
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_STEP_ID"):
        raise SystemExit("Run inventory in a Slurm srun step.")
    import pickle

    import h5py
    import ml_dtypes  # noqa: F401
    import numpy as np

    result = {"job": os.environ["SLURM_JOB_ID"], "packages": {}, "raw": {}, "checkpoint": []}
    for name in ("jax", "jaxlib", "flax", "orbax-checkpoint", "torch", "numpy", "scikit-learn"):
        try:
            result["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][name] = None
    checkpoint = Path("/mnt/data/vhoangth2/pretrained-weights/RoboMME-perceptual-framesamp-modul")
    result["checkpoint"] = [str(p.relative_to(checkpoint)) for p in checkpoint.glob("*/*")]
    for task in ("ButtonUnmask", "ButtonUnmaskSwap", "PickHighlight", "SwingXtimes"):
        path = Path("data/robomme_data_h5") / f"record_dataset_{task}.h5"
        if not path.exists():
            result["raw"][task] = {"missing": str(path)}
            continue
        with h5py.File(path) as raw:
            ep = raw["episode_0"]
            fields = {}
            for key in ("setup", "timestep_0/info"):
                fields[key] = {}
                for name, obj in ep[key].items():
                    if isinstance(obj, h5py.Dataset):
                        fields[key][name] = str(obj[()])[:1800]
            stages = []
            last = None
            for t in range(sum(k.startswith("timestep_") for k in ep)):
                text = str(ep[f"timestep_{t}/info/grounded_subgoal"][()])
                if text != last:
                    stages.append([t, text])
                    last = text
            result["raw"][task] = {"episodes": len(raw), "fields": fields, "stages": stages}
    with Path("data/robomme_preprocessed_data/data/0.pkl").open("rb") as f:
        sample = pickle.load(f)
    result["sample"] = {k: (list(v.shape) if isinstance(v, np.ndarray) else str(v)) for k, v in sample.items()}
    cache = int(np.asarray(sample["epis_idx"]).item())
    step = int(np.asarray(sample["step_idx"]).item())
    feature = np.load(
        f"data/robomme_preprocessed_data/features/episode_{cache}/token_emb_{step}.npy", allow_pickle=True
    ).item()
    result["feature"] = {k: [list(v.shape), str(v.dtype)] for k, v in feature.items()}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
