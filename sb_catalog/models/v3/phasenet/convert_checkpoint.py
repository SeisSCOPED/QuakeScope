"""
Convert a phasenet-retrain training checkpoint into a SeisBench weight pair.

The Denolle-Lab/phasenet-retrain pipeline saves checkpoints as
    {"epoch", "val_loss", "model": <PhaseNetFinetune state_dict>, ...}
where the deployable student weights carry the "model." prefix and the frozen
distillation teacher carries "teacher.". The student is architecturally
identical to SeisBench's `jma_wc` PhaseNet (its fine-tuning parent).

This script strips the wrapper, validates the weights against the jma_wc
architecture, and writes
    <name>.pt.v1   +   <name>.json.v1
into this directory, which the Dockerfile bakes into the image so picking
jobs can use `--weight <name>`.

Usage (champion v7 checkpoint copied from the lab server):
    python convert_checkpoint.py \
        --checkpoint checkpoints/finetune_jma_wc_global_v7/best.pt \
        --name quakescope2026

Requires: torch, seisbench (and internet or a warm SeisBench cache for the
jma_wc reference download on first use).
"""

import argparse
import json
import shutil
from pathlib import Path

import seisbench.models as sbm
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", required=True, help="Path to the training best.pt"
    )
    parser.add_argument(
        "--name", required=True, help="SeisBench weight name, e.g. quakescope2026"
    )
    parser.add_argument(
        "--outdir",
        default=Path(__file__).parent,
        type=Path,
        help="Output directory (default: this models/phasenet/ directory)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Install into the local SeisBench cache and reload via from_pretrained",
    )
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    raw_sd = ckpt["model"]
    student_sd = {
        k[len("model.") :]: v for k, v in raw_sd.items() if k.startswith("model.")
    }
    print(
        f"Loaded checkpoint: epoch={ckpt.get('epoch', '?')} "
        f"val_loss={ckpt.get('val_loss', float('nan')):.6f}, "
        f"{len(student_sd)} student tensors "
        f"({sum(1 for k in raw_sd if k.startswith('teacher.'))} teacher tensors stripped)"
    )

    # Validate against the parent architecture: raises on any shape mismatch
    reference = sbm.PhaseNet.from_pretrained("jma_wc", update=False)
    reference.load_state_dict(student_sd)
    print("State dict matches the jma_wc PhaseNet architecture.")

    args.outdir.mkdir(parents=True, exist_ok=True)
    pt_path = args.outdir / f"{args.name}.pt.v1"
    json_path = args.outdir / f"{args.name}.json.v1"

    torch.save(student_sd, pt_path)

    # Metadata: reuse jma_wc's (same architecture, norm, sampling rate),
    # overriding only the version and provenance comment.
    jma_json = (
        Path(seisbench_cache_model_path())
        / "phasenet"
        / f"jma_wc.json.v{reference.weights_version}"
    )
    if jma_json.exists():
        metadata = json.loads(jma_json.read_text())
    else:  # fall back to docstring metadata carried on the loaded model
        metadata = getattr(reference, "weights_docstring", None) or {}
        if not isinstance(metadata, dict):
            metadata = {}
    metadata["version"] = "1"
    metadata["docstring"] = (
        f"QuakeScope 2026: phasenet-retrain champion fine-tuned from jma_wc "
        f"(converted from {Path(args.checkpoint).name}, "
        f"epoch {ckpt.get('epoch', '?')})"
    )
    json_path.write_text(json.dumps(metadata, indent=2))
    print(f"Wrote {pt_path}\nWrote {json_path}")

    if args.verify:
        cache_dir = Path(seisbench_cache_model_path()) / "phasenet"
        cache_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(pt_path, cache_dir / pt_path.name)
        shutil.copy(json_path, cache_dir / json_path.name)
        model = sbm.PhaseNet.from_pretrained(args.name)
        print(f"Verified: sbm.PhaseNet.from_pretrained('{args.name}') loads. {model}")


def seisbench_cache_model_path() -> str:
    import seisbench

    return str(Path(seisbench.cache_root) / "models" / "v3")


if __name__ == "__main__":
    main()
