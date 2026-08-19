from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from learning2hear.config import EgoGraphSettings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract PE-AV embeddings for an EgoGraph manifest."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audio-column", default="mix_path")
    parser.add_argument("--model", default=EgoGraphSettings.pe_av_model)
    parser.add_argument("--batch-size", type=int, default=EgoGraphSettings.pe_av_batch_size)
    parser.add_argument("--pe-av-repo", type=Path, default=None)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def add_pe_av_to_path(repo: Path | None) -> None:
    configured = os.environ.get("PE_AV_REPO")
    candidates = [
        repo,
        Path(configured) if configured else None,
        REPO_ROOT / "third_party" / "perception_models",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            sys.path.insert(0, str(candidate))
            return
    raise FileNotFoundError(
        "PE-AV source not found. Run scripts/setup_pe_av.sh or pass --pe-av-repo."
    )


def peak_normalize(input_values: torch.Tensor) -> torch.Tensor:
    max_abs = input_values.abs().amax(dim=(-2, -1), keepdim=True)
    eps = torch.finfo(input_values.dtype).eps
    scale = torch.where(
        max_abs > eps,
        float(EgoGraphSettings.waveform_peak) / max_abs.clamp_min(eps),
        torch.ones_like(max_abs),
    )
    return input_values * scale


def main() -> None:
    args = parse_args()
    add_pe_av_to_path(args.pe_av_repo)
    from core.audio_visual_encoder import PEAudioVisual, PEAudioVisualTransform

    with args.manifest.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    audio_paths = [row[args.audio_column] for row in rows]
    if not audio_paths:
        raise ValueError("Manifest is empty")
    device = torch.device(args.device)
    model = PEAudioVisual.from_config(args.model, pretrained=True).to(device).eval()
    transform = PEAudioVisualTransform.from_config(args.model)

    parts = []
    for start in range(0, len(audio_paths), args.batch_size):
        batch_paths = audio_paths[start : start + args.batch_size]
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            inputs = transform(audio=batch_paths).to(device)
            inputs["input_values"] = peak_normalize(inputs["input_values"])
            embeddings = model.encode_audio(
                inputs["input_values"],
                padding_mask=inputs.get("padding_mask"),
                input_features=inputs.get("input_features"),
            ).float()
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        parts.append(embeddings.cpu().numpy())
        completed = min(start + len(batch_paths), len(audio_paths))
        print(f"encoded_audio={completed}/{len(audio_paths)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        audio_embeds=np.concatenate(parts, axis=0).astype(np.float32),
    )
    print(args.output)


if __name__ == "__main__":
    main()
