from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path
from typing import Any

import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from learning2hear.config import TransferDiTSettings


def add_sam_audio_to_path(repo: Path | None) -> None:
    candidates = [
        repo,
        Path(os.environ["SAM_AUDIO_REPO"]) if "SAM_AUDIO_REPO" in os.environ else None,
        REPO_ROOT / "third_party" / "sam-audio",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            sys.path.insert(0, str(candidate))
            return


def patch_ranker_creation() -> None:
    sam_model = importlib.import_module("sam_audio.model.model")

    sam_model.create_ranker = lambda cfg: None


def load_checkpoint(
    path: Path,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = torch.load(path, map_location=device)
    if isinstance(payload, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            value = payload.get(key)
            if isinstance(value, dict):
                args = payload.get("args", {})
                return value, args if isinstance(args, dict) else {}
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported checkpoint payload type: {type(payload)!r}")
    return payload, {}


def save_audio(path: Path, audio: torch.Tensor, sample_rate: int) -> None:
    audio = audio.detach().cpu().float()
    while audio.ndim > 2 and audio.shape[0] == 1:
        audio = audio.squeeze(0)
    array = audio.numpy()
    if array.ndim == 2:
        array = array.T
    sf.write(path, array, sample_rate)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Separate audio with Transfer-DiT.")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--model-id", default=TransferDiTSettings.model_id)
    parser.add_argument("--description", default="walking ego-noise")
    parser.add_argument("--sam-audio-repo", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ode-method", default=TransferDiTSettings.ode_method)
    parser.add_argument(
        "--ode-step-size",
        type=float,
        default=TransferDiTSettings.ode_step_size,
    )
    parser.add_argument("--strict-checkpoint", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    add_sam_audio_to_path(args.sam_audio_repo)
    patch_ranker_creation()

    from sam_audio import SAMAudioProcessor
    from learning2hear import TransferDiT
    from learning2hear.config import TransferDiTSettings
    from learning2hear.models.lora import (
        inject_lora_into_transformer,
        lora_rank_from_state_dict,
    )
    from learning2hear.models.transfer_dit import match_arch_cross_adapter_rank

    device = torch.device(args.device)
    processor = SAMAudioProcessor.from_pretrained(args.model_id)
    model = TransferDiT.from_pretrained(args.model_id).to(device).eval()

    if args.checkpoint is not None:
        state_dict, checkpoint_args = load_checkpoint(args.checkpoint, device)
        match_arch_cross_adapter_rank(model, state_dict)
        lora_rank = lora_rank_from_state_dict(state_dict)
        if lora_rank is not None:
            target_value = checkpoint_args.get(
                "lora_targets",
                ",".join(TransferDiTSettings.lora_targets),
            )
            target_names = {
                name.strip()
                for name in str(target_value).split(",")
                if name.strip()
            }
            inject_lora_into_transformer(
                model,
                rank=int(checkpoint_args.get("lora_r", lora_rank)),
                alpha=float(
                    checkpoint_args.get(
                        "lora_alpha",
                        TransferDiTSettings.lora_alpha,
                    )
                ),
                dropout=float(
                    checkpoint_args.get(
                        "lora_dropout",
                        TransferDiTSettings.lora_dropout,
                    )
                ),
                target_names=target_names,
            )
        unmatched = sorted(set(state_dict) - set(model.state_dict()))
        if unmatched:
            raise RuntimeError(f"Checkpoint keys did not match the model: {unmatched}")
        load_result = torch.nn.Module.load_state_dict(
            model,
            state_dict,
            strict=False,
        )
        if args.strict_checkpoint:
            missing_adapters = [
                key
                for key in load_result.missing_keys
                if ".lora_a." in key
                or ".lora_b." in key
                or ".arch_cross_" in key
            ]
            if missing_adapters:
                raise RuntimeError(
                    "Checkpoint is missing adapter parameters: "
                    f"{missing_adapters}"
                )

    batch = processor(
        descriptions=[args.description],
        audios=[str(args.audio)],
    )
    batch = batch.to(device)

    ode_opt = {
        "method": args.ode_method,
        "options": {"step_size": args.ode_step_size},
    }
    with torch.inference_mode():
        result = model.separate(
            batch,
            ode_opt=ode_opt,
            reranking_candidates=1,
            predict_spans=False,
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sample_rate = int(model.sample_rate)
    save_audio(args.out_dir / "target.wav", result.target[0], sample_rate)
    save_audio(args.out_dir / "residual.wav", result.residual[0], sample_rate)
    print(args.out_dir)


if __name__ == "__main__":
    main()
