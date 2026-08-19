from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from learning2hear.config import TransferDiTSettings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download SAM-Audio weights from Hugging Face."
    )
    parser.add_argument(
        "--model-id",
        default=TransferDiTSettings.model_id,
        help="Hugging Face model id or local model directory.",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=None,
        help="Optional directory to materialize the snapshot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kwargs = {"repo_id": args.model_id}
    if args.local_dir is not None:
        args.local_dir.mkdir(parents=True, exist_ok=True)
        kwargs["local_dir"] = str(args.local_dir)
    path = snapshot_download(**kwargs)
    print(path)


if __name__ == "__main__":
    main()
