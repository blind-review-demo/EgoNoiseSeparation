from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from learning2hear import RecurGraphConfig, run_recurgraph


def parse_args() -> argparse.Namespace:
    defaults = RecurGraphConfig()
    parser = argparse.ArgumentParser(
        description="Run RecurGraph on precomputed PE-AV audio embeddings."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--embedding-key", default="audio_embeds")
    parser.add_argument("--group-column", default="robot")
    parser.add_argument("--seed-ratio", type=float, default=defaults.seed_ratio)
    parser.add_argument("--pca-dim", type=int, default=defaults.pca_dim)
    parser.add_argument("--n-neighbors", type=int, default=defaults.n_neighbors)
    parser.add_argument("--alpha", type=float, default=defaults.alpha)
    parser.add_argument(
        "--selection-threshold",
        type=float,
        default=defaults.selection_threshold,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.manifest.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError("Manifest is empty")
    with np.load(args.embeddings, allow_pickle=False) as archive:
        embeddings = np.asarray(archive[args.embedding_key], dtype=np.float32)
    if len(rows) != len(embeddings):
        raise ValueError(
            f"Manifest/embedding row mismatch: {len(rows)} != {len(embeddings)}"
        )

    config = RecurGraphConfig(
        seed_ratio=args.seed_ratio,
        pca_dim=args.pca_dim,
        n_neighbors=args.n_neighbors,
        alpha=args.alpha,
        selection_threshold=args.selection_threshold,
    )
    grouped_indices: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        grouped_indices.setdefault(row[args.group_column], []).append(index)

    summaries = {}
    fieldnames = [
        *rows[0].keys(),
        "embedding_centroid_similarity",
        "seed_label",
        "ego_noise_dominant_score",
        "selected_for_training",
    ]
    for group, indices_list in sorted(grouped_indices.items()):
        indices = np.asarray(indices_list, dtype=np.int64)
        result = run_recurgraph(embeddings[indices], config)
        summaries[group] = {
            "samples": len(indices),
            "selected_for_training": int(np.sum(result.selected)),
            "diagnostics": result.diagnostics,
        }
        for local_index, global_index in enumerate(indices):
            rows[int(global_index)].update(
                {
                    "embedding_centroid_similarity": repr(
                        float(result.centroid_similarities[local_index])
                    ),
                    "seed_label": str(int(result.seed_labels[local_index])),
                    "ego_noise_dominant_score": repr(
                        float(result.scores[local_index])
                    ),
                    "selected_for_training": str(
                        int(result.selected[local_index])
                    ),
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_path = args.summary or args.output.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "manifest": str(args.manifest),
                "embeddings": str(args.embeddings),
                "output": str(args.output),
                "parameters": config.__dict__,
                "groups": summaries,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
