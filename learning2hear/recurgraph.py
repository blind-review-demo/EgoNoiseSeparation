from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.semi_supervised import LabelSpreading as _GraphPropagator

from learning2hear.config import RecurGraphSettings


def l2_normalize(array: np.ndarray, axis: int = -1) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    denominator = np.linalg.norm(values, axis=axis, keepdims=True)
    return values / np.maximum(denominator, np.finfo(np.float32).eps)


@dataclass(frozen=True)
class RecurGraphConfig:
    """Configuration used for the paper's RecurGraph experiments."""

    pca_dim: int = RecurGraphSettings.pca_dim
    pca_random_state: int = RecurGraphSettings.pca_random_state
    seed_ratio: float = RecurGraphSettings.seed_ratio
    n_neighbors: int = RecurGraphSettings.n_neighbors
    alpha: float = RecurGraphSettings.alpha
    max_iter: int = RecurGraphSettings.max_iter
    tol: float = RecurGraphSettings.tol
    n_jobs: int = RecurGraphSettings.n_jobs
    selection_threshold: float = RecurGraphSettings.selection_threshold


@dataclass(frozen=True)
class RecurGraphResult:
    scores: np.ndarray
    selected: np.ndarray
    seed_labels: np.ndarray
    centroid_similarities: np.ndarray
    class_probabilities: np.ndarray
    embedding_centroid: np.ndarray
    diagnostics: dict[str, Any]


def _validate_embeddings(audio_embeddings: np.ndarray) -> np.ndarray:
    embeddings = np.asarray(audio_embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] < 2:
        raise ValueError(
            "audio_embeddings must contain at least two rows: "
            f"shape={embeddings.shape}"
        )
    if not np.all(np.isfinite(embeddings)):
        raise ValueError("audio_embeddings must be finite")
    return l2_normalize(embeddings, axis=1)


def embedding_centroid(audio_embeddings: np.ndarray) -> np.ndarray:
    """Return the normalized mean of a robot's adaptation-clip embeddings."""

    embeddings = _validate_embeddings(audio_embeddings)
    anchor = np.mean(embeddings, axis=0, keepdims=True)
    if float(np.linalg.norm(anchor)) <= np.finfo(np.float32).eps:
        raise ValueError("The embedding centroid has zero norm")
    return l2_normalize(anchor, axis=1)[0]


def _seed_labels(
    similarities: np.ndarray,
    seed_ratio: float,
) -> tuple[np.ndarray, float, float]:
    rho = float(seed_ratio)
    if not 0.0 < rho < 0.5:
        raise ValueError(f"seed_ratio must be in (0, 0.5), got {rho}")
    lower = float(np.quantile(similarities, rho))
    upper = float(np.quantile(similarities, 1.0 - rho))
    negative = similarities <= lower
    positive = similarities >= upper
    if np.any(negative & positive):
        raise ValueError("Positive and negative seed sets overlap")
    labels = np.full(len(similarities), -1, dtype=np.int64)
    labels[negative] = 0
    labels[positive] = 1
    return labels, lower, upper


def _graph_features(
    embeddings: np.ndarray,
    centroid: np.ndarray,
    config: RecurGraphConfig,
) -> np.ndarray:
    combined = np.concatenate([embeddings, centroid[None, :]], axis=0)
    dimension = min(
        int(config.pca_dim),
        combined.shape[0] - 1,
        combined.shape[1],
    )
    if dimension < 1:
        raise ValueError(f"Not enough data for PCA: shape={combined.shape}")
    projected = PCA(
        n_components=dimension,
        random_state=int(config.pca_random_state),
    ).fit_transform(combined)
    return l2_normalize(projected[:-1], axis=1)


def run_recurgraph(
    audio_embeddings: np.ndarray,
    config: RecurGraphConfig | None = None,
) -> RecurGraphResult:
    """Estimate ego-noise-dominant scores for one robot's clip set.

    Inputs must be PE-AV audio embeddings from a single robot/recording scope.
    The returned score is the propagated probability of the positive
    (ego-noise-dominant) class.
    """

    cfg = config or RecurGraphConfig()
    embeddings = _validate_embeddings(audio_embeddings)
    centroid = embedding_centroid(embeddings)
    similarities = np.asarray(embeddings @ centroid, dtype=np.float32)
    labels, lower, upper = _seed_labels(similarities, cfg.seed_ratio)
    features = _graph_features(embeddings, centroid, cfg)

    neighbor_count = min(int(cfg.n_neighbors), len(features) - 1)
    if neighbor_count < 1:
        raise ValueError("RecurGraph requires at least two clips")
    if not 0.0 < float(cfg.alpha) < 1.0:
        raise ValueError("alpha must be in (0, 1)")

    estimator = _GraphPropagator(
        kernel="knn",
        n_neighbors=neighbor_count,
        alpha=float(cfg.alpha),
        max_iter=int(cfg.max_iter),
        tol=float(cfg.tol),
        n_jobs=int(cfg.n_jobs),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        estimator.fit(features, labels)

    classes = np.asarray(estimator.classes_, dtype=np.int64)
    if classes.tolist() != [0, 1]:
        raise ValueError(f"Unexpected propagated classes: {classes.tolist()}")
    probabilities = np.asarray(estimator.label_distributions_, dtype=np.float32)
    if probabilities.shape != (len(embeddings), 2):
        raise ValueError(f"Unexpected probability shape: {probabilities.shape}")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Propagated probabilities must be finite")

    scores = probabilities[:, 1]
    selected = scores > float(cfg.selection_threshold)
    return RecurGraphResult(
        scores=scores,
        selected=selected,
        seed_labels=labels,
        centroid_similarities=similarities,
        class_probabilities=probabilities,
        embedding_centroid=centroid,
        diagnostics={
            "lower_seed_threshold": lower,
            "upper_seed_threshold": upper,
            "negative_seed_count": int(np.sum(labels == 0)),
            "positive_seed_count": int(np.sum(labels == 1)),
            "unlabeled_count": int(np.sum(labels == -1)),
            "graph_feature_dimension": int(features.shape[1]),
            "n_neighbors": neighbor_count,
            "n_iter": int(estimator.n_iter_),
            "converged": not any(
                issubclass(item.category, ConvergenceWarning) for item in caught
            ),
        },
    )


def run_recurgraph_by_group(
    audio_embeddings: np.ndarray,
    groups: Sequence[str],
    config: RecurGraphConfig | None = None,
) -> dict[str, RecurGraphResult]:
    """Run RecurGraph independently for every robot/group label."""

    embeddings = np.asarray(audio_embeddings, dtype=np.float32)
    labels = np.asarray(groups, dtype=str)
    if embeddings.ndim != 2 or len(embeddings) != len(labels):
        raise ValueError(
            "audio_embeddings and groups must have matching first dimensions"
        )
    return {
        group: run_recurgraph(embeddings[labels == group], config)
        for group in sorted(set(labels.tolist()))
    }


__all__ = [
    "RecurGraphConfig",
    "RecurGraphResult",
    "embedding_centroid",
    "run_recurgraph",
    "run_recurgraph_by_group",
]
