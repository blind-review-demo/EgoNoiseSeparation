__all__ = [
    "RecurGraphConfig",
    "RecurGraphResult",
    "LayerCrossResidualAdapter",
    "TransferDiT",
    "TransferDiTTransformer",
    "embedding_centroid",
    "run_recurgraph",
    "run_recurgraph_by_group",
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if name in {
        "RecurGraphConfig",
        "RecurGraphResult",
        "embedding_centroid",
        "run_recurgraph",
        "run_recurgraph_by_group",
    }:
        from learning2hear.recurgraph import (
            RecurGraphConfig,
            RecurGraphResult,
            embedding_centroid,
            run_recurgraph,
            run_recurgraph_by_group,
        )

        return {
            "RecurGraphConfig": RecurGraphConfig,
            "RecurGraphResult": RecurGraphResult,
            "embedding_centroid": embedding_centroid,
            "run_recurgraph": run_recurgraph,
            "run_recurgraph_by_group": run_recurgraph_by_group,
        }[name]
    from learning2hear.models.transfer_dit import (
        LayerCrossResidualAdapter,
        TransferDiT,
        TransferDiTTransformer,
    )

    exports = {
        "LayerCrossResidualAdapter": LayerCrossResidualAdapter,
        "TransferDiT": TransferDiT,
        "TransferDiTTransformer": TransferDiTTransformer,
    }
    return exports[name]
