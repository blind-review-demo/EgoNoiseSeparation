__all__ = [
    "EgoGraphConfig",
    "EgoGraphResult",
    "LayerCrossResidualAdapter",
    "TransferDiT",
    "TransferDiTTransformer",
    "run_egograph",
    "run_egograph_by_group",
    "self_anchor",
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if name in {
        "EgoGraphConfig",
        "EgoGraphResult",
        "run_egograph",
        "run_egograph_by_group",
        "self_anchor",
    }:
        from learning2hear.egograph import (
            EgoGraphConfig,
            EgoGraphResult,
            run_egograph,
            run_egograph_by_group,
            self_anchor,
        )

        return {
            "EgoGraphConfig": EgoGraphConfig,
            "EgoGraphResult": EgoGraphResult,
            "run_egograph": run_egograph,
            "run_egograph_by_group": run_egograph_by_group,
            "self_anchor": self_anchor,
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
