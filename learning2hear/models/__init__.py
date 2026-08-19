__all__ = [
    "LayerCrossResidualAdapter",
    "TransferDiT",
    "TransferDiTTransformer",
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
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
