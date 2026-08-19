from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

import torch


class LoRALinear(torch.nn.Module):
    """Linear layer with the LoRA parameterization used during training."""

    def __init__(
        self,
        base: torch.nn.Linear,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.rank = int(rank)
        self.scaling = float(alpha) / float(rank)
        self.dropout = (
            torch.nn.Dropout(float(dropout))
            if float(dropout) > 0.0
            else torch.nn.Identity()
        )
        self.lora_a = torch.nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = torch.nn.Linear(rank, base.out_features, bias=False)
        self.lora_a.to(device=base.weight.device, dtype=base.weight.dtype)
        self.lora_b.to(device=base.weight.device, dtype=base.weight.dtype)
        torch.nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_b.weight)
        self.base.requires_grad_(False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.lora_b(self.lora_a(self.dropout(inputs)))
        return self.base(inputs) + residual * self.scaling


def _replace_module(
    root: torch.nn.Module,
    module_name: str,
    new_module: torch.nn.Module,
) -> None:
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def inject_lora_into_transformer(
    model: torch.nn.Module,
    *,
    rank: int,
    alpha: float,
    dropout: float,
    target_names: Iterable[str],
) -> list[str]:
    transformer = getattr(model, "transformer", None)
    if not isinstance(transformer, torch.nn.Module):
        raise TypeError("model.transformer must be a torch module")
    targets = {str(name) for name in target_names}
    replaced = []
    for module_name, module in list(transformer.named_modules()):
        if not module_name or not isinstance(module, torch.nn.Linear):
            continue
        leaf_name = module_name.rsplit(".", 1)[-1]
        if targets and leaf_name not in targets and module_name not in targets:
            continue
        _replace_module(
            transformer,
            module_name,
            LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout),
        )
        replaced.append(module_name)
    if not replaced:
        raise RuntimeError("No transformer linear layers matched the LoRA targets")
    return replaced


def lora_rank_from_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> int | None:
    ranks = {
        int(value.shape[0])
        for key, value in state_dict.items()
        if key.endswith(".lora_a.weight") and value.ndim == 2
    }
    if not ranks:
        return None
    if len(ranks) != 1:
        raise ValueError(f"Checkpoint contains inconsistent LoRA ranks: {sorted(ranks)}")
    return ranks.pop()


__all__ = ["LoRALinear", "inject_lora_into_transformer", "lora_rank_from_state_dict"]
