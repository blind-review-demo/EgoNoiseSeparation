from __future__ import annotations

import unittest

import torch

from learning2hear.models.lora import (
    LoRALinear,
    inject_lora_into_transformer,
    lora_rank_from_state_dict,
)


class ToyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = torch.nn.ModuleDict(
            {
                "wq": torch.nn.Linear(4, 4, bias=False),
                "untouched": torch.nn.Linear(4, 4, bias=False),
            }
        )


class LoRATest(unittest.TestCase):
    def test_injection_matches_checkpoint_keys(self) -> None:
        model = ToyModel()
        replaced = inject_lora_into_transformer(
            model,
            rank=2,
            alpha=4.0,
            dropout=0.0,
            target_names={"wq"},
        )
        self.assertEqual(replaced, ["wq"])
        self.assertIsInstance(model.transformer["wq"], LoRALinear)
        self.assertIsInstance(model.transformer["untouched"], torch.nn.Linear)
        state = model.state_dict()
        adapter_state = {
            key: value
            for key, value in state.items()
            if ".lora_a." in key or ".lora_b." in key
        }
        self.assertEqual(lora_rank_from_state_dict(adapter_state), 2)
        self.assertFalse(set(adapter_state) - set(model.state_dict()))


if __name__ == "__main__":
    unittest.main()
