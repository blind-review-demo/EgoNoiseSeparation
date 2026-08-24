from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from einops import rearrange

from learning2hear.config import TransferDiTSettings
from sam_audio.model.base import BaseModel
from sam_audio.model.codec import DACVAE
from sam_audio.model.config import SAMAudioConfig as TransferDiTConfig
from sam_audio.model.config import TransformerConfig
from sam_audio.model.model import DFLT_ODE_OPT
from sam_audio.model.model import SAMAudio as FoundationSeparator
from sam_audio.model.model import SeparationResult
from sam_audio.model.transformer import DiT, gate, modulate


def zero_linear(module: torch.nn.Linear) -> None:
    torch.nn.init.zeros_(module.weight)
    if module.bias is not None:
        torch.nn.init.zeros_(module.bias)


def init_adapter_down(module: torch.nn.Linear) -> None:
    torch.nn.init.kaiming_uniform_(module.weight, a=5**0.5)
    if module.bias is not None:
        torch.nn.init.zeros_(module.bias)


class LayerCrossResidualAdapter(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        rank: int = TransferDiTSettings.transfer_dit_layer_residual_rank,
    ) -> None:
        super().__init__()
        self.arch_cross_down = torch.nn.Linear(dim, rank, bias=False)
        self.arch_cross_up = torch.nn.Linear(rank, dim, bias=False)
        init_adapter_down(self.arch_cross_down)
        zero_linear(self.arch_cross_up)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.arch_cross_up(F.silu(self.arch_cross_down(h)))


def match_arch_cross_adapter_rank(
    model: torch.nn.Module,
    state_dict: Dict[str, torch.Tensor],
) -> Optional[int]:
    transformer = getattr(model, "transformer", None)
    adapters = getattr(transformer, "arch_cross_adapters", None)
    if not isinstance(adapters, torch.nn.ModuleList) or not adapters:
        return None
    down_key = "transformer.arch_cross_adapters.0.arch_cross_down.weight"
    down_weight = state_dict.get(down_key)
    if down_weight is None or down_weight.ndim != 2:
        return None
    checkpoint_rank = int(down_weight.shape[0])
    current_down = adapters[0].arch_cross_down
    if checkpoint_rank == int(current_down.out_features):
        return checkpoint_rank
    dim = int(current_down.in_features)
    transformer.arch_cross_adapters = torch.nn.ModuleList(
        LayerCrossResidualAdapter(dim, checkpoint_rank)
        for _ in range(len(adapters))
    ).to(device=current_down.weight.device, dtype=current_down.weight.dtype)
    return checkpoint_rank


class TransferDiTTransformer(DiT):
    """Transfer-DiT backbone with per-layer cross-position residuals."""

    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        for layer in self.layers:
            layer.cross_attention = None
        self.y_embedder = None
        residual_rank = int(TransferDiTSettings.transfer_dit_layer_residual_rank)
        self.arch_cross_adapters = torch.nn.ModuleList(
            LayerCrossResidualAdapter(int(config.dim), residual_rank) for _ in self.layers
        )

    def _forward_layer_with_cross_residual(
        self,
        layer: torch.nn.Module,
        adapter: torch.nn.Module,
        h: torch.Tensor,
        t0: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        biases = layer.scale_shift_table[None] + t0.reshape(h.size(0), 6, -1)
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = biases.chunk(6, dim=1)

        h_attn = layer.attention(
            modulate(layer.attention_norm(h), shift_msa, scale_msa),
            key_padding_mask=padding_mask,
            rope=self.rope_embeddings,
        )
        h = h + gate(h_attn, gate_msa)
        h = h + adapter(h)
        h_ff = layer.feed_forward(modulate(layer.ffn_norm(h), shift_mlp, scale_mlp))
        return h + gate(h_ff, gate_mlp)

    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        *,
        padding_mask: Optional[torch.Tensor] = None,
        memory: Optional[torch.Tensor] = None,
        memory_padding_mask: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        del memory, memory_padding_mask
        x = rearrange(x, "b l c-> b c l")
        h = self.x_embedder(x)
        h = rearrange(h, "b c l -> b l c")
        h = F.dropout(h, p=self.dropout, training=self.training)

        t = self.t_embedder(time)
        t0 = self.t_block_non_linearity(t)
        t0 = self.t_block(t0)

        for index, layer in enumerate(self.layers):
            h = self._forward_layer_with_cross_residual(
                layer,
                self.arch_cross_adapters[index],
                h,
                t0,
                padding_mask,
            )

        shift, scale = (self.final_layer_scale_shift_table[None] + t[:, None]).chunk(
            2, dim=1
        )
        if self.norm is not None:
            h = self.norm(h)
        h = modulate(h, shift, scale)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.output(h)


class TransferDiT(FoundationSeparator):
    """Prompt-free SAM-Audio separator used by EgoNoiseSeparation."""

    config_cls = TransferDiTConfig
    revision = None

    _removed_prefixes = (
        "memory_proj.",
        "text_encoder.",
        "vision_encoder.",
        "align_masked_video.",
        "embed_anchors.",
        "span_predictor.",
        "span_predictor_transform.",
        "visual_ranker.",
        "text_ranker.",
        "transformer.y_embedder.",
    )
    _removed_patterns = (
        re.compile(r"^transformer\.layers\.\d+\.cross_attention\."),
    )

    def __init__(self, cfg: TransferDiTConfig):
        BaseModel.__init__(self)
        self.audio_codec = DACVAE(cfg.audio_codec)
        self.transformer = TransferDiTTransformer(cfg.transformer)
        self.proj = torch.nn.Linear(cfg.in_channels, cfg.transformer.dim)
        self.timestep_emb = None
        self.visual_ranker = None
        self.text_ranker = None

    @property
    def sample_rate(self):
        return self.audio_codec.sample_rate

    def align_inputs(
        self,
        noisy_audio: torch.Tensor,
        audio_features: torch.Tensor,
        masked_video_features: Optional[torch.Tensor] = None,
        anchor_ids: Optional[torch.Tensor] = None,
        anchor_alignment: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del masked_video_features, anchor_ids, anchor_alignment
        x = torch.cat(
            [
                noisy_audio,
                torch.zeros_like(audio_features),
                audio_features,
            ],
            dim=2,
        )
        return self.proj(x)

    def forward(
        self,
        noisy_audio: torch.Tensor,
        audio_features: torch.Tensor,
        time: torch.Tensor,
        text_features: Optional[torch.Tensor] = None,
        masked_video_features: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        anchor_ids: Optional[torch.Tensor] = None,
        anchor_alignment: Optional[torch.Tensor] = None,
        audio_pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del text_features, text_mask
        aligned_inputs = self.align_inputs(
            noisy_audio,
            audio_features,
            masked_video_features=masked_video_features,
            anchor_ids=anchor_ids,
            anchor_alignment=anchor_alignment,
        )
        return self.transformer(
            aligned_inputs,
            time,
            padding_mask=audio_pad_mask,
        )

    def _get_audio_features(self, audios: torch.Tensor) -> torch.Tensor:
        audio_features = self.audio_codec(audios).transpose(1, 2)
        return torch.cat([audio_features, audio_features], dim=2)

    def _repeat_for_reranking(self, tensor, candidates: int):
        if tensor is None or candidates <= 1:
            return tensor
        batch = tensor.size(0)
        rest = tensor.shape[1:]
        return (
            tensor.unsqueeze(1)
            .expand(batch, candidates, *rest)
            .reshape(batch * candidates, *rest)
        )

    def _unrepeat_from_reranking(self, tensor, candidates: int):
        if tensor is None or candidates <= 1:
            return tensor
        return tensor[::candidates]

    def _get_forward_args(self, batch, candidates: int = 1) -> Dict[str, Any]:
        audio_features = self._get_audio_features(batch.audios)
        return {
            "audio_features": self._repeat_for_reranking(audio_features, candidates),
            "text_features": None,
            "text_mask": None,
            "masked_video_features": None,
            "anchor_ids": None,
            "anchor_alignment": None,
            "audio_pad_mask": self._repeat_for_reranking(
                batch.audio_pad_mask, candidates
            ),
        }

    @torch.inference_mode()
    def separate(
        self,
        batch,
        noise: Optional[torch.Tensor] = None,
        ode_opt: Dict[str, Any] = DFLT_ODE_OPT,
        reranking_candidates: int = 1,
        predict_spans: bool = False,
    ) -> SeparationResult:
        del predict_spans
        return super().separate(
            batch=batch,
            noise=noise,
            ode_opt=ode_opt,
            reranking_candidates=reranking_candidates,
            predict_spans=False,
        )

    @classmethod
    def _is_removed_key(cls, key: str) -> bool:
        return key.startswith(cls._removed_prefixes) or any(
            pattern.search(key) for pattern in cls._removed_patterns
        )

    @classmethod
    def _is_transfer_key(cls, key: str) -> bool:
        return ".arch_cross_" in key or key.startswith("transformer.arch_cross_")

    def load_state_dict(self, state_dict, strict: bool = True):
        filtered = {
            key: value
            for key, value in state_dict.items()
            if not self._is_removed_key(key)
        }
        result = torch.nn.Module.load_state_dict(self, filtered, strict=False)
        if strict:
            missing = [
                key
                for key in result.missing_keys
                if not self._is_removed_key(key) and not self._is_transfer_key(key)
            ]
            unexpected = [
                key for key in result.unexpected_keys if not self._is_removed_key(key)
            ]
            if missing or unexpected:
                raise RuntimeError(
                    f"Missing keys: {missing}, unexpected_keys: {unexpected}"
                )
        return result


__all__ = [
    "LayerCrossResidualAdapter",
    "match_arch_cross_adapter_rank",
    "TransferDiT",
    "TransferDiTTransformer",
]
