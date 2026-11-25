"""Frozen external vision encoders used by the linear-probe baselines.

The adapter deliberately exposes the same token-level interface as the I-JEPA
VisionTransformer: ``forward`` returns patch tokens and ``return_layer_outputs``
returns normalized outputs from the last transformer blocks.  Consequently the
existing ViTClassifier heads can be reused without a second probing code path.

Only timm is imported lazily, when an external model is actually requested.
Existing I-JEPA runs therefore do not require timm and do not take this path.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ExternalEncoderSpec:
    timm_name: str
    embed_dim: int
    patch_size: int
    crop_size: int
    normalization: str


EXTERNAL_ENCODERS = {
    "dinov2_vitg14": ExternalEncoderSpec(
        "vit_giant_patch14_dinov2.lvd142m", 1536, 14, 224, "imagenet"
    ),
    "dinov2_vitl14": ExternalEncoderSpec(
        "vit_large_patch14_dinov2.lvd142m", 1024, 14, 224, "imagenet"
    ),
    "mae_vith14": ExternalEncoderSpec(
        "vit_huge_patch14_224.mae", 1280, 14, 224, "imagenet"
    ),
    "sup_vith14_in21k": ExternalEncoderSpec(
        "vit_huge_patch14_224.orig_in21k", 1280, 14, 224, "imagenet"
    ),
    "clip_vitl14": ExternalEncoderSpec(
        "vit_large_patch14_clip_224.openai", 1024, 14, 224, "clip"
    ),
    "clip_vitl14_336": ExternalEncoderSpec(
        "vit_large_patch14_clip_336.openai", 1024, 14, 336, "clip"
    ),
}


def is_external_encoder(model_name):
    return model_name in EXTERNAL_ENCODERS


def get_external_encoder_spec(model_name):
    try:
        return EXTERNAL_ENCODERS[model_name]
    except KeyError as exc:
        keys = ", ".join(sorted(EXTERNAL_ENCODERS))
        raise ValueError(f"Unknown external encoder {model_name!r}; choose from {keys}") from exc


def validate_external_encoder_config(model_name, meta_args, data_args, mask_args=None):
    """Fail before downloading weights if architecture/preprocessing disagree."""
    spec = get_external_encoder_spec(model_name)
    mask_args = mask_args or {}
    embed_dim = int(meta_args.get("embed_dim", spec.embed_dim))
    crop_size = int(data_args.get("crop_size", meta_args.get("crop_size", spec.crop_size)))
    patch_size = int(mask_args.get("patch_size", meta_args.get("patch_size", spec.patch_size)))
    normalization = str(data_args.get("normalization", spec.normalization)).lower()
    mismatches = []
    if embed_dim != spec.embed_dim:
        mismatches.append(f"meta.embed_dim must be {spec.embed_dim}, got {embed_dim}")
    if crop_size != spec.crop_size:
        mismatches.append(f"data.crop_size must be {spec.crop_size}, got {crop_size}")
    if patch_size != spec.patch_size:
        mismatches.append(f"patch_size must be {spec.patch_size}, got {patch_size}")
    if normalization != spec.normalization:
        mismatches.append(
            f"data.normalization must be {spec.normalization!r}, got {normalization!r}"
        )
    if mismatches:
        raise ValueError(f"Invalid config for {model_name}: " + "; ".join(mismatches))
    return spec


class TimmExternalEncoder(nn.Module):
    """Expose normalized pre-projection timm tokens in the I-JEPA format."""

    def __init__(self, backbone, embed_dim):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = int(embed_dim)
        self._num_prefix_tokens = self._infer_num_prefix_tokens(backbone)
        self._has_class_token = bool(
            getattr(backbone, "has_class_token", getattr(backbone, "cls_token", None) is not None)
        )
        self.requires_grad_(False)
        self.eval()

    @staticmethod
    def _infer_num_prefix_tokens(backbone):
        value = getattr(backbone, "num_prefix_tokens", None)
        if value is not None:
            return int(value)
        has_cls = bool(
            getattr(backbone, "has_class_token", getattr(backbone, "cls_token", None) is not None)
        )
        # torch.hub DINOv2 uses num_register_tokens; timm calls it num_reg_tokens.
        registers = int(
            getattr(
                backbone,
                "num_reg_tokens",
                getattr(backbone, "num_register_tokens", 0),
            )
        )
        return int(has_cls) + registers

    def _split_tokens(self, tokens):
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3:
            raise ValueError("External ViT forward_features must return a [B, N, D] tensor")
        if tokens.shape[-1] != self.embed_dim:
            raise ValueError(
                f"External token width {tokens.shape[-1]} != configured {self.embed_dim}"
            )
        if tokens.shape[1] <= self._num_prefix_tokens:
            raise ValueError("External ViT output contains no patch tokens")
        return tokens[:, self._num_prefix_tokens :], tokens[:, : self._num_prefix_tokens]

    def forward(self, x, return_layer_outputs=False, num_last_layers=4, **_kwargs):
        if not return_layer_outputs:
            tokens = self.backbone.forward_features(x)
            patch_tokens, _ = self._split_tokens(tokens)
            return patch_tokens

        # norm=True matches I-JEPA, which applies its final norm to every selected
        # block. NLC + timm's prefix splitting excludes CLS and all registers.
        output = self.backbone.forward_intermediates(
            x,
            indices=int(num_last_layers),
            return_prefix_tokens=False,
            norm=True,
            stop_early=False,
            output_fmt="NLC",
            intermediates_only=True,
        )
        layer_outputs = output
        if isinstance(output, tuple) and len(output) == 2 and isinstance(output[1], (list, tuple)):
            layer_outputs = output[1]
        if not isinstance(layer_outputs, (list, tuple)) or not layer_outputs:
            raise ValueError("timm forward_intermediates returned no transformer features")
        layer_outputs = list(layer_outputs)
        for tokens in layer_outputs:
            if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3:
                raise ValueError("Intermediate external features must be [B, N, D] tensors")
            if tokens.shape[-1] != self.embed_dim:
                raise ValueError("Intermediate external feature width does not match embed_dim")
        return layer_outputs[-1], layer_outputs

    def forward_for_cache(self, x, num_last_layers=4):
        """Return normalized patch layers and CLS from one intermediate pass."""
        output = self.backbone.forward_intermediates(
            x,
            indices=int(num_last_layers),
            return_prefix_tokens=self._has_class_token,
            norm=True,
            stop_early=False,
            output_fmt="NLC",
            intermediates_only=True,
        )
        layers = []
        cls = None
        for item in output:
            if isinstance(item, tuple) and len(item) == 2:
                patch_tokens, prefix_tokens = item
                layers.append(patch_tokens)
                if prefix_tokens is not None and prefix_tokens.shape[1]:
                    cls = prefix_tokens[:, 0]
            else:
                layers.append(item)
        if len(layers) != num_last_layers:
            raise ValueError("Unexpected number of timm intermediate layers")
        return layers[-1], layers, cls

    def forward_cls(self, x):
        """Return the normalized, pre-projection CLS token (an ablation only)."""
        if not self._has_class_token or self._num_prefix_tokens < 1:
            raise ValueError("CLS representation requested for an encoder without a CLS token")
        tokens = self.backbone.forward_features(x)
        _, prefix_tokens = self._split_tokens(tokens)
        return prefix_tokens[:, 0]


def build_external_encoder(
    model_name,
    crop_size=None,
    pretrained=True,
    device=None,
    meta_args=None,
    data_args=None,
    mask_args=None,
):
    """Build and freeze one registered external encoder using public timm weights."""
    meta_args = dict(meta_args or {})
    data_args = dict(data_args or {})
    if crop_size is not None:
        data_args["crop_size"] = int(crop_size)
    spec = validate_external_encoder_config(model_name, meta_args, data_args, mask_args)
    try:
        import timm
    except ImportError as exc:
        raise ImportError(
            "External encoders require timm; install the project requirements first"
        ) from exc

    backbone = timm.create_model(
        spec.timm_name,
        pretrained=bool(pretrained),
        num_classes=0,  # Drop classifiers and, importantly, CLIP's 768-wide projection.
        img_size=int(data_args.get("crop_size", spec.crop_size)),
    )
    actual_dim = int(getattr(backbone, "embed_dim", getattr(backbone, "num_features", 0)))
    if actual_dim != spec.embed_dim:
        raise ValueError(
            f"timm model {spec.timm_name} has width {actual_dim}, expected {spec.embed_dim}"
        )
    encoder = TimmExternalEncoder(backbone, spec.embed_dim)
    if device is not None:
        encoder.to(device)
    return encoder
