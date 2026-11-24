import torch
import pytest
import torch.nn as nn

from src.models.external_encoders import (
    EXTERNAL_ENCODERS,
    TimmExternalEncoder,
    build_external_encoder,
    validate_external_encoder_config,
)
from src.models.head import ViTClassifier


class TinyTokenViT(nn.Module):
    """Small deterministic ViT stand-in with CLS and two register tokens."""

    def __init__(self, embed_dim, num_reg_tokens=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_features = embed_dim
        self.has_class_token = True
        self.num_reg_tokens = num_reg_tokens
        self.num_prefix_tokens = 1 + num_reg_tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.reg_token = nn.Parameter(torch.zeros(1, num_reg_tokens, embed_dim))
        self.scale = nn.Parameter(torch.ones(1))

    def _tokens(self, x):
        batch = x.shape[0]
        patches = 3
        base = x.mean(dim=(1, 2, 3)).reshape(batch, 1, 1)
        offsets = torch.arange(patches, device=x.device, dtype=x.dtype).reshape(1, patches, 1)
        patch_tokens = base + offsets
        patch_tokens = patch_tokens.expand(batch, patches, self.embed_dim)
        cls = torch.full((batch, 1, self.embed_dim), 1000.0, device=x.device)
        registers = torch.full(
            (batch, self.num_reg_tokens, self.embed_dim), -1000.0, device=x.device
        )
        return torch.cat((cls, registers, patch_tokens), dim=1) * self.scale

    def forward_features(self, x):
        return self._tokens(x)

    def forward_intermediates(self, x, **kwargs):
        assert kwargs["indices"] == 4
        assert kwargs["norm"] is True
        assert kwargs["output_fmt"] == "NLC"
        assert kwargs["return_prefix_tokens"] is False
        patches = self._tokens(x)[:, self.num_prefix_tokens :]
        # Match timm: prefix tokens have already been removed from intermediates.
        return [patches + float(layer) for layer in range(4)]


@pytest.mark.parametrize("key", sorted(EXTERNAL_ENCODERS))
@pytest.mark.parametrize("representation", ["last_avgpool", "last4_avgpool_concat"])
def test_each_external_key_random_mocked_features_and_frozen(key, representation, monkeypatch):
    spec = EXTERNAL_ENCODERS[key]
    calls = []

    def fake_create_model(name, **kwargs):
        calls.append((name, kwargs))
        return TinyTokenViT(spec.embed_dim)

    monkeypatch.setattr("timm.create_model", fake_create_model)
    encoder = build_external_encoder(
        key,
        pretrained=False,
        crop_size=spec.crop_size,
        meta_args={"embed_dim": spec.embed_dim},
        data_args={"normalization": spec.normalization},
        mask_args={"patch_size": spec.patch_size},
    )
    assert calls == [
        (
            spec.timm_name,
            {
                "pretrained": False,
                "num_classes": 0,
                "img_size": spec.crop_size,
            },
        )
    ]
    assert not encoder.training
    assert all(not parameter.requires_grad for parameter in encoder.parameters())

    model = ViTClassifier(
        encoder,
        num_classes=5,
        embed_dim=spec.embed_dim,
        representation_type=representation,
        head_type="linear",
    )
    images = torch.randn(2, 3, 8, 8)
    features = model._extract_representation(images)
    multiplier = 4 if representation == "last4_avgpool_concat" else 1
    assert features.shape == (2, multiplier * spec.embed_dim)
    assert model(images).shape == (2, 5)
    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
    assert all(parameter.requires_grad for parameter in model.head.parameters())


def test_patch_pool_excludes_cls_and_registers_and_cls_is_preprojection():
    backbone = TinyTokenViT(embed_dim=4, num_reg_tokens=2)
    encoder = TimmExternalEncoder(backbone, embed_dim=4)
    images = torch.zeros(1, 3, 8, 8)
    patch_tokens = encoder(images)
    assert patch_tokens.shape == (1, 3, 4)
    assert torch.equal(patch_tokens[0, :, 0], torch.tensor([0.0, 1.0, 2.0]))
    assert not torch.any(patch_tokens == 1000.0)
    assert not torch.any(patch_tokens == -1000.0)

    classifier = ViTClassifier(
        encoder, num_classes=2, embed_dim=4, representation_type="last_avgpool"
    )
    assert torch.equal(
        classifier._extract_representation(images),
        torch.ones(1, 4),
    )
    cls_classifier = ViTClassifier(
        encoder, num_classes=2, embed_dim=4, representation_type="cls"
    )
    # 1000 comes from the mocked pre-projection token, not a 768-wide CLIP head.
    assert torch.equal(cls_classifier._extract_representation(images), torch.full((1, 4), 1000.0))


@pytest.mark.parametrize("key", sorted(EXTERNAL_ENCODERS))
def test_registered_ids_build_with_actual_timm_on_meta_device(key):
    """Check real constructors/dims without allocating or downloading huge weights."""
    timm = pytest.importorskip("timm")
    spec = EXTERNAL_ENCODERS[key]
    with torch.device("meta"):
        backbone = timm.create_model(
            spec.timm_name,
            pretrained=False,
            num_classes=0,
            img_size=spec.crop_size,
        )
    assert backbone.embed_dim == spec.embed_dim
    assert backbone.patch_embed.patch_size == (spec.patch_size, spec.patch_size)
    assert backbone.patch_embed.img_size == (spec.crop_size, spec.crop_size)
    assert isinstance(backbone.head, nn.Identity)


def test_real_timm_tiny_random_weight_adapter_without_download():
    timm = pytest.importorskip("timm")
    backbone = timm.create_model(
        "vit_tiny_patch16_224", pretrained=False, num_classes=0, img_size=32
    )
    encoder = TimmExternalEncoder(backbone, embed_dim=192)
    images = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        expected = backbone.forward_features(images)[:, backbone.num_prefix_tokens :]
        actual = encoder(images)
        final, layers = encoder(images, return_layer_outputs=True, num_last_layers=4)
    assert torch.allclose(actual, expected)
    assert actual.shape == (2, 4, 192)
    assert len(layers) == 4
    assert all(layer.shape == (2, 4, 192) for layer in layers)
    assert torch.equal(final, layers[-1])


def test_external_config_validation_rejects_wrong_clip_projection_or_resolution():
    with pytest.raises(ValueError, match="crop_size"):
        validate_external_encoder_config(
            "clip_vitl14_336",
            {"embed_dim": 1024},
            {"crop_size": 224, "normalization": "clip"},
            {"patch_size": 14},
        )
    with pytest.raises(ValueError, match="embed_dim"):
        validate_external_encoder_config(
            "clip_vitl14",
            {"embed_dim": 768},
            {"crop_size": 224, "normalization": "clip"},
            {"patch_size": 14},
        )
