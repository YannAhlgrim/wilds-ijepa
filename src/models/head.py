import torch
import torch.nn as nn


class ViTClassifier(nn.Module):
    def __init__(
        self,
        encoder,
        num_classes,
        embed_dim,
        representation_type="last_avgpool",
        head_type="linear",
    ):
        super().__init__()
        self.encoder = encoder
        self.representation_type = str(representation_type).lower()

        if self.representation_type == "last_avgpool":
            in_dim = embed_dim
        elif self.representation_type == "last4_avgpool_concat":
            in_dim = 4 * embed_dim
        else:
            raise ValueError(f"Unknown representation_type: {representation_type}")

        head_type = str(head_type).lower()
        if head_type == "linear":
            self.head = nn.Linear(in_dim, num_classes)
        elif head_type == "bn_linear":
            self.head = nn.Sequential(
                nn.BatchNorm1d(in_dim, affine=False, eps=1e-6),
                nn.Linear(in_dim, num_classes),
            )
        else:
            raise ValueError(f"Unknown head_type: {head_type}")

        for module in self.head.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.01)
                nn.init.zeros_(module.bias)

    def _extract_representation(self, x):
        if self.representation_type == "last_avgpool":
            features = self.encoder(x)
            return features.mean(dim=1)

        _, layer_outputs = self.encoder(
            x, return_layer_outputs=True, num_last_layers=4
        )
        pooled = [layer_tokens.mean(dim=1) for layer_tokens in layer_outputs]
        return torch.cat(pooled, dim=-1)

    def forward(self, x):
        if any(p.requires_grad for p in self.encoder.parameters()):
            representation = self._extract_representation(x)
        else:
            with torch.no_grad():
                representation = self._extract_representation(x)

        logits = self.head(representation)
        return logits
