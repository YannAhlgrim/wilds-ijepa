import torch
import torch.nn as nn


class ViTClassifier(nn.Module):
    def __init__(
        self,
        encoder,
        num_classes,
        embed_dim,
        probe_type="linear",
        mlp_hidden_dim=None,
        dropout=0.0,
    ):
        super().__init__()
        self.encoder = encoder

        probe_type = str(probe_type).lower()
        if probe_type == "linear":
            self.head = nn.Linear(embed_dim, num_classes)
        elif probe_type == "mlp":
            if mlp_hidden_dim is None:
                raise ValueError("mlp_hidden_dim must be set for probe_type='mlp'")
            self.head = nn.Sequential(
                nn.Linear(embed_dim, mlp_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden_dim, num_classes),
            )
        else:
            raise ValueError(f"Unknown probe_type: {probe_type}")

        for module in self.head.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        # ViT -> (B, N, D)
        if any(p.requires_grad for p in self.encoder.parameters()):
            features = self.encoder(x)
        else:
            with torch.no_grad():
                features = self.encoder(x)

        # Average Pool -> (B, D)
        avg_embed = features.mean(dim=1)

        logits = self.head(avg_embed)
        return logits
