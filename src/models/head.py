import torch
import torch.nn as nn


class ViTClassifier(nn.Module):
    def __init__(self, encoder, num_classes, embed_dim):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)

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
