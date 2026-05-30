import torch
import torch.nn as nn


class MLP(nn.Module):
    """A shallow Multi-Layer Perceptron for MNIST."""
    def __init__(self, hidden_size: int = 256, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class SmallCNN(nn.Module):
    """Small convolutional neural network for MNIST."""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class MNISTTransformer(nn.Module):
    """A small Transformer Encoder for MNIST using 7x7 image patches."""
    def __init__(self, patch_size: int = 7, embed_dim: int = 64, num_heads: int = 4,
                 depth: int = 2, num_classes: int = 10):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (28 // patch_size) ** 2
        self.patch_dim = patch_size * patch_size
        self.patch_embed = nn.Linear(self.patch_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x: [B, 1, 28, 28] -> [B, 16, 49]
        b = x.size(0)
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(b, 1, -1, self.patch_size, self.patch_size)
        patches = patches.squeeze(1).flatten(2)
        tokens = self.patch_embed(patches) + self.pos_embed
        encoded = self.encoder(tokens)
        pooled = encoded.mean(dim=1)
        return self.head(pooled)


def get_model(name: str) -> nn.Module:
    name = name.lower()
    if name == "mlp":
        return MLP()
    if name == "cnn":
        return SmallCNN()
    if name == "transformer":
        return MNISTTransformer()
    raise ValueError(f"Unknown model: {name}. Choose from mlp, cnn, transformer.")
