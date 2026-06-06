"""Bayesian CNN used for active learning on MNIST.

Architecture follows the paper:
    conv - relu - conv - relu - maxpool - dropout - dense - relu - dropout - dense
Dropout layers are kept active at test time to perform MC-dropout, which
approximates sampling from the model posterior (Gal & Ghahramani, 2016).
"""
import torch
import torch.nn as nn

from config import Config


class BayesianCNN(nn.Module):
    """Small CNN with dropout before every weight layer."""

    def __init__(self, cfg: Config):
        super().__init__()
        f, k = cfg.num_filters, cfg.kernel_size

        # Convolutional feature extractor.
        self.features = nn.Sequential(
            nn.Conv2d(1, f, kernel_size=k),
            nn.ReLU(inplace=True),
            nn.Conv2d(f, f, kernel_size=k),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(cfg.pool_size_kernel),
            nn.Dropout(cfg.dropout_conv),   # dropout after conv block
        )

        # Infer flattened feature size from a dummy 28x28 input.
        with torch.no_grad():
            n = self.features(torch.zeros(1, 1, 28, 28)).numel()

        # Classifier head.
        self.classifier = nn.Sequential(
            nn.Linear(n, cfg.dense_units),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout_dense),  # dropout before final layer
            nn.Linear(cfg.dense_units, cfg.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)  # returns logits


def enable_dropout(model: nn.Module) -> None:
    """Set only the dropout layers to train mode (for MC-dropout at test time).

    Everything else stays in eval mode. Safe here because the model has no
    batch-norm; this keeps weights deterministic while sampling dropout masks.
    """
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()
