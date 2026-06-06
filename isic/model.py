"""Bayesian VGG16 for melanoma classification (section 5.5).

We take VGG16 pre-trained on ImageNet, replace the 1000-way head with a
2-way (benign/malignant) head, and keep the two dropout-0.5 layers in the
classifier so we can do MC-dropout at test time.
"""
import torch
import torch.nn as nn
import torchvision

from isic.config import ISICConfig


def build_vgg16(cfg: ISICConfig, pretrained: bool = True) -> nn.Module:
    """Return a VGG16 with a 2-class head and dropout p set from the config."""
    if pretrained:
        weights = torchvision.models.VGG16_Weights.IMAGENET1K_V1
        model = torchvision.models.vgg16(weights=weights)
    else:
        model = torchvision.models.vgg16(weights=None)  # for offline tests

    # torchvision VGG classifier:
    #   Linear(25088,4096) ReLU Dropout Linear(4096,4096) ReLU Dropout Linear(4096,1000)
    # Set dropout probability and swap the final layer for a 2-way output.
    for m in model.classifier:
        if isinstance(m, nn.Dropout):
            m.p = cfg.dropout_p
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 2)

    if cfg.freeze_features:
        for p in model.features.parameters():
            p.requires_grad = False

    return model


def enable_dropout(model: nn.Module) -> None:
    """Activate dropout layers at test time for MC-dropout sampling."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
