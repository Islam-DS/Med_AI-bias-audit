"""
Model: ImageNet-pretrained ConvNeXt-Tiny (via timm) with the classifier head
swapped for multi-label output.

Why ConvNeXt-Tiny via timm, not a hand-picked torchvision model? timm is the
de facto standard library for pretrained vision backbones in both industry
and research — it's what you cite when a reviewer asks "why this backbone,"
because the answer is "the maintained, benchmarked, widely-used option,"
not a bespoke choice. ConvNeXt (Liu et al., 2022) is a modern pure-conv
architecture that matches/beats Vision Transformers at comparable size
while keeping the training stability and lower data requirements of CNNs —
a reasonable, defensible pick for a ~15k-image fine-tuning task where a ViT
would be more data-hungry.
"""

import timm
import torch
import torch.nn as nn

import config


def build_model(num_classes: int = None) -> nn.Module:
    if num_classes is None:
        num_classes = len(config.TARGET_FINDINGS)

    # timm's num_classes argument replaces the pretrained classifier head
    # for us, sized for our multi-label output — no manual layer surgery
    # needed regardless of the specific backbone's head architecture.
    model = timm.create_model(
        config.BACKBONE, pretrained=True, num_classes=num_classes,
        drop_path_rate=config.DROP_PATH_RATE,
    )

    if config.FREEZE_EARLY_STAGES:
        # ConvNeXt-Tiny is 28M params; ~15.7k training images is small
        # enough that a full fine-tune overfits within a handful of epochs
        # (observed directly: val loss climbs from epoch 8 on while train
        # loss keeps dropping). Freezing the stem + first two stages (low-
        # level, generic features that transfer well from ImageNet as-is)
        # and only fine-tuning the last two stages + head is the standard
        # partial fine-tuning strategy for exactly this capacity/data
        # mismatch — it cuts trainable params substantially and lets the
        # model use its capacity on task-specific high-level features only.
        for param in model.stem.parameters():
            param.requires_grad = False
        for param in model.stages[0].parameters():
            param.requires_grad = False
        for param in model.stages[1].parameters():
            param.requires_grad = False

    return model


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # Apple Silicon
        return torch.device("mps")
    return torch.device("cpu")
