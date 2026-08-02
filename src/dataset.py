"""
PyTorch Dataset for the processed ChestX-ray14 subset.

Design note: we keep subgroup metadata (sex, age_group, view_position)
attached to every __getitem__ return, not just the image + label. This is
deliberate — it means the exact same Dataset object can be reused unchanged
for both training (which only needs image + label) and the bias audit
(which needs the subgroup fields too). One dataset class, two uses.
"""

from pathlib import Path

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import GroupShuffleSplit

import config


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(train: bool):
    # Grayscale(num_output_channels=3) must run on the PIL image, before
    # ToTensor. Run on an already-1-channel tensor, torchvision's
    # rgb_to_grayscale expands using the *input* tensor's shape, which is
    # already (1,H,W) — so it silently stays 1-channel instead of becoming
    # 3-channel, and ResNet18's first conv (which expects 3 channels) would
    # fail on the very first batch.
    if train:
        return transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            # Small rotation only — X-rays are orientation-sensitive,
            # aggressive augmentation can distort clinically meaningful
            # structure, so we keep this conservative.
            transforms.RandomRotation(degrees=5),
            transforms.Grayscale(num_output_channels=3),  # CXRs are grayscale;
            # we replicate to 3 channels so we can use ImageNet-pretrained
            # backbones without modifying the first conv layer.
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


class ChestXrayDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, train: bool = True):
        self.df = dataframe.reset_index(drop=True)
        self.transform = get_transforms(train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = config.build_image_index()[row["Image Index"]]
        image = Image.open(img_path).convert("L")  # load as grayscale
        image = self.transform(image)

        labels = torch.tensor(
            [row[f] for f in config.TARGET_FINDINGS], dtype=torch.float32
        )

        subgroup = {
            "sex": row["sex"],
            "age_group": str(row["age_group"]),
            "view_position": row["view_position"],
            "image_index": row["Image Index"],
        }
        return image, labels, subgroup


def load_and_split():
    """Load processed CSV and split into train/val/test, grouped by Patient ID.

    NIH ChestX-ray14 has multiple images per patient (follow-up scans). A
    row-level random split lets the same patient appear in both train and
    test, which leaks patient-specific appearance into the "held-out" set
    and inflates test metrics. GroupShuffleSplit (the standard scikit-learn
    tool for exactly this case) guarantees every image of a given patient
    lands in exactly one split.
    """
    df = pd.read_csv(config.PROCESSED_CSV)

    splitter1 = GroupShuffleSplit(
        n_splits=1, test_size=config.TEST_FRACTION, random_state=config.RANDOM_SEED
    )
    trainval_idx, test_idx = next(splitter1.split(df, groups=df["Patient ID"]))
    trainval_df, test_df = df.iloc[trainval_idx], df.iloc[test_idx]

    val_fraction_of_trainval = config.VAL_FRACTION / (1 - config.TEST_FRACTION)
    splitter2 = GroupShuffleSplit(
        n_splits=1, test_size=val_fraction_of_trainval, random_state=config.RANDOM_SEED
    )
    train_idx, val_idx = next(
        splitter2.split(trainval_df, groups=trainval_df["Patient ID"])
    )
    train_df = trainval_df.iloc[train_idx].reset_index(drop=True)
    val_df = trainval_df.iloc[val_idx].reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    assert set(train_df["Patient ID"]) & set(val_df["Patient ID"]) == set()
    assert set(train_df["Patient ID"]) & set(test_df["Patient ID"]) == set()
    assert set(val_df["Patient ID"]) & set(test_df["Patient ID"]) == set()

    print(f"Split sizes -> train: {len(train_df)}, val: {len(val_df)}, "
          f"test: {len(test_df)} (grouped by Patient ID, no patient overlap)")
    return train_df, val_df, test_df
