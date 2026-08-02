"""
Step 4: generate saliency maps using Captum's Integrated Gradients, and
specifically compare maps for CORRECT vs INCORRECT predictions within the
subgroup(s) that showed the biggest gap in bias_analysis.py.

Why Integrated Gradients rather than plain Grad-CAM? IG gives per-pixel
attribution relative to a defined baseline (we use a black/zero image),
with a formal theoretical grounding (it satisfies "completeness": attributions
sum to the difference between the model's output on the image vs the
baseline). It's a genuinely different and slightly more rigorous tool than
Grad-CAM, which only shows coarse class-activation regions from a late
conv layer. Captum gives us both in one library if you want to compare —
IG is used here as the primary method.

Run: python src/interpretability.py
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients

import config
from dataset import ChestXrayDataset, load_and_split, IMAGENET_MEAN, IMAGENET_STD
from model import build_model, get_device


def denormalize(img_tensor):
    """Undo ImageNet normalization for display purposes."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = img_tensor.cpu() * std + mean
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


def make_saliency_figure(model, device, image_tensor, finding_idx, finding_name,
                          true_label, save_path, title_suffix=""):
    model.eval()
    image_tensor = image_tensor.unsqueeze(0).to(device)
    image_tensor.requires_grad_()

    ig = IntegratedGradients(model)
    baseline = torch.zeros_like(image_tensor)

    attributions = ig.attribute(
        image_tensor, baselines=baseline, target=finding_idx, n_steps=50
    )

    with torch.no_grad():
        pred_prob = torch.sigmoid(model(image_tensor))[0, finding_idx].item()

    attr_np = attributions.squeeze(0).cpu().detach().numpy()
    attr_map = np.mean(np.abs(attr_np), axis=0)  # collapse channels

    orig_img = denormalize(image_tensor.squeeze(0))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(orig_img)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(orig_img)
    axes[1].imshow(attr_map, cmap="jet", alpha=0.5)
    axes[1].set_title(f"Integrated Gradients: {finding_name}\n"
                       f"true={int(true_label)}  pred_prob={pred_prob:.2f}"
                       f"{title_suffix}")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    device = get_device()

    if not config.CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"No trained model found at {config.CHECKPOINT_PATH}. "
            "Run `python src/train.py` first."
        )

    metrics_path = config.REPORTS_DIR / "subgroup_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(
            "Run `python src/bias_analysis.py` first so we know which "
            "subgroup/finding combination to focus the saliency comparison on."
        )

    metrics = pd.read_csv(metrics_path)
    # Pick the finding+axis with the widest AUROC range across subgroups as
    # the most interesting case to visualize — but only among subgroups
    # flagged "reliable" (see bias_analysis.py: below
    # MIN_POSITIVES_FOR_RELIABLE_AUROC positives, a single sample can swing
    # AUROC by a huge margin). Without this filter, this picked
    # Pneumothorax/age_group purely because its 80+ bucket has ~1 positive
    # case — a sample-size artifact, not a real failure mode worth a
    # headline saliency comparison.
    reliable_metrics = metrics[metrics["reliable"] == True]
    spread = (reliable_metrics.groupby(["finding", "axis"])["AUROC"]
              .agg(lambda x: x.max() - x.min() if len(x) >= 2 else np.nan)
              .reset_index(name="spread")
              .dropna(subset=["spread"])
              .sort_values("spread", ascending=False))
    if len(spread) == 0:
        raise RuntimeError(
            "No finding/axis combination has 2+ reliable subgroups to "
            "compare — check subgroup_metrics.csv."
        )
    top = spread.iloc[0]
    finding_name, axis = top["finding"], top["axis"]
    finding_idx = config.TARGET_FINDINGS.index(finding_name)
    print(f"Focusing saliency comparison on: finding='{finding_name}', "
          f"axis='{axis}' (widest AUROC spread = {top['spread']:.3f})")

    _train_df, _val_df, test_df = load_and_split()
    test_ds = ChestXrayDataset(test_df, train=False)

    checkpoint = torch.load(config.CHECKPOINT_PATH, map_location=device)
    model = build_model().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Find a few correct and incorrect examples to visualize
    n_examples = 4
    found_correct, found_incorrect = 0, 0

    with torch.no_grad():
        for i in range(len(test_ds)):
            if found_correct >= n_examples and found_incorrect >= n_examples:
                break
            image, labels, subgroup = test_ds[i]
            true_label = labels[finding_idx].item()
            prob = torch.sigmoid(
                model(image.unsqueeze(0).to(device))
            )[0, finding_idx].item()
            pred_label = 1 if prob >= 0.5 else 0
            is_correct = (pred_label == true_label)

            if is_correct and found_correct < n_examples:
                save_path = (config.FIGURES_DIR /
                             f"saliency_correct_{found_correct}"
                             f"_{subgroup[axis]}.png")
                make_saliency_figure(
                    model, device, image, finding_idx, finding_name,
                    true_label, save_path,
                    title_suffix=f"\n[{axis}={subgroup[axis]}] CORRECT"
                )
                found_correct += 1

            elif not is_correct and found_incorrect < n_examples:
                save_path = (config.FIGURES_DIR /
                             f"saliency_incorrect_{found_incorrect}"
                             f"_{subgroup[axis]}.png")
                make_saliency_figure(
                    model, device, image, finding_idx, finding_name,
                    true_label, save_path,
                    title_suffix=f"\n[{axis}={subgroup[axis]}] INCORRECT"
                )
                found_incorrect += 1

    print(f"\nSaved {found_correct} correct + {found_incorrect} incorrect "
          f"saliency maps to {config.FIGURES_DIR}")
    print("Look for: does the model attend to anatomically plausible regions "
          "on correct predictions, and does that break down (e.g. attending "
          "to image borders/tubes/text artifacts) on incorrect ones - "
          "especially within the subgroup with lower AUROC?")


if __name__ == "__main__":
    main()
