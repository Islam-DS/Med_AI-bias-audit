"""
Step 2: train the multi-label chest X-ray classifier.

Modern fine-tuning recipe used here, and why each piece is there:
  - BCEWithLogitsLoss with per-class pos_weight: each finding is an
    independent binary decision (a patient can have several findings or
    none), and findings range from 20.6% positive (Infiltration) to 2.9%
    (Cardiomegaly). Unweighted BCE lets the loss be dominated by the common
    classes; pos_weight = n_negative/n_positive rebalances the gradient
    contribution per class instead of silently under-training the rare ones.
  - Linear warmup + cosine LR decay: standard modern schedule for
    fine-tuning a pretrained backbone — warmup avoids destructive early
    updates to pretrained weights, cosine decay avoids the abrupt LR drops
    of step schedules.
  - Mixed precision (autocast + GradScaler): halves memory and speeds up
    training on the GPU with no meaningful accuracy cost — standard practice
    for any GPU training run in 2024+.
  - Early stopping on val AUROC: NUM_EPOCHS is a ceiling, not a target;
    stopping when val AUROC plateaus avoids overfitting past convergence
    and wasting compute.
  - Temperature scaling fit at the end: see calibration.py.

Uses TorchMetrics for AUROC because implementing multi-label AUROC
bookkeeping by hand is exactly the kind of boilerplate a real library
should handle.

Run: python src/train.py
"""

import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.classification import MultilabelAUROC
from tqdm import tqdm

import config
from calibration import fit_temperature, expected_calibration_error
from dataset import ChestXrayDataset, load_and_split
from model import build_model, get_device


def compute_pos_weight(train_df) -> torch.Tensor:
    """sqrt of the raw imbalance ratio, not the raw ratio itself. Cardiomegaly
    at 2.9% positive gives a raw ratio of ~34x — weighting the loss that
    aggressively overcorrects and destabilizes training for a small backbone
    fine-tune. The sqrt-dampened weight is a standard softer alternative that
    still meaningfully upweights rare classes without dominating the loss."""
    weights = []
    for finding in config.TARGET_FINDINGS:
        n_pos = train_df[finding].sum()
        n_neg = len(train_df) - n_pos
        weights.append((n_neg / max(n_pos, 1)) ** 0.5)
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None,
              amp_dtype=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    use_amp = amp_dtype is not None

    auroc_metric = MultilabelAUROC(
        num_labels=len(config.TARGET_FINDINGS), average=None
    ).to(device)

    total_loss = 0.0
    with torch.set_grad_enabled(is_train):
        for images, labels, _subgroup in tqdm(loader, leave=False):
            images, labels = images.to(device), labels.to(device)

            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                 enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                if scaler is not None:
                    # float16 path: needs loss scaling to avoid gradient
                    # underflow.
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # bfloat16 (or no AMP) path: same dynamic range as
                    # float32, no scaler needed.
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * images.size(0)
            auroc_metric.update(logits.float(), labels.int())

    avg_loss = total_loss / len(loader.dataset)
    per_class_auroc = auroc_metric.compute()
    return avg_loss, per_class_auroc


@torch.no_grad()
def collect_logits(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for images, labels, _subgroup in tqdm(loader, desc="Collecting val logits", leave=False):
        images = images.to(device)
        logits = model(images).cpu()
        all_logits.append(logits)
        all_labels.append(labels)
    return torch.cat(all_logits), torch.cat(all_labels)


def main():
    device = get_device()
    print(f"Using device: {device}")

    train_df, val_df, _test_df = load_and_split()
    train_ds = ChestXrayDataset(train_df, train=True)
    val_ds = ChestXrayDataset(val_df, train=False)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE,
                               shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE,
                             shuffle=False, num_workers=4)

    model = build_model().to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"Trainable params: {n_trainable/1e6:.1f}M / {n_total/1e6:.1f}M total")

    pos_weight = compute_pos_weight(train_df).to(device)
    print(f"Class pos_weight (rarer findings weighted higher): "
          f"{dict(zip(config.TARGET_FINDINGS, pos_weight.tolist()))}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(trainable_params, lr=config.LEARNING_RATE,
                                   weight_decay=config.WEIGHT_DECAY)

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=config.WARMUP_EPOCHS
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(config.NUM_EPOCHS - config.WARMUP_EPOCHS, 1)
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[config.WARMUP_EPOCHS]
    )

    use_amp = config.USE_AMP and device.type == "cuda"
    # bfloat16 has the same exponent range as float32 (just less mantissa),
    # so it doesn't need a GradScaler and doesn't suffer the gradient
    # underflow float16 can — and it's more numerically stable for
    # LayerNorm-heavy architectures like ConvNeXt. Only fall back to
    # float16+GradScaler on GPUs that don't support bf16.
    if use_amp and torch.cuda.is_bf16_supported():
        amp_dtype, scaler = torch.bfloat16, None
    elif use_amp:
        amp_dtype, scaler = torch.float16, torch.amp.GradScaler(device.type)
    else:
        amp_dtype, scaler = None, None
    print(f"Mixed precision (AMP): {use_amp} (dtype={amp_dtype})")

    best_mean_auroc = 0.0
    best_state = None
    epochs_without_improvement = 0
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config.NUM_EPOCHS + 1):
        train_loss, train_auroc = run_epoch(
            model, train_loader, criterion, device, optimizer, scaler, amp_dtype
        )
        val_loss, val_auroc = run_epoch(
            model, val_loader, criterion, device, optimizer=None, scaler=None,
            amp_dtype=amp_dtype
        )
        scheduler.step()

        mean_val_auroc = val_auroc.mean().item()
        current_lr = scheduler.get_last_lr()[0]
        print(f"\nEpoch {epoch}/{config.NUM_EPOCHS}  (lr={current_lr:.2e})")
        print(f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"mean_val_AUROC={mean_val_auroc:.4f}")
        for finding, auc in zip(config.TARGET_FINDINGS, val_auroc.tolist()):
            print(f"    {finding:15s} AUROC={auc:.4f}")

        if mean_val_auroc > best_mean_auroc:
            best_mean_auroc = mean_val_auroc
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            print(f"  -> new best (mean AUROC {best_mean_auroc:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.EARLY_STOP_PATIENCE:
                print(f"\nEarly stopping: no improvement in "
                      f"{config.EARLY_STOP_PATIENCE} epochs.")
                break

    print(f"\nTraining complete. Best mean val AUROC: {best_mean_auroc:.4f}")

    # ---- Calibration: fit temperature scaling on the validation set ----
    model.load_state_dict(best_state)
    val_logits, val_labels = collect_logits(model, val_loader, device)
    val_logits, val_labels = val_logits.to(device), val_labels.to(device)

    ece_before = expected_calibration_error(
        torch.sigmoid(val_logits).cpu().numpy(), val_labels.cpu().numpy()
    )
    temperature = fit_temperature(val_logits, val_labels)
    ece_after = expected_calibration_error(
        torch.sigmoid(val_logits / temperature).cpu().numpy(), val_labels.cpu().numpy()
    )
    print(f"\nTemperature scaling: T={temperature:.3f}")
    print(f"  Expected Calibration Error before: {ece_before:.4f}")
    print(f"  Expected Calibration Error after:  {ece_after:.4f}")

    torch.save({
        "model_state_dict": best_state,
        "backbone": config.BACKBONE,
        "temperature": temperature,
        "target_findings": config.TARGET_FINDINGS,
    }, config.CHECKPOINT_PATH)
    print(f"Checkpoint saved to: {config.CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
