"""
Post-hoc temperature scaling (Guo et al., 2017) for calibrating predicted
probabilities. A model can be discriminative (good AUROC — it ranks positives
above negatives) while being poorly calibrated (a predicted 0.8 doesn't
actually mean an 80% empirical positive rate). Reporting AUROC alone and
staying silent on calibration is exactly the kind of gap a "trustworthy AI"
reviewer would flag — a confidently wrong model is more dangerous in a
clinical decision-support context than an honestly uncertain one.

Fits a single scalar temperature T on the validation set: divide logits by T
before sigmoid at inference time. T > 1 softens overconfident predictions.
"""

import numpy as np
import torch
import torch.nn as nn


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 50) -> float:
    """logits, labels: (N, num_classes) tensors from the held-out validation set.

    Optimizes log(T) rather than T directly. T appears as a divisor of the
    logits, so the loss surface isn't convex in T globally and unconstrained
    LBFGS can walk straight through T=0 into negative territory (observed in
    practice: it converged to T=-101, which *inverted* every prediction and
    made calibration worse, not better). Optimizing log(T) and exponentiating
    guarantees T > 0 for any optimizer step, which is the standard fix.
    """
    log_temperature = nn.Parameter(torch.zeros(1, device=logits.device))
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.05, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature)
        loss = criterion(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return torch.exp(log_temperature).item()


def expected_calibration_error(probs, labels, n_bins: int = 10) -> float:
    """Mean absolute gap between predicted confidence and empirical positive
    rate, bucketed into n_bins — the standard scalar calibration summary
    metric (Naeini et al., 2015)."""
    probs = np.asarray(probs).ravel()
    labels = np.asarray(labels).ravel()

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (probs >= lo) & ((probs < hi) if i < n_bins - 1 else (probs <= hi))
        if mask.sum() == 0:
            continue
        bin_conf = probs[mask].mean()
        bin_acc = labels[mask].mean()
        ece += (mask.sum() / len(probs)) * abs(bin_conf - bin_acc)
    return float(ece)
