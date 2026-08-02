"""
Step 3: THE CORE CONTRIBUTION OF THIS PROJECT.

Runs the trained model on the held-out test set, then breaks performance
down by subgroup (sex, age_group, view_position) instead of just reporting
one overall number. For each subgroup we compute AUROC, sensitivity,
specificity, and run a statistical test (bootstrapped CI + a permutation
test) on whether the subgroup gap is likely real or just noise.

Why bootstrapped CIs + permutation test, not just "look, the numbers are
different"? With ~1-2k test samples per subgroup, small differences are
easily just sampling noise. A gap you can't show is statistically
meaningful isn't a finding — it's an artifact you'd get called out on in
review. This is the difference between "I made a bar chart" and "I did an
audit."

Run: python src/bias_analysis.py
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, confusion_matrix
from tqdm import tqdm

import config
from dataset import ChestXrayDataset, load_and_split
from model import build_model, get_device


N_BOOTSTRAP = 1000
RANDOM_STATE = np.random.default_rng(config.RANDOM_SEED)


def get_predictions(model, loader, device, temperature: float = 1.0):
    """Run inference once, keep predictions + labels + subgroup metadata
    together so we can slice by any subgroup afterward without re-running
    the model. Applies temperature scaling (fit in train.py) so the audit's
    probability estimates are calibrated, not just correctly ranked."""
    model.eval()
    all_probs, all_labels = [], []
    all_sex, all_age, all_view, all_idx = [], [], [], []

    with torch.no_grad():
        for images, labels, subgroup in tqdm(loader, desc="Running inference"):
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits / temperature).cpu().numpy()

            all_probs.append(probs)
            all_labels.append(labels.numpy())
            all_sex.extend(subgroup["sex"])
            all_age.extend(subgroup["age_group"])
            all_view.extend(subgroup["view_position"])
            all_idx.extend(subgroup["image_index"])

    probs = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    meta = pd.DataFrame({
        "image_index": all_idx,
        "sex": all_sex,
        "age_group": all_age,
        "view_position": all_view,
    })
    return probs, labels, meta


def auroc_with_ci(y_true, y_score, n_bootstrap=N_BOOTSTRAP):
    """Bootstrap a 95% CI around AUROC. Returns (auroc, ci_low, ci_high).
    Returns NaNs if the subgroup has only one class present (AUROC
    undefined)."""
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan, np.nan

    point_estimate = roc_auc_score(y_true, y_score)

    n = len(y_true)
    boot_scores = []
    for _ in range(n_bootstrap):
        idx = RANDOM_STATE.integers(0, n, n)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        boot_scores.append(roc_auc_score(yt, ys))

    if len(boot_scores) < 50:
        return point_estimate, np.nan, np.nan

    ci_low, ci_high = np.percentile(boot_scores, [2.5, 97.5])
    return point_estimate, ci_low, ci_high


def sensitivity_specificity(y_true, y_score, threshold=0.5):
    y_pred = (y_score >= threshold).astype(int)
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    return sensitivity, specificity


def equalized_odds_gap(y_true_a, y_score_a, y_true_b, y_score_b, threshold=0.5):
    """Equalized Odds gap (Hardt et al., 2016): max(|TPR_a - TPR_b|,
    |FPR_a - FPR_b|). AUROC gaps summarize ranking quality but say nothing
    about the operating-point behavior a clinician actually sees at a fixed
    decision threshold — this is the standard fairness-literature metric for
    that, and it's what an ML-fairness-literate reviewer will look for
    alongside AUROC."""
    sens_a, spec_a = sensitivity_specificity(y_true_a, y_score_a, threshold)
    sens_b, spec_b = sensitivity_specificity(y_true_b, y_score_b, threshold)
    if any(np.isnan(x) for x in (sens_a, spec_a, sens_b, spec_b)):
        return np.nan
    fpr_a, fpr_b = 1 - spec_a, 1 - spec_b
    return max(abs(sens_a - sens_b), abs(fpr_a - fpr_b))


def permutation_test_gap(y_true_a, y_score_a, y_true_b, y_score_b,
                          n_permutations=500):
    """Test whether the AUROC gap between two subgroups (a vs b) is bigger
    than we'd expect if subgroup membership were random. Returns a p-value."""
    if len(np.unique(y_true_a)) < 2 or len(np.unique(y_true_b)) < 2:
        return np.nan

    observed_gap = abs(
        roc_auc_score(y_true_a, y_score_a) - roc_auc_score(y_true_b, y_score_b)
    )

    combined_true = np.concatenate([y_true_a, y_true_b])
    combined_score = np.concatenate([y_score_a, y_score_b])
    n_a = len(y_true_a)
    n_total = len(combined_true)

    perm_gaps = []
    for _ in range(n_permutations):
        perm_idx = RANDOM_STATE.permutation(n_total)
        group_a_idx = perm_idx[:n_a]
        group_b_idx = perm_idx[n_a:]
        yt_a, ys_a = combined_true[group_a_idx], combined_score[group_a_idx]
        yt_b, ys_b = combined_true[group_b_idx], combined_score[group_b_idx]
        if len(np.unique(yt_a)) < 2 or len(np.unique(yt_b)) < 2:
            continue
        gap = abs(roc_auc_score(yt_a, ys_a) - roc_auc_score(yt_b, ys_b))
        perm_gaps.append(gap)

    if len(perm_gaps) < 50:
        return np.nan

    p_value = np.mean(np.array(perm_gaps) >= observed_gap)
    if p_value == 0:
        # With a finite number of permutations, "p=0.000" overstates
        # precision — the test literally cannot resolve anything smaller
        # than 1/n_permutations. Report the resolution limit instead of a
        # number that reads as "impossible by chance," which it isn't.
        return f"<{1 / n_permutations:.3f}"
    return p_value


def audit_subgroup_axis(probs, labels, meta, axis: str, finding_idx: int,
                         finding_name: str):
    """For one finding and one subgroup axis (e.g. 'sex'), compute metrics
    per subgroup value and pairwise gap significance."""
    rows = []
    y_score_all = probs[:, finding_idx]
    y_true_all = labels[:, finding_idx]

    groups = meta[axis].dropna().unique()
    group_data = {}
    group_reliable = {}

    for g in groups:
        mask = (meta[axis] == g).values
        yt, ys = y_true_all[mask], y_score_all[mask]
        group_data[g] = (yt, ys)

        n_pos = int(yt.sum())
        n_neg = int(len(yt) - n_pos)
        reliable = min(n_pos, n_neg) >= config.MIN_POSITIVES_FOR_RELIABLE_AUROC
        group_reliable[g] = reliable

        auc, ci_low, ci_high = auroc_with_ci(yt, ys)
        sens, spec = sensitivity_specificity(yt, ys)
        rows.append({
            "finding": finding_name,
            "axis": axis,
            "subgroup": g,
            "n_samples": int(mask.sum()),
            "n_positive": n_pos,
            "prevalence": float(yt.mean()) if len(yt) else np.nan,
            "AUROC": auc,
            "AUROC_CI_low": ci_low,
            "AUROC_CI_high": ci_high,
            "sensitivity": sens,
            "specificity": spec,
            "reliable": reliable,
        })

    metrics_df = pd.DataFrame(rows)

    # Pairwise significance tests between subgroup values (e.g. Male vs
    # Female, or AP vs PA)
    sig_rows = []
    group_names = list(group_data.keys())
    for i in range(len(group_names)):
        for j in range(i + 1, len(group_names)):
            g1, g2 = group_names[i], group_names[j]
            yt1, ys1 = group_data[g1]
            yt2, ys2 = group_data[g2]
            p_value = permutation_test_gap(yt1, ys1, yt2, ys2)
            auc1, _, _ = auroc_with_ci(yt1, ys1, n_bootstrap=200)
            auc2, _, _ = auroc_with_ci(yt2, ys2, n_bootstrap=200)
            eo_gap = equalized_odds_gap(yt1, ys1, yt2, ys2)
            # p_value is either a float, NaN, or the string "<X" sentinel
            # (see permutation_test_gap) for a raw p_value of exactly 0 —
            # which is by definition < 0.05, so the string case is always
            # significant regardless of comparison operators that don't
            # work on strings.
            if isinstance(p_value, str):
                is_significant = True
            elif np.isnan(p_value):
                is_significant = False
            else:
                is_significant = p_value < 0.05
            sig_rows.append({
                "finding": finding_name,
                "axis": axis,
                "group_a": g1,
                "group_b": g2,
                "AUROC_a": auc1,
                "AUROC_b": auc2,
                "gap": abs(auc1 - auc2) if not (np.isnan(auc1) or np.isnan(auc2)) else np.nan,
                "p_value": p_value,
                "significant_p<0.05": is_significant,
                "equalized_odds_gap": eo_gap,
                "reliable": group_reliable[g1] and group_reliable[g2],
            })

    sig_df = pd.DataFrame(sig_rows)
    return metrics_df, sig_df


def main():
    device = get_device()
    print(f"Using device: {device}")

    if not config.CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"No trained model found at {config.CHECKPOINT_PATH}. "
            "Run `python src/train.py` first."
        )

    _train_df, _val_df, test_df = load_and_split()
    test_ds = ChestXrayDataset(test_df, train=False)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE,
                              shuffle=False, num_workers=4)

    checkpoint = torch.load(config.CHECKPOINT_PATH, map_location=device)
    model = build_model().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    temperature = checkpoint.get("temperature", 1.0)
    print(f"Loaded checkpoint (backbone={checkpoint.get('backbone', 'unknown')}, "
          f"calibration temperature={temperature:.3f})")

    probs, labels, meta = get_predictions(model, test_loader, device, temperature)

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    all_metrics, all_sig = [], []
    axes = ["sex", "age_group", "view_position"]

    for finding_idx, finding_name in enumerate(config.TARGET_FINDINGS):
        for axis in axes:
            print(f"\nAuditing '{finding_name}' across '{axis}'...")
            metrics_df, sig_df = audit_subgroup_axis(
                probs, labels, meta, axis, finding_idx, finding_name
            )
            all_metrics.append(metrics_df)
            all_sig.append(sig_df)

    metrics_out = pd.concat(all_metrics, ignore_index=True)
    sig_out = pd.concat(all_sig, ignore_index=True)

    metrics_path = config.REPORTS_DIR / "subgroup_metrics.csv"
    sig_path = config.REPORTS_DIR / "statistical_tests.csv"
    metrics_out.to_csv(metrics_path, index=False)
    sig_out.to_csv(sig_path, index=False)

    print(f"\nSaved subgroup metrics to {metrics_path}")
    print(f"Saved significance tests to {sig_path}")

    # Print the headline findings: statistically significant gaps, split by
    # whether both subgroups had enough positives for the AUROC estimate to
    # be trustworthy (see config.MIN_POSITIVES_FOR_RELIABLE_AUROC).
    sig_findings = sig_out[sig_out["significant_p<0.05"] == True]
    reliable_findings = sig_findings[sig_findings["reliable"] == True]
    unreliable_findings = sig_findings[sig_findings["reliable"] == False]

    print(f"\n{'='*60}")
    print(f"STATISTICALLY SIGNIFICANT SUBGROUP GAPS (p < 0.05): "
          f"{len(sig_findings)} found "
          f"({len(reliable_findings)} reliable, {len(unreliable_findings)} "
          f"small-sample - see below)")
    print(f"{'='*60}")
    cols = ["finding", "axis", "group_a", "group_b", "AUROC_a", "AUROC_b",
            "gap", "p_value", "equalized_odds_gap"]
    if len(reliable_findings) > 0:
        print(reliable_findings[cols].to_string(index=False))
    else:
        print("No statistically significant gaps found among subgroups with "
              "enough positive cases to trust the AUROC estimate - consider "
              "this a genuine (if less dramatic) finding: report it honestly "
              "rather than searching for a gap that isn't there.")
    if len(unreliable_findings) > 0:
        print(f"\n--- {len(unreliable_findings)} additional 'significant' "
              f"gaps involve a subgroup with fewer than "
              f"{config.MIN_POSITIVES_FOR_RELIABLE_AUROC} positive or negative "
              f"cases (e.g. age 80+) - a single misranked sample can swing "
              f"AUROC by a huge margin at that size. Reported for "
              f"transparency, not as findings: ---")
        print(unreliable_findings[cols].to_string(index=False))


if __name__ == "__main__":
    main()
