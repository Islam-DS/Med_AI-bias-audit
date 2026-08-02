"""
Step 5 (run after bias_analysis.py): turns the CSV outputs into bar chart
figures and a plain-English markdown summary — the thing you'd actually
paste into your SOP or a technical report appendix.

Run: python src/make_report.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

import config


def plot_subgroup_auroc(metrics: pd.DataFrame):
    for axis in metrics["axis"].unique():
        sub = metrics[metrics["axis"] == axis]
        plt.figure(figsize=(10, 5))
        sns.barplot(data=sub, x="finding", y="AUROC", hue="subgroup")
        plt.title(f"AUROC by {axis}, per finding")
        plt.ylim(0.4, 1.0)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        out_path = config.FIGURES_DIR / f"auroc_by_{axis}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved {out_path}")


def format_p(p) -> str:
    """p_value is a float, NaN, or the '<X' sentinel string (see
    bias_analysis.py's permutation_test_gap, which returns e.g. '<0.002'
    when the raw p-value is exactly 0). Once this column round-trips
    through a CSV, pandas stores the *entire* column as strings as soon as
    any row contains a non-numeric entry like '<0.002' — so even ordinary
    numeric p-values arrive here as strings (e.g. "0.048"), not floats.
    Handle all of these without crashing."""
    if isinstance(p, str):
        if p.startswith("<"):
            return p
        return f"{float(p):.4f}"
    return f"{p:.4f}"


def write_summary(metrics: pd.DataFrame, sig: pd.DataFrame, checkpoint: dict):
    lines = ["# Subgroup Bias Audit — Summary\n"]

    lines.append(
        f"**Model:** {checkpoint.get('backbone', 'unknown')} "
        f"(calibration temperature T={checkpoint.get('temperature', 1.0):.3f} — "
        "see `results/reports/` training log for Expected Calibration Error "
        "before/after scaling).\n"
    )

    all_sig_gaps = sig[sig["significant_p<0.05"] == True]
    sig_gaps = all_sig_gaps[all_sig_gaps["reliable"] == True].sort_values(
        "gap", ascending=False
    )
    unreliable_gaps = all_sig_gaps[all_sig_gaps["reliable"] == False].sort_values(
        "gap", ascending=False
    )

    lines.append(f"**{len(sig_gaps)} statistically significant (p<0.05) "
                  f"subgroup performance gaps found** across "
                  f"{sig['finding'].nunique()} findings and "
                  f"{sig['axis'].nunique()} subgroup axes "
                  f"(subgroup pairs with fewer than "
                  f"{config.MIN_POSITIVES_FOR_RELIABLE_AUROC} "
                  f"positive/negative cases on either side are excluded here "
                  f"and reported separately below — see 'Small-sample gaps').\n")

    if len(sig_gaps) > 0:
        lines.append("## Significant gaps, largest first\n")
        for _, row in sig_gaps.iterrows():
            lines.append(
                f"- **{row['finding']}**, by *{row['axis']}*: "
                f"{row['group_a']} AUROC={row['AUROC_a']:.3f} vs "
                f"{row['group_b']} AUROC={row['AUROC_b']:.3f} "
                f"(AUROC gap={row['gap']:.3f}, p={format_p(row['p_value'])}, "
                f"Equalized Odds gap={row['equalized_odds_gap']:.3f})"
            )
        lines.append("")
        lines.append(
            "**Important caveat to include in your report:** a statistically "
            "significant AUROC gap does not by itself prove unfair bias — "
            "confounding factors (e.g. AP view correlating with sicker, "
            "less mobile patients) can produce a real gap that isn't "
            "'the model is prejudiced,' but 'the model's performance is not "
            "uniform across the conditions under which the image was "
            "acquired.' Discuss which explanation your saliency maps support. "
            "The Equalized Odds gap (max difference in true-positive rate or "
            "false-positive rate between the two subgroups at the default "
            "0.5 threshold) is reported alongside AUROC because AUROC "
            "summarizes ranking quality across all thresholds, while "
            "Equalized Odds reflects the actual operating point a clinician "
            "would see in practice — the two can disagree."
        )
    else:
        lines.append(
            "No statistically significant gaps were found among subgroups "
            "large enough to trust the AUROC estimate. This is a legitimate, "
            "reportable finding — it's worth stating the sample size and "
            "power limitations honestly rather than implying the model is "
            "bias-free."
        )

    if len(unreliable_gaps) > 0:
        lines.append("\n## Small-sample gaps (reported, not claimed as findings)\n")
        lines.append(
            "These involve a subgroup with too few positive/negative cases "
            "for AUROC to be a stable estimate (a single misranked sample "
            "can swing it by a large margin, e.g. one Pneumothorax case in "
            "an n=66 age group). Included for transparency, not as evidence "
            "of a subgroup effect:\n"
        )
        for _, row in unreliable_gaps.iterrows():
            lines.append(
                f"- **{row['finding']}**, by *{row['axis']}*: "
                f"{row['group_a']} AUROC={row['AUROC_a']:.3f} vs "
                f"{row['group_b']} AUROC={row['AUROC_b']:.3f} "
                f"(gap={row['gap']:.3f}, p={format_p(row['p_value'])})"
            )

    lines.append("\n## Full subgroup metrics table\n")
    lines.append(metrics.round(3).to_markdown(index=False))

    out_path = config.REPORTS_DIR / "audit_summary.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved plain-English summary to {out_path}")


def main():
    metrics = pd.read_csv(config.REPORTS_DIR / "subgroup_metrics.csv")
    sig = pd.read_csv(config.REPORTS_DIR / "statistical_tests.csv")
    checkpoint = torch.load(config.CHECKPOINT_PATH, map_location="cpu")

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_subgroup_auroc(metrics)
    write_summary(metrics, sig, checkpoint)


if __name__ == "__main__":
    main()
