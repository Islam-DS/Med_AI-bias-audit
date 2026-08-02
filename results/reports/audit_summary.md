# Subgroup Bias Audit — Summary

**Model:** convnext_tiny (calibration temperature T=0.785 — see `results/reports/` training log for Expected Calibration Error before/after scaling).

**14 statistically significant (p<0.05) subgroup performance gaps found** across 5 findings and 3 subgroup axes (subgroup pairs with fewer than 10 positive/negative cases on either side are excluded here and reported separately below — see 'Small-sample gaps').

## Significant gaps, largest first

- **Effusion**, by *age_group*: 80+ AUROC=0.611 vs 20-39 AUROC=0.895 (AUROC gap=0.284, p=0.0020, Equalized Odds gap=0.353)
- **Effusion**, by *age_group*: 80+ AUROC=0.611 vs 0-19 AUROC=0.862 (AUROC gap=0.251, p=0.0020, Equalized Odds gap=0.347)
- **Effusion**, by *age_group*: 40-59 AUROC=0.861 vs 80+ AUROC=0.611 (AUROC gap=0.250, p=0.0040, Equalized Odds gap=0.283)
- **Effusion**, by *age_group*: 60-79 AUROC=0.845 vs 80+ AUROC=0.611 (AUROC gap=0.234, p=<0.002, Equalized Odds gap=0.278)
- **Pneumothorax**, by *view_position*: PA AUROC=0.831 vs AP AUROC=0.680 (AUROC gap=0.151, p=0.0020, Equalized Odds gap=0.241)
- **Infiltration**, by *view_position*: PA AUROC=0.586 vs AP AUROC=0.713 (AUROC gap=0.127, p=<0.002, Equalized Odds gap=0.374)
- **Cardiomegaly**, by *view_position*: PA AUROC=0.920 vs AP AUROC=0.811 (AUROC gap=0.109, p=<0.002, Equalized Odds gap=0.302)
- **Effusion**, by *view_position*: PA AUROC=0.894 vs AP AUROC=0.791 (AUROC gap=0.103, p=<0.002, Equalized Odds gap=0.162)
- **Atelectasis**, by *view_position*: PA AUROC=0.832 vs AP AUROC=0.737 (AUROC gap=0.095, p=<0.002, Equalized Odds gap=0.119)
- **Pneumothorax**, by *sex*: Male AUROC=0.749 vs Female AUROC=0.844 (AUROC gap=0.095, p=0.0360, Equalized Odds gap=0.137)
- **Cardiomegaly**, by *age_group*: 60-79 AUROC=0.830 vs 40-59 AUROC=0.905 (AUROC gap=0.075, p=0.0240, Equalized Odds gap=0.047)
- **Infiltration**, by *age_group*: 60-79 AUROC=0.718 vs 40-59 AUROC=0.647 (AUROC gap=0.071, p=0.0080, Equalized Odds gap=0.115)
- **Atelectasis**, by *age_group*: 40-59 AUROC=0.789 vs 20-39 AUROC=0.851 (AUROC gap=0.063, p=0.0320, Equalized Odds gap=0.077)
- **Effusion**, by *age_group*: 60-79 AUROC=0.845 vs 20-39 AUROC=0.895 (AUROC gap=0.050, p=0.0480, Equalized Odds gap=0.075)

**Important caveat to include in your report:** a statistically significant AUROC gap does not by itself prove unfair bias — confounding factors (e.g. AP view correlating with sicker, less mobile patients) can produce a real gap that isn't 'the model is prejudiced,' but 'the model's performance is not uniform across the conditions under which the image was acquired.' Discuss which explanation your saliency maps support. The Equalized Odds gap (max difference in true-positive rate or false-positive rate between the two subgroups at the default 0.5 threshold) is reported alongside AUROC because AUROC summarizes ranking quality across all thresholds, while Equalized Odds reflects the actual operating point a clinician would see in practice — the two can disagree.

## Small-sample gaps (reported, not claimed as findings)

These involve a subgroup with too few positive/negative cases for AUROC to be a stable estimate (a single misranked sample can swing it by a large margin, e.g. one Pneumothorax case in an n=66 age group). Included for transparency, not as evidence of a subgroup effect:

- **Pneumothorax**, by *age_group*: 80+ AUROC=0.077 vs 20-39 AUROC=0.844 (gap=0.767, p=<0.002)
- **Pneumothorax**, by *age_group*: 40-59 AUROC=0.817 vs 80+ AUROC=0.077 (gap=0.740, p=0.0045)
- **Pneumothorax**, by *age_group*: 60-79 AUROC=0.792 vs 80+ AUROC=0.077 (gap=0.715, p=<0.002)
- **Pneumothorax**, by *age_group*: 80+ AUROC=0.077 vs 0-19 AUROC=0.639 (gap=0.562, p=0.0109)

## Full subgroup metrics table

| finding      | axis          | subgroup   |   n_samples |   n_positive |   prevalence |   AUROC |   AUROC_CI_low |   AUROC_CI_high |   sensitivity |   specificity | reliable   |
|:-------------|:--------------|:-----------|------------:|-------------:|-------------:|--------:|---------------:|----------------:|--------------:|--------------:|:-----------|
| Effusion     | sex           | Male       |        1853 |          205 |        0.111 |   0.862 |          0.836 |           0.884 |         0.659 |         0.87  | True       |
| Effusion     | sex           | Female     |        1484 |          201 |        0.135 |   0.859 |          0.834 |           0.885 |         0.622 |         0.867 | True       |
| Effusion     | age_group     | 60-79      |         911 |          146 |        0.16  |   0.845 |          0.815 |           0.875 |         0.623 |         0.852 | True       |
| Effusion     | age_group     | 40-59      |        1549 |          181 |        0.117 |   0.861 |          0.832 |           0.886 |         0.691 |         0.857 | True       |
| Effusion     | age_group     | 80+        |          66 |           12 |        0.182 |   0.611 |          0.479 |           0.742 |         0.583 |         0.574 | True       |
| Effusion     | age_group     | 20-39      |         623 |           44 |        0.071 |   0.895 |          0.858 |           0.931 |         0.568 |         0.927 | True       |
| Effusion     | age_group     | 0-19       |         188 |           23 |        0.122 |   0.862 |          0.762 |           0.938 |         0.522 |         0.921 | True       |
| Effusion     | view_position | PA         |        2039 |          194 |        0.095 |   0.894 |          0.871 |           0.917 |         0.639 |         0.928 | True       |
| Effusion     | view_position | AP         |        1298 |          212 |        0.163 |   0.791 |          0.763 |           0.82  |         0.642 |         0.766 | True       |
| Cardiomegaly | sex           | Male       |        1853 |           53 |        0.029 |   0.895 |          0.85  |           0.933 |         0.396 |         0.965 | True       |
| Cardiomegaly | sex           | Female     |        1484 |           70 |        0.047 |   0.861 |          0.825 |           0.894 |         0.329 |         0.934 | True       |
| Cardiomegaly | age_group     | 60-79      |         911 |           40 |        0.044 |   0.83  |          0.773 |           0.88  |         0.3   |         0.958 | True       |
| Cardiomegaly | age_group     | 40-59      |        1549 |           49 |        0.032 |   0.905 |          0.863 |           0.941 |         0.347 |         0.955 | True       |
| Cardiomegaly | age_group     | 80+        |          66 |            1 |        0.015 |   0.985 |          0.953 |           1     |         1     |         0.815 | False      |
| Cardiomegaly | age_group     | 20-39      |         623 |           14 |        0.022 |   0.905 |          0.874 |           0.934 |         0.143 |         0.952 | True       |
| Cardiomegaly | age_group     | 0-19       |         188 |           19 |        0.101 |   0.87  |          0.77  |           0.948 |         0.632 |         0.929 | True       |
| Cardiomegaly | view_position | PA         |        2039 |           58 |        0.028 |   0.92  |          0.886 |           0.95  |         0.517 |         0.959 | True       |
| Cardiomegaly | view_position | AP         |        1298 |           65 |        0.05  |   0.811 |          0.763 |           0.852 |         0.215 |         0.939 | True       |
| Atelectasis  | sex           | Male       |        1853 |          220 |        0.119 |   0.807 |          0.779 |           0.834 |         0.536 |         0.842 | True       |
| Atelectasis  | sex           | Female     |        1484 |          168 |        0.113 |   0.794 |          0.76  |           0.826 |         0.405 |         0.898 | True       |
| Atelectasis  | age_group     | 60-79      |         911 |          133 |        0.146 |   0.813 |          0.776 |           0.844 |         0.594 |         0.823 | True       |
| Atelectasis  | age_group     | 40-59      |        1549 |          174 |        0.112 |   0.789 |          0.756 |           0.821 |         0.477 |         0.859 | True       |
| Atelectasis  | age_group     | 80+        |          66 |           14 |        0.212 |   0.739 |          0.585 |           0.871 |         0.5   |         0.75  | True       |
| Atelectasis  | age_group     | 20-39      |         623 |           40 |        0.064 |   0.851 |          0.805 |           0.897 |         0.4   |         0.93  | True       |
| Atelectasis  | age_group     | 0-19       |         188 |           27 |        0.144 |   0.744 |          0.649 |           0.827 |         0.037 |         0.963 | True       |
| Atelectasis  | view_position | PA         |        2039 |          175 |        0.086 |   0.832 |          0.804 |           0.859 |         0.446 |         0.911 | True       |
| Atelectasis  | view_position | AP         |        1298 |          213 |        0.164 |   0.737 |          0.704 |           0.772 |         0.507 |         0.792 | True       |
| Infiltration | sex           | Male       |        1853 |          411 |        0.222 |   0.696 |          0.666 |           0.727 |         0.197 |         0.95  | True       |
| Infiltration | sex           | Female     |        1484 |          319 |        0.215 |   0.652 |          0.617 |           0.69  |         0.241 |         0.919 | True       |
| Infiltration | age_group     | 60-79      |         911 |          224 |        0.246 |   0.718 |          0.681 |           0.756 |         0.281 |         0.934 | True       |
| Infiltration | age_group     | 40-59      |        1549 |          324 |        0.209 |   0.647 |          0.615 |           0.683 |         0.167 |         0.928 | True       |
| Infiltration | age_group     | 80+        |          66 |           10 |        0.152 |   0.638 |          0.462 |           0.816 |         0     |         0.946 | True       |
| Infiltration | age_group     | 20-39      |         623 |          124 |        0.199 |   0.687 |          0.632 |           0.745 |         0.185 |         0.974 | True       |
| Infiltration | age_group     | 0-19       |         188 |           48 |        0.255 |   0.734 |          0.655 |           0.812 |         0.375 |         0.879 | True       |
| Infiltration | view_position | PA         |        2039 |          347 |        0.17  |   0.586 |          0.554 |           0.62  |         0.02  |         0.99  | True       |
| Infiltration | view_position | AP         |        1298 |          383 |        0.295 |   0.713 |          0.684 |           0.743 |         0.394 |         0.837 | True       |
| Pneumothorax | sex           | Male       |        1853 |           68 |        0.037 |   0.749 |          0.685 |           0.806 |         0.324 |         0.966 | True       |
| Pneumothorax | sex           | Female     |        1484 |           63 |        0.042 |   0.844 |          0.774 |           0.9   |         0.46  |         0.956 | True       |
| Pneumothorax | age_group     | 60-79      |         911 |           47 |        0.052 |   0.792 |          0.71  |           0.872 |         0.404 |         0.968 | True       |
| Pneumothorax | age_group     | 40-59      |        1549 |           50 |        0.032 |   0.817 |          0.743 |           0.883 |         0.4   |         0.956 | True       |
| Pneumothorax | age_group     | 80+        |          66 |            1 |        0.015 |   0.077 |          0.015 |           0.139 |         0     |         0.938 | False      |
| Pneumothorax | age_group     | 20-39      |         623 |           26 |        0.042 |   0.844 |          0.753 |           0.922 |         0.462 |         0.961 | True       |
| Pneumothorax | age_group     | 0-19       |         188 |            7 |        0.037 |   0.639 |          0.389 |           0.858 |         0     |         0.994 | False      |
| Pneumothorax | view_position | PA         |        2039 |           94 |        0.046 |   0.831 |          0.779 |           0.877 |         0.457 |         0.955 | True       |
| Pneumothorax | view_position | AP         |        1298 |           37 |        0.029 |   0.68  |          0.578 |           0.773 |         0.216 |         0.973 | True       |