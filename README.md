# Subgroup Bias & Failure-Mode Audit of a Chest X-ray Classifier

## What this is, in plain terms

**The problem.** AI can now look at a chest X-ray and guess whether a patient has certain diseases,
often quite accurately. Hospitals are increasingly interested in using this kind of AI to help spot
disease faster. Normally, when someone builds a model like this, they report one number — "my model
is 85% accurate" — and that number is treated as the whole story.

**Why that's a problem.** One overall accuracy number can hide a lot. A model can be very accurate
*on average* while being noticeably worse for specific groups of patients — for example, older
patients, or patients whose X-ray was taken lying down in bed (which usually means they were sicker)
instead of standing up. If a hospital trusted the overall accuracy without checking this, the model
could quietly perform worse for certain patients without anyone noticing. That's not just a technical
detail — it's a patient-safety problem.

**Our solution.** We built a chest X-ray AI model, but instead of stopping at "here's the accuracy,"
we deliberately went looking for these hidden weak spots. We tested the model separately on different
patient groups (by sex, age, and how the X-ray was taken), used statistics to check whether any
differences we found were real or just chance, and visualized what the model was actually looking at
when it got things wrong. We also checked whether the model's confidence levels could be trusted, not
just whether its final answers were right.

**How we did it, simply put:**
- **The model — ConvNeXt-Tiny.** We used a modern, publicly available, pre-trained AI model
  (ConvNeXt-Tiny, via the widely-used `timm` library) and fine-tuned it on chest X-rays, rather than
  designing one from scratch. Think of it like hiring someone who already reads well and training them
  on your specific case files, instead of teaching someone to read from zero — it's faster and more
  reliable, and it's the standard approach in the field.
- **Making sure it's honest — calibration.** We checked whether the model's stated confidence matches
  reality: if it says "90% sure," is it actually right 90% of the time? Using a technique called
  temperature scaling, we corrected the model so its confidence numbers can be trusted, not just its
  final yes/no answer — important for any AI whose output might inform a medical decision.
- **Measuring fairness properly — Equalized Odds.** Instead of only comparing one summary accuracy
  score between patient groups, we also measured how often the model *missed* a real disease and how
  often it *wrongly flagged* a healthy patient as sick, separately for each group. This is a more
  direct, real-world way of asking "does this model treat different groups of patients the same way?"
  than a single accuracy number can answer.

---

**What this project actually does:** trains a multi-label chest X-ray classifier on NIH ChestX-ray14,
then rigorously audits *where and why it fails* — broken down by patient sex, age group, and
acquisition view (AP vs PA) — using per-subgroup metrics, statistical tests, and Captum-based
saliency maps to visualize what the model is actually looking at when it's wrong.

Most student projects stop at "here's my model's accuracy." This project's contribution is the
audit: showing that accuracy alone hides systematic failure patterns tied to *who* the patient is
and *how* the image was acquired — which is exactly the kind of finding EU AI Act-era "trustworthy
AI" reviewers and Erasmus Mundus admissions committees care about.

---

## Why this design (read this before you touch code)

- **Why multi-label, 5 findings, not all 14?** Keeps training fast (CPU-feasible, GPU trivial) while
  still being genuinely multi-pathology, which is more realistic than a toy binary task.
- **Why View Position as a subgroup axis, not just Sex/Age?** Most bias audits stop at demographics.
  AP vs PA view correlates with how sick/mobile a patient was when the X-ray was taken (portable AP
  scans are often taken on more severely ill, bedridden patients) — so a model can look "accurate"
  while actually partly keying off acquisition artifacts rather than pathology. This is the paper's
  actual novel angle.
- **Why ConvNeXt-Tiny via `timm`, not ResNet18?** `timm` is the standard library for pretrained
  vision backbones in both industry and research. ConvNeXt is a modern pure-conv
  architecture that matches/beats Vision Transformers at comparable size while keeping the training
  stability and lower data requirements of CNNs — a defensible choice for a ~15k-image fine-tuning
  task where a ViT would be more data-hungry.
- **Why patient-grouped train/val/test splits?** NIH ChestX-ray14 has multiple images per patient
  (follow-up scans). A row-level split lets the same patient appear in both train and test, leaking
  patient-specific appearance into the "held-out" set and inflating test metrics. We use
  scikit-learn's `GroupShuffleSplit` — the standard tool for exactly this — grouped by Patient ID.
- **Why pos_weight, cosine LR schedule, mixed precision, and early stopping?** Findings range from
  20.6% positive (Infiltration) to 2.9% (Cardiomegaly) — unweighted BCE under-trains the rare classes.
  Linear warmup + cosine decay is the standard modern fine-tuning schedule. AMP mixed precision is
  free speed/memory on any modern GPU. Early stopping on val AUROC treats the epoch count as a
  ceiling, not a target, so training doesn't run past convergence.
- **Why temperature scaling on top of AUROC?** AUROC measures ranking quality, not calibration — a
  model can rank correctly while still being overconfident. Temperature scaling  is
  fit post-hoc on the validation set and reported via Expected Calibration Error before/after, which
  is what "trustworthy AI" review actually expects beyond a bare accuracy number.
- **Why Equalized Odds gap alongside the AUROC gap?** AUROC gaps summarize ranking-quality
  differences across all thresholds; Equalized Odds reports the max difference
  in true-positive/false-positive rate at the actual 0.5 decision threshold a clinician would see —
  the standard fairness-literature metric, and the two can disagree.
- **Why Captum, not raw Grad-CAM from scratch or SHAP?** Captum is the maintained, PyTorch-native
  interpretability library (built by Meta), covers Grad-CAM AND Integrated Gradients with one
  consistent API, and is a legitimate, well-known tool you can defend in an interview without needing
  to explain three different libraries.
---

## 1. Get the data (do this first, outside this codebase)

NIH ChestX-ray14 is public, no registration required, hosted directly by NIH:

1. Go to: https://nihcc.app.box.com/v/ChestXray-NIHCC
2. Download `Data_Entry_2017_v2020.csv` (the metadata file — contains Image Index, Finding Labels,
   Patient Age, Patient Sex, View Position).
3. Download at least a few of the `images_00N.tar.gz` batches (there are 12 total, ~112,000 images,
   ~42GB total). **You do not need all 12** — 2-3 batches (~10-15k images) is plenty for this project
   and trains much faster.
4. Extract all images into a single flat folder: `data/images/`
5. Place the CSV at: `data/Data_Entry_2017_v2020.csv`

Final structure should look like:
```
data/
  Data_Entry_2017_v2020.csv
  images/
    00000001_000.png
    00000001_001.png
    ...
```

## 2. Environment setup

```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install torch torchvision timm captum torchmetrics pandas numpy scikit-learn \
            matplotlib seaborn pillow tqdm scipy tabulate
```

If you have an NVIDIA GPU, install the CUDA build of torch instead (check
https://pytorch.org/get-started/locally/ for the right command for your CUDA version).

## 3. Run the pipeline

```bash
# Step 1: prepare labels, splits, subgroup fields (age bins, etc.)
python src/prepare_data.py

# Step 2: train the classifier (uses GPU automatically if available)
python src/train.py

# Step 3: run the full subgroup bias audit (metrics + stats tests)
python src/bias_analysis.py

# Step 4: generate saliency maps for a sample of correct vs incorrect predictions
python src/interpretability.py

# Step 5: (optional) launch a tiny demo viewer
streamlit run src/app_streamlit.py
```

Each script prints what it's doing and why at each step — read the console output, not just the code.

## 4. What comes out of this

- `results/model_best.pt` — checkpoint dict: model weights, backbone name, fitted calibration
  temperature, target findings list
- `results/reports/subgroup_metrics.csv` — AUROC, sensitivity, specificity per finding, per subgroup
- `results/reports/statistical_tests.csv` — AUROC gap + p-value + Equalized Odds gap per subgroup pair
- `results/figures/` — bar charts of subgroup performance gaps + saliency map comparisons
- `results/reports/audit_summary.md` — auto-generated plain-English summary of your findings

## 5. What you should be able to explain afterward (use this as a self-check)

- Why AUROC per class, not just overall accuracy, for a multi-label imbalanced task
- What Integrated Gradients is actually computing (attribution relative to a baseline, not just "hot
  spots")
- Why a performance gap between subgroups doesn't automatically mean "bias" — confounding factors
  (e.g. AP scans correlating with sicker patients) need to be discussed, not just reported
- The difference between statistical significance and practical/clinical significance of a gap
- Why a patient-grouped split matters (patient-level leakage inflates test metrics) and how
  `GroupShuffleSplit` prevents it
- What temperature scaling corrects that AUROC can't tell you (calibration vs. ranking quality), and
  what Expected Calibration Error measures
- Why the Equalized Odds gap is reported alongside the AUROC gap instead of AUROC alone

If you can't answer these fluently, that's the signal to slow down and actually read the code with
me before moving to the next script — not a reason to skip ahead.
