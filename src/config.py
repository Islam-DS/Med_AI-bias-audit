"""
Central configuration. Deliberately a plain Python file, not Hydra/YAML —
for a project this size, one readable file beats a config framework.
Change values here rather than hunting through scripts.
"""

from pathlib import Path

# ---- Paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
METADATA_CSV = DATA_DIR / "Data_Entry_2017_v2020.csv"
PROCESSED_CSV = DATA_DIR / "processed_labels.csv"

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
REPORTS_DIR = RESULTS_DIR / "reports"
CHECKPOINT_PATH = RESULTS_DIR / "model_best.pt"

_image_index_cache = None


def build_image_index() -> dict:
    """Map filename -> full path for every .png under IMAGES_DIR.

    Downloaded NIH batches extract as nested folders (e.g.
    images_001/images/*.png) rather than one flat directory, and the exact
    set of batches you have varies by how many you downloaded. Scanning
    recursively means any batch layout works without manually flattening
    files. Cached at module level since IMAGES_DIR won't change mid-run and
    this is called from both prepare_data.py and dataset.py.
    """
    global _image_index_cache
    if _image_index_cache is None:
        _image_index_cache = {p.name: p for p in IMAGES_DIR.rglob("*.png")}
    return _image_index_cache

# ---- Task definition ---------------------------------------------------
# We deliberately restrict to 5 common findings rather than all 14 labels.
# This keeps training fast and the audit focused and interpretable, while
# still being a genuine multi-label problem (not a toy binary task).
TARGET_FINDINGS = [
    "Effusion",
    "Cardiomegaly",
    "Atelectasis",
    "Infiltration",
    "Pneumothorax",
]

# Subgroup axes we will audit performance across.
# Sex and age are the "expected" demographic axes.
# View Position (AP vs PA) is the deliberately less obvious axis: it's an
# acquisition characteristic, not a patient trait, but it correlates with
# how the scan was taken (portable/bedside AP scans vs standing PA scans),
# which itself correlates with patient severity/mobility. A model that
# performs very differently across AP/PA may be partly keying off that
# confound rather than true pathology signal.
AGE_BINS = [0, 20, 40, 60, 80, 120]
AGE_LABELS = ["0-19", "20-39", "40-59", "60-79", "80+"]

# Below this many positive (or negative) cases in a subgroup, AUROC is not a
# stable estimate — a single misranked sample can swing it by a huge margin
# (e.g. one Pneumothorax case in an n=66 age group). Subgroup comparisons
# below this threshold are flagged "unreliable" rather than treated as
# findings; see bias_analysis.py.
MIN_POSITIVES_FOR_RELIABLE_AUROC = 10

# ---- Model ---------------------------------------------------------------
# ConvNeXt-Tiny (Liu et al. 2022): a modern conv architecture that beats
# ResNet-family models at comparable parameter count, available pretrained
# via timm (the standard library for this in both industry and research —
# more current and better maintained than hand-picking torchvision models).
# ~28M params, comfortably fits an 8GB+ GPU at batch size 32 / 224px.
BACKBONE = "convnext_tiny"
DROP_PATH_RATE = 0.1  # stochastic depth: randomly drops entire residual
                      # branches during training. Standard regularizer for
                      # ConvNeXt fine-tuning; ~15k training images is small
                      # relative to this backbone's 28M params, so without it
                      # the model converges (and starts overfitting) within
                      # 3-4 epochs.
FREEZE_EARLY_STAGES = True  # freeze stem + first 2 of 4 ConvNeXt stages;
                             # only fine-tune the last 2 stages + head. See
                             # model.py for why.

# ---- Training ----------------------------------------------------------
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 25          # upper bound — early stopping (below) ends training
                         # once val AUROC stops improving, so this is a ceiling
                         # rather than a fixed budget.
EARLY_STOP_PATIENCE = 7  # stop if mean val AUROC hasn't improved in this many
                          # epochs; avoids wasting compute past convergence.
WARMUP_EPOCHS = 2         # linear LR warmup, then cosine decay — standard
                           # modern recipe for fine-tuning pretrained backbones.
LEARNING_RATE = 1e-4      # ConvNeXt's LayerNorms are more sensitive to
                          # aggressive LR than ResNet's BatchNorms during a
                          # full fine-tune; 3e-4 (fine for ResNet18) destabilized
                          # training and early-stopped at a worse optimum.
WEIGHT_DECAY = 1e-4
USE_AMP = True            # mixed-precision training (torch.autocast + GradScaler)
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
RANDOM_SEED = 42

# Use a subset of the full dataset if you only downloaded a few image
# batches. Set to None to use everything found in IMAGES_DIR.
MAX_SAMPLES = None
