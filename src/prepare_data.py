"""
Step 1: turn the raw NIH metadata CSV into a clean table we can train and
audit from.

The raw CSV has columns like:
    Image Index, Finding Labels, Follow-up #, Patient ID, Patient Age,
    Patient Sex, View Position, ...
where "Finding Labels" is a pipe-separated string e.g. "Effusion|Infiltration"
or "No Finding".

We convert this into:
    - one binary column per target finding (multi-hot encoding)
    - a clean age-bin column
    - a filtered set of rows where the image file actually exists on disk
      (since you likely only downloaded a subset of the 12 image batches)

Run: python src/prepare_data.py
"""

import pandas as pd
from pathlib import Path

import config


def load_raw_metadata() -> pd.DataFrame:
    if not config.METADATA_CSV.exists():
        raise FileNotFoundError(
            f"Could not find {config.METADATA_CSV}.\n"
            "Download Data_Entry_2017_v2020.csv from "
            "https://nihcc.app.box.com/v/ChestXray-NIHCC and place it in data/. "
            "See README.md section 1."
        )
    df = pd.read_csv(config.METADATA_CSV)
    return df


def filter_to_downloaded_images(df: pd.DataFrame) -> pd.DataFrame:
    """Only keep rows for images you actually downloaded (you likely only
    grabbed a few of the 12 batches, not the full 42GB)."""
    if not config.IMAGES_DIR.exists():
        raise FileNotFoundError(
            f"Could not find {config.IMAGES_DIR}. Extract the downloaded "
            "images_00N.tar.gz batches into data/images/. See README.md."
        )
    available = set(config.build_image_index().keys())
    before = len(df)
    df = df[df["Image Index"].isin(available)].copy()
    print(f"Filtered metadata from {before} rows to {len(df)} rows "
          f"matching images actually present in {config.IMAGES_DIR}")
    if len(df) == 0:
        raise RuntimeError(
            "No matching images found. Check that data/images/ contains "
            ".png files whose names match the 'Image Index' column."
        )
    return df


def add_multilabel_columns(df: pd.DataFrame) -> pd.DataFrame:
    for finding in config.TARGET_FINDINGS:
        df[finding] = df["Finding Labels"].apply(
            lambda s: 1 if finding in str(s).split("|") else 0
        )
    return df


def add_subgroup_columns(df: pd.DataFrame) -> pd.DataFrame:
    df["Patient Age"] = pd.to_numeric(df["Patient Age"], errors="coerce")
    # NIH data has some clearly erroneous ages (e.g. >120); drop those rows
    df = df[df["Patient Age"].between(0, 120)].copy()

    df["age_group"] = pd.cut(
        df["Patient Age"], bins=config.AGE_BINS, labels=config.AGE_LABELS,
        right=False
    )
    df["sex"] = df["Patient Sex"].map({"M": "Male", "F": "Female"})
    df["view_position"] = df["View Position"]  # already 'AP' or 'PA'
    return df


def main():
    df = load_raw_metadata()
    df = filter_to_downloaded_images(df)
    df = add_multilabel_columns(df)
    df = add_subgroup_columns(df)

    if config.MAX_SAMPLES:
        df = df.sample(n=min(config.MAX_SAMPLES, len(df)),
                        random_state=config.RANDOM_SEED)

    keep_cols = (["Image Index", "Patient ID", "sex", "age_group", "view_position"]
                 + config.TARGET_FINDINGS)
    df_out = df[keep_cols].reset_index(drop=True)

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(config.PROCESSED_CSV, index=False)

    print(f"\nSaved processed labels to {config.PROCESSED_CSV}")
    print(f"Total samples: {len(df_out)}")
    print("\nPositive rate per finding:")
    for f in config.TARGET_FINDINGS:
        rate = df_out[f].mean()
        print(f"  {f:15s}: {rate:.1%}")
    print("\nSubgroup distribution:")
    print(df_out["sex"].value_counts())
    print(df_out["view_position"].value_counts())
    print(df_out["age_group"].value_counts().sort_index())


if __name__ == "__main__":
    main()
