"""
automate_RiecoEdward.py
=======================
Script otomasi preprocessing dataset Heart Disease UCI.
Menghasilkan dataset siap latih di folder dataset_preprocessing/.

Author  : Rieco Edward
Dataset : Heart Disease UCI (Cleveland)
Python  : 3.12.7
"""

import os
import sys
import logging
import argparse
import warnings
import io
import requests
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import joblib

warnings.filterwarnings("ignore")

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("preprocessing.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)

# Constants
RAW_PATH   = os.path.join(os.path.dirname(__file__), "..", "dataset_raw", "heart.csv")
OUT_DIR    = os.path.join(os.path.dirname(__file__), "dataset_preprocessing")
RANDOM_STATE = 42
TEST_SIZE    = 0.20
VAL_SIZE     = 0.10   # fraction of total (split from train)

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
TARGET = "target"

# 1. Load Data
def load_data(path: str) -> pd.DataFrame:
    """Load raw CSV. Falls back to UCI download (multi-URL) if file not found."""

    UCI_COLUMNS = [
        "age", "sex", "cp", "trestbps", "chol",
        "fbs", "restecg", "thalach", "exang",
        "oldpeak", "slope", "ca", "thal", "target"
    ]

    FALLBACK_URLS = [
        (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/"
            "heart-disease/processed.cleveland.data",
            "uci_raw"
        ),
    ]

    if os.path.exists(path):
        log.info(f"Loading local file: {path}")
        df = pd.read_csv(path)
    else:
        log.warning(f"File not found at {path}. Mencoba download...")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df = None

        for url, fmt in FALLBACK_URLS:
            try:
                log.info(f"  Trying: {url}")
                resp = requests.get(url, timeout=20)
                resp.raise_for_status()
                if fmt == "uci_raw":
                    df = pd.read_csv(
                        io.StringIO(resp.text),
                        header=None,
                        names=UCI_COLUMNS,
                        na_values="?"
                    )
                else:
                    df = pd.read_csv(io.StringIO(resp.text))
                log.info(f"  Berhasil download. Shape: {df.shape}")
                break
            except Exception as e:
                log.warning(f"  Gagal ({url}): {e}")

        if df is None:
            log.warning("Semua URL gagal. Membuat dataset synthetic untuk demo...")
            rng = np.random.default_rng(42)
            n = 303
            df = pd.DataFrame({
                "age"     : rng.integers(29, 77, n).astype(float),
                "sex"     : rng.integers(0, 2, n).astype(float),
                "cp"      : rng.integers(0, 4, n).astype(float),
                "trestbps": rng.integers(94, 200, n).astype(float),
                "chol"    : rng.integers(126, 564, n).astype(float),
                "fbs"     : rng.integers(0, 2, n).astype(float),
                "restecg" : rng.integers(0, 3, n).astype(float),
                "thalach" : rng.integers(71, 202, n).astype(float),
                "exang"   : rng.integers(0, 2, n).astype(float),
                "oldpeak" : rng.uniform(0, 6.2, n).round(1),
                "slope"   : rng.integers(0, 3, n).astype(float),
                "ca"      : rng.integers(0, 4, n).astype(float),
                "thal"    : rng.choice([3.0, 6.0, 7.0], n),
                "target"  : rng.integers(0, 5, n).astype(float),
            })

        # Normalisasi kolom UCI Cleveland format
        if df["thal"].max() > 3:
            thal_map = {3.0: 1, 6.0: 2, 7.0: 3}
            df["thal"] = df["thal"].map(thal_map).fillna(df["thal"])
        df["ca"] = df["ca"].clip(0, 3)

        df.to_csv(path, index=False)
        log.info(f"Saved to {path}")

    log.info(f"Loaded dataset shape: {df.shape}")
    return df 

# 2. Validate Schema
def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all expected columns exist; rename if necessary."""
    # Beberapa versi dataset menggunakan 'condition' alih-alih 'target'
    if "condition" in df.columns and TARGET not in df.columns:
        df = df.rename(columns={"condition": TARGET})
        log.info("Renamed column 'condition' → 'target'")

    expected_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    log.info("Schema validation passed.")
    return df

# 3. Handle Missing Values
def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values: median for numeric, mode for categorical."""
    before = df.isnull().sum().sum()
    log.info(f"Missing values before imputation: {before}")

    num_imputer = SimpleImputer(strategy="median")
    df[NUMERIC_FEATURES] = num_imputer.fit_transform(df[NUMERIC_FEATURES])

    cat_imputer = SimpleImputer(strategy="most_frequent")
    df[CATEGORICAL_FEATURES] = cat_imputer.fit_transform(df[CATEGORICAL_FEATURES])

    after = df.isnull().sum().sum()
    log.info(f"Missing values after imputation: {after}")
    return df

# 4. Remove Duplicates
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    log.info(f"Duplicates removed: {removed} rows  (before={before}, after={len(df)})")
    return df

# 5. Outlier Handling (IQR Capping)
def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Cap outliers at IQR fence per numeric feature."""
    for col in NUMERIC_FEATURES:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        clipped = ((df[col] < lower) | (df[col] > upper)).sum()
        df[col] = df[col].clip(lower, upper)
        if clipped:
            log.info(f"  {col}: {clipped} outlier(s) capped to [{lower:.2f}, {upper:.2f}]")
    return df

# 6. Encode Target
def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Binarize target: 0 = no disease, 1 = disease."""
    df[TARGET] = (df[TARGET] > 0).astype(int)
    log.info(f"Target distribution:\n{df[TARGET].value_counts().to_string()}")
    return df

# 7. Encode Categorical Features
def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode multi-class categoricals; keep binary as-is."""
    binary_cats  = ["sex", "fbs", "exang"]
    ohe_cats     = ["cp", "restecg", "slope", "ca", "thal"]

    # Binary: just ensure int
    for col in binary_cats:
        df[col] = df[col].astype(int)

    # One-hot encoding
    df = pd.get_dummies(df, columns=ohe_cats, drop_first=False, dtype=int)
    log.info(f"Shape after OHE: {df.shape}")
    return df

# 8. Feature Engineering
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Create domain-informed derived features."""
    # Age group buckets
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 40, 55, 65, 120],
        labels=[0, 1, 2, 3]
    ).astype(int)

    # Cholesterol-to-max-heart-rate ratio
    df["chol_thalach_ratio"] = df["chol"] / (df["thalach"] + 1e-8)

    # Blood pressure risk flag (≥140 mmHg systolic)
    df["high_bp"] = (df["trestbps"] >= 140).astype(int)

    log.info(f"Feature engineering complete. New shape: {df.shape}")
    return df

# 9. Scale Numeric Features
def scale_features(
    X_train: pd.DataFrame,
    X_val:   pd.DataFrame,
    X_test:  pd.DataFrame,
    out_dir: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """StandardScaler fit on train, transform all splits."""
    numeric_cols = [
        c for c in NUMERIC_FEATURES + ["chol_thalach_ratio"]
        if c in X_train.columns
    ]
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_val[numeric_cols]   = scaler.transform(X_val[numeric_cols])
    X_test[numeric_cols]  = scaler.transform(X_test[numeric_cols])

    scaler_path = os.path.join(out_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    log.info(f"Scaler saved to {scaler_path}")
    return X_train, X_val, X_test

# 10. Split & Save
def split_and_save(df: pd.DataFrame, out_dir: str) -> None:
    """Train/val/test split and save to CSV."""
    os.makedirs(out_dir, exist_ok=True)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # First split: train+val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    # Second split: train vs val
    val_relative = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_relative,
        random_state=RANDOM_STATE,
        stratify=y_trainval
    )

    # Scale
    X_train, X_val, X_test = scale_features(X_train, X_val, X_test, out_dir)

    # Reconstruct DataFrames
    train_df = X_train.copy(); train_df[TARGET] = y_train.values
    val_df   = X_val.copy();   val_df[TARGET]   = y_val.values
    test_df  = X_test.copy();  test_df[TARGET]  = y_test.values

    # Save
    train_df.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    val_df.to_csv(  os.path.join(out_dir, "val.csv"),   index=False)
    test_df.to_csv( os.path.join(out_dir, "test.csv"),  index=False)

    log.info(
        f"Datasets saved to {out_dir}\n"
        f"  train : {train_df.shape}\n"
        f"  val   : {val_df.shape}\n"
        f"  test  : {test_df.shape}"
    )

    # Feature list
    feature_path = os.path.join(out_dir, "feature_names.txt")
    with open(feature_path, "w") as f:
        f.write("\n".join(X_train.columns.tolist()))
    log.info(f"Feature names saved to {feature_path}")

# Main Pipeline
def run_pipeline(raw_path: str = RAW_PATH, out_dir: str = OUT_DIR) -> None:
    log.info("=" * 55)
    log.info("  Heart Disease Preprocessing Pipeline — Starting  ")
    log.info("=" * 55)

    df = load_data(raw_path)
    df = validate_schema(df)
    df = handle_missing(df)
    df = remove_duplicates(df)
    df = handle_outliers(df)
    df = encode_target(df)
    df = encode_categoricals(df)
    df = feature_engineering(df)
    split_and_save(df, out_dir)

    log.info("=" * 55)
    log.info("  Pipeline completed successfully!                  ")
    log.info("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Heart Disease Preprocessing Pipeline")
    parser.add_argument("--raw",     default=RAW_PATH, help="Path to raw CSV")
    parser.add_argument("--out_dir", default=OUT_DIR,  help="Output directory")
    args = parser.parse_args()
    run_pipeline(args.raw, args.out_dir)