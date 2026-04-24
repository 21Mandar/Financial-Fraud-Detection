"""
preprocessing.py — Data loading, cleaning, feature engineering, and train/test splitting.

IEEE-CIS Fraud Detection Dataset
Expected folder structure:
    data/
        train_transaction.csv
        train_identity.csv

Usage:
    from preprocessing import load_and_preprocess
    X_train, X_test, y_cls_train, y_cls_test, y_reg_train, y_reg_test, feature_names = load_and_preprocess()
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------------

MISSING_THRESHOLD = 0.80        # Drop columns with >80% missing
TEST_SIZE = 0.20
RANDOM_STATE = 42

# Categorical columns we know from EDA
KNOWN_CAT_COLS = [
    'ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain',
    'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
    'id_12', 'id_15', 'id_16', 'id_23', 'id_27', 'id_28', 'id_29',
    'id_30', 'id_31', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38',
    'DeviceType', 'DeviceInfo'
]

# High-cardinality categoricals to frequency-encode rather than one-hot
HIGH_CARD_COLS = ['P_emaildomain', 'R_emaildomain', 'id_30', 'id_31',
                  'id_33', 'DeviceInfo']

# ---------------------------------------------------------------------------
# 2. LOADING
# ---------------------------------------------------------------------------

def load_data(data_dir='data'):
    """Load and merge transaction + identity tables."""
    tx_path = os.path.join(data_dir, 'train_transaction.csv')
    id_path = os.path.join(data_dir, 'train_identity.csv')

    print("[1/6] Loading transaction data ...")
    df_tx = pd.read_csv(tx_path)
    print(f"       Transactions: {df_tx.shape}")

    print("[2/6] Loading identity data ...")
    df_id = pd.read_csv(id_path)
    print(f"       Identity: {df_id.shape}")

    print("       Merging on TransactionID ...")
    df = df_tx.merge(df_id, on='TransactionID', how='left')
    print(f"       Merged: {df.shape}")
    return df

# ---------------------------------------------------------------------------
# 3. FEATURE ENGINEERING
# ---------------------------------------------------------------------------

def engineer_features(df):
    """Create derived features from raw data."""
    print("[3/6] Engineering features ...")

    # 3a. Time features from TransactionDT (seconds from reference point)
    df['TransactionHour'] = (df['TransactionDT'] / 3600) % 24
    df['TransactionDayOfWeek'] = (df['TransactionDT'] / (3600 * 24)) % 7

    # 3b. Log-transform TransactionAmt (used as regression target)
    df['LogTransactionAmt'] = np.log1p(df['TransactionAmt'])

    # 3c. Transaction amount decimal feature (fraud often has round amounts)
    df['TransactionAmt_decimal'] = (
        (df['TransactionAmt'] - df['TransactionAmt'].astype(int)) * 1000
    ).astype(int)

    # 3d. Card-level aggregates (mean amount per card1)
    card1_mean = df.groupby('card1')['TransactionAmt'].transform('mean')
    df['card1_TransactionAmt_mean'] = card1_mean
    df['card1_TransactionAmt_diff'] = df['TransactionAmt'] - card1_mean

    # 3e. addr1 + addr2 interaction
    df['addr1_addr2'] = df['addr1'].astype(str) + '_' + df['addr2'].astype(str)

    print(f"       Shape after feature engineering: {df.shape}")
    return df

# ---------------------------------------------------------------------------
# 4. CLEANING & ENCODING
# ---------------------------------------------------------------------------

def clean_and_encode(df):
    """Drop high-missing columns, encode categoricals, impute numerics."""
    print("[4/6] Cleaning and encoding ...")

    # 4a. Drop columns above missing threshold
    frac_missing = df.isnull().mean()
    cols_to_drop = frac_missing[frac_missing > MISSING_THRESHOLD].index.tolist()

    # Never drop the targets
    for col in ['isFraud', 'TransactionAmt', 'LogTransactionAmt']:
        if col in cols_to_drop:
            cols_to_drop.remove(col)

    df.drop(columns=cols_to_drop, inplace=True)
    print(f"       Dropped {len(cols_to_drop)} columns (>{MISSING_THRESHOLD*100:.0f}% missing)")

    # 4b. Identify remaining categorical columns present in df
    cat_cols = [c for c in KNOWN_CAT_COLS if c in df.columns]
    # Also catch any object / string columns not in the known list
    obj_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
    cat_cols = list(set(cat_cols + obj_cols))

    # 4c. Frequency-encode high-cardinality categoricals
    high_card_present = [c for c in HIGH_CARD_COLS if c in cat_cols]
    for col in high_card_present:
        freq = df[col].value_counts(normalize=True)
        df[col + '_freq'] = df[col].map(freq).astype(np.float32)
        cat_cols.remove(col)
        df.drop(columns=[col], inplace=True)

    # 4d. Label-encode remaining low-cardinality categoricals
    label_encoders = {}
    for col in cat_cols:
        if col not in df.columns:
            continue
        le = LabelEncoder()
        df[col] = df[col].astype(str)          # handle NaN as a category
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # 4e. Create missingness indicator features for columns with >5% missing
    remaining_missing = df.isnull().mean()
    indicator_cols = remaining_missing[
        (remaining_missing > 0.05) & (remaining_missing <= MISSING_THRESHOLD)
    ].index.tolist()
    for col in indicator_cols:
        if col not in ['isFraud', 'TransactionAmt', 'LogTransactionAmt']:
            df[col + '_missing'] = df[col].isnull().astype(np.int8)

    # 4f. Impute remaining numeric missing values with median
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    imputer = SimpleImputer(strategy='median')
    df[num_cols] = imputer.fit_transform(df[num_cols])

    print(f"       Shape after cleaning: {df.shape}")
    return df, label_encoders

# ---------------------------------------------------------------------------
# 5. SPLIT & SCALE
# ---------------------------------------------------------------------------

def split_and_scale(df):
    """Separate targets, split, and standardize features."""
    print("[5/6] Splitting and scaling ...")

    # Targets
    y_cls = df['isFraud'].values                       # binary classification
    y_reg = df['LogTransactionAmt'].values             # regression (log amt)

    # Drop targets + columns not useful for modeling
    drop_cols = ['isFraud', 'TransactionAmt', 'LogTransactionAmt',
                 'TransactionID', 'TransactionDT']
    drop_cols = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=drop_cols)

    feature_names = X.columns.tolist()

    # Train / test split — stratified on fraud label
    X_train, X_test, y_cls_train, y_cls_test, y_reg_train, y_reg_test = (
        train_test_split(X, y_cls, y_reg,
                         test_size=TEST_SIZE,
                         random_state=RANDOM_STATE,
                         stratify=y_cls)
    )

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"       Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"       Fraud rate — train: {y_cls_train.mean():.4f}, test: {y_cls_test.mean():.4f}")

    return (X_train, X_test,
            y_cls_train, y_cls_test,
            y_reg_train, y_reg_test,
            feature_names, scaler)

# ---------------------------------------------------------------------------
# 6. PUBLIC API
# ---------------------------------------------------------------------------

def load_and_preprocess(data_dir='data'):
    """
    Full pipeline: load → engineer → clean → split → scale.

    Returns
    -------
    X_train, X_test          : np.ndarray  (scaled features)
    y_cls_train, y_cls_test  : np.ndarray  (binary fraud labels)
    y_reg_train, y_reg_test  : np.ndarray  (log transaction amounts)
    feature_names            : list[str]
    scaler                   : fitted StandardScaler
    """
    df = load_data(data_dir)
    df = engineer_features(df)
    df, _ = clean_and_encode(df)
    result = split_and_scale(df)
    print("[6/6] Preprocessing complete!\n")
    return result


if __name__ == '__main__':
    # Quick test
    result = load_and_preprocess()
    X_tr, X_te, y_c_tr, y_c_te, y_r_tr, y_r_te, fnames, scaler = result
    print(f"Features: {len(fnames)}")
    print(f"Train samples: {X_tr.shape[0]}, Test samples: {X_te.shape[0]}")