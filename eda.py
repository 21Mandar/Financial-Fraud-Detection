"""
eda.py — Exploratory Data Analysis for IEEE-CIS Fraud Detection Dataset.

Generates all EDA figures for the final report and GitHub repo.
Run this script BEFORE the main pipeline.

Usage:
    python eda.py
    python eda.py --data_dir path/to/data

Output: All plots saved to figures/eda/
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

SAVE_DIR = 'figures/eda'
os.makedirs(SAVE_DIR, exist_ok=True)


def load_data(data_dir='data'):
    """Load and merge transaction + identity data."""
    print("Loading data ...")
    df_tx = pd.read_csv(os.path.join(data_dir, 'train_transaction.csv'))
    df_id = pd.read_csv(os.path.join(data_dir, 'train_identity.csv'))
    df = df_tx.merge(df_id, on='TransactionID', how='left')
    print(f"  Merged shape: {df.shape}")
    print(f"  Transactions: {len(df):,}")
    print(f"  Features: {df.shape[1]}")
    return df


# ---------------------------------------------------------------------------
# 1. DATASET OVERVIEW
# ---------------------------------------------------------------------------

def print_dataset_summary(df):
    """Print basic dataset statistics."""
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    print(f"\nShape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Data types
    dtypes = df.dtypes.value_counts()
    print(f"\nColumn types:")
    for dtype, count in dtypes.items():
        print(f"  {dtype}: {count}")

    # Target distribution
    fraud_counts = df['isFraud'].value_counts()
    fraud_rate = df['isFraud'].mean()
    print(f"\nTarget distribution (isFraud):")
    print(f"  Not Fraud (0): {fraud_counts[0]:,} ({1-fraud_rate:.2%})")
    print(f"  Fraud (1):     {fraud_counts[1]:,} ({fraud_rate:.2%})")

    # Transaction amount
    print(f"\nTransactionAmt statistics:")
    for label, group in df.groupby('isFraud')['TransactionAmt']:
        tag = 'Fraud' if label == 1 else 'Not Fraud'
        print(f"  {tag:10s} — mean: ${group.mean():.2f}, "
              f"median: ${group.median():.2f}, "
              f"max: ${group.max():,.2f}")

    # Missing values
    total_missing = df.isnull().sum().sum()
    total_cells = df.shape[0] * df.shape[1]
    print(f"\nMissing values: {total_missing:,} / {total_cells:,} "
          f"({total_missing/total_cells:.2%})")

    cols_with_missing = (df.isnull().sum() > 0).sum()
    print(f"Columns with any missing: {cols_with_missing} / {df.shape[1]}")


# ---------------------------------------------------------------------------
# 2. CLASS DISTRIBUTION
# ---------------------------------------------------------------------------

def plot_class_distribution(df):
    """Bar chart of fraud vs non-fraud."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Counts
    counts = df['isFraud'].value_counts()
    labels = ['Not Fraud', 'Fraud']
    colors = ['#4A90D9', '#FF6B6B']

    axes[0].bar(labels, counts.values, color=colors, edgecolor='white')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Fraud vs Non-Fraud (Absolute Counts)')
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 5000, f'{v:,}', ha='center', fontweight='bold')

    # Proportions
    props = [1 - df['isFraud'].mean(), df['isFraud'].mean()]
    axes[1].bar(labels, props, color=colors, edgecolor='white')
    axes[1].set_ylabel('Proportion')
    axes[1].set_title('Fraud vs Non-Fraud (Proportions)')
    for i, v in enumerate(props):
        axes[1].text(i, v + 0.01, f'{v:.2%}', ha='center', fontweight='bold')

    plt.suptitle('Class Imbalance — 96.5% Non-Fraud vs 3.5% Fraud',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'class_distribution.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {SAVE_DIR}/class_distribution.png")


# ---------------------------------------------------------------------------
# 3. MISSING VALUES
# ---------------------------------------------------------------------------

def plot_missing_values(df, top_n=40):
    """Bar chart of features by missingness percentage."""
    frac_missing = df.isnull().mean().sort_values(ascending=False)
    frac_missing = frac_missing[frac_missing > 0].head(top_n)

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(range(len(frac_missing)), frac_missing.values, color='#4A90D9')
    ax.set_xticks(range(len(frac_missing)))
    ax.set_xticklabels(frac_missing.index, rotation=90, fontsize=8)
    ax.set_ylabel('Fraction Missing')
    ax.set_title(f'Top {top_n} Features by Missing Percentage')
    ax.axhline(y=0.80, color='red', linestyle='--', label='80% threshold')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'missing_values.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {SAVE_DIR}/missing_values.png")

    # Count columns by missingness bucket
    all_missing = df.isnull().mean()
    print(f"  Columns >90% missing: {(all_missing > 0.90).sum()}")
    print(f"  Columns 80-90% missing: {((all_missing > 0.80) & (all_missing <= 0.90)).sum()}")
    print(f"  Columns 50-80% missing: {((all_missing > 0.50) & (all_missing <= 0.80)).sum()}")
    print(f"  Columns <50% missing: {((all_missing > 0) & (all_missing <= 0.50)).sum()}")
    print(f"  Columns with no missing: {(all_missing == 0).sum()}")


# ---------------------------------------------------------------------------
# 4. TRANSACTION AMOUNT DISTRIBUTION
# ---------------------------------------------------------------------------

def plot_transaction_amount(df):
    """Log(TransactionAmt) distribution by class."""
    df['LogAmt'] = np.log1p(df['TransactionAmt'])

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Side-by-side histograms
    for ax, (label, color, title) in zip(axes, [
        (0, '#4A90D9', 'Non-Fraud'), (1, '#FF6B6B', 'Fraud')
    ]):
        subset = df[df['isFraud'] == label]['LogAmt']
        ax.hist(subset, bins=80, color=color, edgecolor='white', alpha=0.85)
        ax.set_xlabel('log(TransactionAmt + 1)')
        ax.set_ylabel('Count')
        ax.set_title(f'{title} — Log(TransactionAmt) Distribution')
        ax.text(0.95, 0.95, f'n={len(subset):,}\nmean={subset.mean():.2f}\nstd={subset.std():.2f}',
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Transaction Amount Distributions by Class',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'transaction_amount_by_class.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {SAVE_DIR}/transaction_amount_by_class.png")

    # Overlapping KDE
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, color, name in [(0, '#4A90D9', 'Not Fraud'), (1, '#FF6B6B', 'Fraud')]:
        subset = df[df['isFraud'] == label]['LogAmt']
        subset.plot.kde(ax=ax, color=color, label=name, linewidth=2)
    ax.set_xlabel('log(TransactionAmt + 1)')
    ax.set_title('KDE of Log Transaction Amount — Fraud vs Non-Fraud')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'transaction_amount_kde.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {SAVE_DIR}/transaction_amount_kde.png")


# ---------------------------------------------------------------------------
# 5. TEMPORAL PATTERNS
# ---------------------------------------------------------------------------

def plot_temporal_patterns(df):
    """Transaction hour-of-day distribution by class."""
    df['TransactionHour'] = (df['TransactionDT'] / 3600) % 24

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for ax, (label, color, title) in zip(axes, [
        (0, '#FF9800', 'Non-Fraud by Hour'), (1, '#FFEB3B', 'Fraud by Hour')
    ]):
        subset = df[df['isFraud'] == label]['TransactionHour']
        ax.hist(subset, bins=48, color=color, edgecolor='white', alpha=0.85)
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Count')
        ax.set_title(title)

    plt.suptitle('Temporal Patterns — Transaction Volume by Hour',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'temporal_patterns.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {SAVE_DIR}/temporal_patterns.png")

    # Fraud rate by hour
    hourly_fraud = df.groupby(df['TransactionHour'].astype(int))['isFraud'].mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(hourly_fraud.index, hourly_fraud.values, color='#FF5722', edgecolor='white')
    ax.axhline(y=df['isFraud'].mean(), color='black', linestyle='--',
               label=f'Overall: {df["isFraud"].mean():.4f}')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Fraud Rate')
    ax.set_title('Fraud Rate by Hour of Day')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'fraud_rate_by_hour.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {SAVE_DIR}/fraud_rate_by_hour.png")


# ---------------------------------------------------------------------------
# 6. CATEGORICAL FEATURE ANALYSIS
# ---------------------------------------------------------------------------

def plot_categorical_fraud_rates(df):
    """Fraud rate by card network and product type."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Card network
    if 'card4' in df.columns:
        fraud_by_card = df.groupby('card4')['isFraud'].mean().sort_values(ascending=True)
        axes[0].barh(fraud_by_card.index, fraud_by_card.values, color='#4A90D9')
        axes[0].set_xlabel('Fraud Rate')
        axes[0].set_title('Fraud Rate by Card Network')
        for i, (idx, val) in enumerate(fraud_by_card.items()):
            axes[0].text(val + 0.002, i, f'{val:.3f}', va='center')

    # ProductCD
    if 'ProductCD' in df.columns:
        fraud_by_prod = df.groupby('ProductCD')['isFraud'].mean().sort_values(ascending=True)
        axes[1].barh(fraud_by_prod.index, fraud_by_prod.values, color='#4A90D9')
        axes[1].set_xlabel('Fraud Rate')
        axes[1].set_title('Fraud Rate by Product Type')
        for i, (idx, val) in enumerate(fraud_by_prod.items()):
            axes[1].text(val + 0.002, i, f'{val:.3f}', va='center')

    plt.suptitle('Fraud Rate by Categorical Features',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'categorical_fraud_rates.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {SAVE_DIR}/categorical_fraud_rates.png")


# ---------------------------------------------------------------------------
# 7. FEATURE CORRELATIONS
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(df, top_n=20):
    """Heatmap of top features correlated with isFraud."""
    # Select only numeric columns
    num_df = df.select_dtypes(include=[np.number])

    # Get top features by absolute correlation with target
    corr_with_target = num_df.corr()['isFraud'].abs().sort_values(ascending=False)
    top_features = corr_with_target.head(top_n + 1).index.tolist()  # +1 for isFraud itself

    corr_matrix = num_df[top_features].corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='Reds',
                square=True, ax=ax, annot_kws={'size': 7},
                linewidths=0.5)
    ax.set_title(f'Correlation Heatmap — Top {top_n} Features by Target Correlation',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'correlation_heatmap.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {SAVE_DIR}/correlation_heatmap.png")

    # Print top correlations
    print(f"\n  Top 10 features positively correlated with isFraud:")
    pos_corr = num_df.corr()['isFraud'].sort_values(ascending=False)
    for feat, corr in pos_corr.iloc[1:11].items():
        print(f"    {feat:20s}: {corr:.4f}")

    print(f"\n  Top 10 features negatively correlated with isFraud:")
    neg_corr = num_df.corr()['isFraud'].sort_values(ascending=True)
    for feat, corr in neg_corr.head(10).items():
        print(f"    {feat:20s}: {corr:.4f}")


# ---------------------------------------------------------------------------
# 8. V-FEATURE GROUP ANALYSIS
# ---------------------------------------------------------------------------

def plot_v_feature_groups(df):
    """Analyze the V-feature groups for multicollinearity."""
    v_cols = [c for c in df.columns if c.startswith('V')]
    if len(v_cols) == 0:
        print("  No V-features found.")
        return

    print(f"\n  V-features found: {len(v_cols)}")

    # Correlation among top V-features
    num_df = df[v_cols + ['isFraud']].select_dtypes(include=[np.number])
    top_v = num_df.corr()['isFraud'].abs().sort_values(ascending=False).head(16).index.tolist()
    if 'isFraud' in top_v:
        top_v.remove('isFraud')

    if len(top_v) > 2:
        corr_v = num_df[top_v].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_v, annot=True, fmt='.2f', cmap='coolwarm',
                    square=True, ax=ax, annot_kws={'size': 8})
        ax.set_title('Top V-Features: Inter-Correlation Matrix',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, 'v_feature_correlation.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {SAVE_DIR}/v_feature_correlation.png")

        # Identify highly correlated pairs
        pairs = []
        for i in range(len(top_v)):
            for j in range(i+1, len(top_v)):
                c = corr_v.iloc[i, j]
                if abs(c) > 0.90:
                    pairs.append((top_v[i], top_v[j], c))
        if pairs:
            print(f"\n  Highly correlated V-feature pairs (|r| > 0.90):")
            for a, b, c in sorted(pairs, key=lambda x: -abs(x[2])):
                print(f"    {a} ↔ {b}: {c:.3f}")


# ---------------------------------------------------------------------------
# 9. CATEGORICAL COLUMN SUMMARY
# ---------------------------------------------------------------------------

def print_categorical_summary(df):
    """Summarize categorical columns."""
    print("\n" + "=" * 60)
    print("CATEGORICAL FEATURES")
    print("=" * 60)

    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    print(f"\nNumber of categorical features: {len(cat_cols)}")

    for col in cat_cols:
        n_unique = df[col].nunique()
        missing_pct = df[col].isnull().mean() * 100
        print(f"  {col:25s}: {n_unique:5d} unique values, {missing_pct:5.1f}% missing")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='EDA for IEEE-CIS Fraud Detection')
    parser.add_argument('--data_dir', type=str, default='data')
    args = parser.parse_args()

    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("IEEE-CIS Fraud Detection Dataset")
    print("=" * 60)

    df = load_data(args.data_dir)

    print_dataset_summary(df)
    print_categorical_summary(df)

    print("\n" + "=" * 60)
    print("GENERATING EDA PLOTS")
    print("=" * 60)

    plot_class_distribution(df)
    plot_missing_values(df)
    plot_transaction_amount(df)
    plot_temporal_patterns(df)
    plot_categorical_fraud_rates(df)
    plot_correlation_heatmap(df)
    plot_v_feature_groups(df)

    print("\n" + "=" * 60)
    print(f"EDA COMPLETE — All plots saved to '{SAVE_DIR}/'")
    print("=" * 60)


if __name__ == '__main__':
    main()