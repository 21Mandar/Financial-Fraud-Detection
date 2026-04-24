"""
evaluation.py — Model evaluation, comparison, and visualization.

Generates ALL plots needed for the final report:
    - Regression comparison bar charts (MAE, RMSE, R²)
    - Classification comparison grouped bars (AUC-ROC, PR-AUC, F1)
    - With vs without anomaly scores delta chart (Q3)
    - ROC curves (all models overlaid)
    - Precision-Recall curves (all models overlaid)
    - Confusion matrices grid
    - Residual-fraud correlation analysis (Q2)
    - SHAP feature importance (bee swarm + bar)
    - Text classification reports

Usage:
    from evaluation import run_evaluation
    run_evaluation(reg_results, cls_results, y_test=y_test,
                   predictions_dict=preds, ...)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc, roc_auc_score,
    confusion_matrix, classification_report
)
import os
import warnings
warnings.filterwarnings("ignore")

SAVE_DIR = 'figures'
os.makedirs(SAVE_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. REGRESSION COMPARISON (Q1)
# ---------------------------------------------------------------------------

def plot_regression_comparison(reg_results):
    """Bar chart comparing regressor performance."""
    df = pd.DataFrame(reg_results)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = ['MAE', 'RMSE', 'R2']
    titles = ['Mean Absolute Error ↓', 'Root Mean Squared Error ↓', 'R² Score ↑']
    colors = ['#2196F3', '#FF9800', '#4CAF50']

    for ax, metric, title, color in zip(axes, metrics, titles, colors):
        bars = ax.barh(df['model'], df[metric], color=color, edgecolor='white')
        ax.set_xlabel(metric)
        ax.set_title(title)
        for bar, val in zip(bars, df[metric]):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}', va='center', fontsize=10)

    plt.suptitle('Stage 1: Regressor Comparison (Q1)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'regression_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {SAVE_DIR}/regression_comparison.png")


# ---------------------------------------------------------------------------
# 2. CLASSIFICATION COMPARISON (Q3 grouped bars)
# ---------------------------------------------------------------------------

def plot_classification_comparison(cls_results):
    """Grouped bar chart comparing classifiers base vs +anomaly."""
    df = pd.DataFrame(cls_results)
    df_base = df[df['anomaly_scores'] == False].copy()
    df_aug = df[df['anomaly_scores'] == True].copy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metrics = ['AUC_ROC', 'PR_AUC', 'F1']
    titles = ['AUC-ROC ↑', 'PR-AUC ↑', 'F1 Score ↑']

    for ax, metric, title in zip(axes, metrics, titles):
        base_names = [m.replace(' (base)', '') for m in df_base['model']]
        x = np.arange(len(base_names))
        width = 0.35
        ax.bar(x - width/2, df_base[metric].values, width,
               label='Base', color='#2196F3', edgecolor='white')
        if len(df_aug) > 0:
            ax.bar(x + width/2, df_aug[metric].values, width,
                   label='+Anomaly Scores', color='#FF5722', edgecolor='white')
        ax.set_xlabel('Model')
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(base_names, rotation=30, ha='right')
        ax.legend()

    plt.suptitle('Stage 2: Classifier Comparison — Base vs +Anomaly Scores (Q3)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'classification_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {SAVE_DIR}/classification_comparison.png")


# ---------------------------------------------------------------------------
# 3. ROC CURVES
# ---------------------------------------------------------------------------

def plot_roc_curves(y_test, predictions_dict):
    """Overlay ROC curves for all models."""
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(predictions_dict)))

    for (name, y_proba), color in zip(predictions_dict.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_val = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f'{name} (AUC = {auc_val:.4f})')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves — All Classifiers', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.01])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'roc_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {SAVE_DIR}/roc_curves.png")


# ---------------------------------------------------------------------------
# 4. PRECISION-RECALL CURVES
# ---------------------------------------------------------------------------

def plot_pr_curves(y_test, predictions_dict):
    """Overlay Precision-Recall curves for all models."""
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(predictions_dict)))
    baseline = y_test.mean()

    for (name, y_proba), color in zip(predictions_dict.items(), colors):
        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = auc(rec, prec)
        ax.plot(rec, prec, color=color, linewidth=2,
                label=f'{name} (PR-AUC = {pr_auc:.4f})')

    ax.axhline(y=baseline, color='gray', linestyle='--', linewidth=1,
               label=f'Baseline (fraud rate = {baseline:.4f})')
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves — All Classifiers',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.01])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'pr_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {SAVE_DIR}/pr_curves.png")


# ---------------------------------------------------------------------------
# 5. CONFUSION MATRICES
# ---------------------------------------------------------------------------

def plot_confusion_matrices(y_test, predictions_dict, threshold=0.5):
    """Confusion matrix grid for all models."""
    n = len(predictions_dict)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4.5*rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, (name, y_proba) in enumerate(predictions_dict.items()):
        y_pred = (y_proba >= threshold).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt=',d', cmap='Blues', ax=axes[i],
                    xticklabels=['Not Fraud', 'Fraud'],
                    yticklabels=['Not Fraud', 'Fraud'])
        axes[i].set_title(name, fontsize=10, fontweight='bold')
        axes[i].set_ylabel('Actual')
        axes[i].set_xlabel('Predicted')

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Confusion Matrices (threshold = 0.5)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'confusion_matrices.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {SAVE_DIR}/confusion_matrices.png")


# ---------------------------------------------------------------------------
# 6. TEXT CLASSIFICATION REPORTS
# ---------------------------------------------------------------------------

def save_classification_reports(y_test, predictions_dict, threshold=0.5):
    """Save detailed sklearn classification reports to a text file."""
    path = os.path.join(SAVE_DIR, 'classification_reports.txt')
    with open(path, 'w') as f:
        for name, y_proba in predictions_dict.items():
            y_pred = (y_proba >= threshold).astype(int)
            f.write(f"\n{'='*60}\n  {name}\n{'='*60}\n")
            f.write(classification_report(y_test, y_pred,
                    target_names=['Not Fraud', 'Fraud']))
            f.write("\n")
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# 7. ANOMALY SCORE IMPACT (Q3 delta)
# ---------------------------------------------------------------------------

def plot_anomaly_score_impact(cls_results):
    """Delta chart: AUC-ROC improvement from anomaly scores."""
    df = pd.DataFrame(cls_results)
    df_base = df[df['anomaly_scores'] == False].copy()
    df_aug = df[df['anomaly_scores'] == True].copy()

    if len(df_aug) == 0:
        print("  No augmented results to compare.")
        return

    df_base['clean_name'] = df_base['model'].str.replace(' (base)', '', regex=False)
    df_aug['clean_name'] = df_aug['model'].str.replace(' (+anomaly)', '', regex=False)
    merged = df_base.merge(df_aug, on='clean_name', suffixes=('_base', '_aug'))

    fig, ax = plt.subplots(figsize=(10, 5))
    deltas = merged['AUC_ROC_aug'] - merged['AUC_ROC_base']
    colors = ['#4CAF50' if d > 0 else '#F44336' for d in deltas]
    bars = ax.barh(merged['clean_name'], deltas, color=colors, edgecolor='white')
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_xlabel('ΔAUC-ROC (with anomaly scores − without)')
    ax.set_title('Research Question 3: Impact of Anomaly Scores on AUC-ROC')
    for bar, d in zip(bars, deltas):
        xpos = bar.get_width() + 0.001 if d >= 0 else bar.get_width() - 0.001
        ha = 'left' if d >= 0 else 'right'
        ax.text(xpos, bar.get_y() + bar.get_height()/2,
                f'{d:+.4f}', va='center', ha=ha, fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'anomaly_score_impact.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {SAVE_DIR}/anomaly_score_impact.png")


# ---------------------------------------------------------------------------
# 8. RESIDUAL-FRAUD ANALYSIS (Q2)
# ---------------------------------------------------------------------------

def plot_residual_fraud_analysis(anomaly_train, y_cls_train):
    """Fraud rate by residual decile for each regressor."""
    fig, axes = plt.subplots(1, len(anomaly_train), figsize=(6*len(anomaly_train), 5))
    if len(anomaly_train) == 1:
        axes = [axes]

    for ax, (name, residuals) in zip(axes, anomaly_train.items()):
        deciles = pd.qcut(residuals, q=10, labels=False, duplicates='drop')
        df_tmp = pd.DataFrame({'decile': deciles, 'fraud': y_cls_train})
        fraud_rate = df_tmp.groupby('decile')['fraud'].mean()
        ax.bar(fraud_rate.index, fraud_rate.values, color='#FF5722', edgecolor='white')
        ax.axhline(y=y_cls_train.mean(), color='black', linestyle='--',
                   label=f'Overall fraud rate: {y_cls_train.mean():.4f}')
        ax.set_xlabel('Residual Decile (0=lowest, 9=highest)')
        ax.set_ylabel('Fraud Rate')
        ax.set_title(f'{name} — Fraud Rate by Residual Decile')
        ax.legend()

    plt.suptitle('Research Question 2: Do High Residuals Correlate with Fraud?',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'residual_fraud_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {SAVE_DIR}/residual_fraud_analysis.png")


# ---------------------------------------------------------------------------
# 9. SHAP ANALYSIS
# ---------------------------------------------------------------------------

def run_shap_analysis(model, X_test, feature_names, model_name='XGBoost',
                      max_display=20):
    """SHAP bee-swarm and bar importance plots."""
    try:
        import shap
    except ImportError:
        print("  SHAP not installed. Skipping. Install with: pip install shap")
        return

    print(f"\n  Running SHAP analysis for {model_name} ...")
    n_sample = min(1000, X_test.shape[0])
    idx = np.random.choice(X_test.shape[0], n_sample, replace=False)
    X_sample = X_test[idx]

    if model_name.lower() in ['xgboost', 'random forest', 'rf']:
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.KernelExplainer(
            model.predict_proba if hasattr(model, 'predict_proba') else model.predict,
            shap.sample(X_test, 100))

    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1]

    if feature_names is not None and len(feature_names) < X_sample.shape[1]:
        extra = X_sample.shape[1] - len(feature_names)
        feature_names = feature_names + [f'anomaly_score_{i}' for i in range(extra)]

    # Bee-swarm plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                      max_display=max_display, show=False)
    plt.title(f'SHAP Feature Importance — {model_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'shap_importance.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {SAVE_DIR}/shap_importance.png")

    # Bar plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                      max_display=max_display, plot_type='bar', show=False)
    plt.title(f'Mean |SHAP| Feature Importance — {model_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'shap_importance_bar.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {SAVE_DIR}/shap_importance_bar.png")


# ---------------------------------------------------------------------------
# 10. MASTER EVALUATION FUNCTION
# ---------------------------------------------------------------------------

def run_evaluation(reg_results, cls_results,
                   anomaly_train=None, y_cls_train=None,
                   best_model=None, X_test=None, y_test=None,
                   feature_names=None, predictions_dict=None):
    """
    Generate all evaluation plots for the final report.

    Parameters
    ----------
    reg_results      : list[dict] — Stage 1 regression metrics
    cls_results      : list[dict] — Stage 2 classification metrics
    anomaly_train    : dict {name: residuals} — from Stage 1
    y_cls_train      : np.array — train labels (for Q2)
    best_model       : trained model — for SHAP
    X_test           : np.array — test features
    y_test           : np.array — test labels
    feature_names    : list[str]
    predictions_dict : dict {model_name: y_proba} — for ROC/PR/CM
    """
    print("\n" + "=" * 70)
    print("EVALUATION & VISUALIZATION")
    print("=" * 70)

    if reg_results:
        plot_regression_comparison(reg_results)

    if cls_results:
        plot_classification_comparison(cls_results)
        plot_anomaly_score_impact(cls_results)

    if predictions_dict and y_test is not None:
        plot_roc_curves(y_test, predictions_dict)
        plot_pr_curves(y_test, predictions_dict)
        plot_confusion_matrices(y_test, predictions_dict)
        save_classification_reports(y_test, predictions_dict)

    if anomaly_train is not None and y_cls_train is not None:
        plot_residual_fraud_analysis(anomaly_train, y_cls_train)

    if best_model is not None and X_test is not None:
        model_name = 'XGBoost' if hasattr(best_model, 'get_booster') else 'Model'
        run_shap_analysis(best_model, X_test, feature_names, model_name)

    print(f"\n  All figures saved to '{SAVE_DIR}/' directory.")