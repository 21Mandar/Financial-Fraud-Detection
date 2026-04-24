"""
cross_validation.py — 5-Fold Stratified Cross-Validation & Hyperparameter Tuning.

Satisfies course requirement: "Investigate effect of hyper-parameters (cross-validation)"

For each model, performs:
    1. Grid search over a defined hyperparameter grid
    2. 5-fold stratified cross-validation for each hyperparameter combination
    3. Reports mean ± std for all metrics across folds
    4. Selects best hyperparameters based on AUC-ROC
    5. Generates hyperparameter sensitivity plots

Usage:
    from cross_validation import run_cross_validation
    best_params, cv_results = run_cross_validation(X_train, y_cls_train, y_reg_train)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,
    precision_recall_curve, auc
)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
import time
import warnings
warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    _test = xgb.XGBClassifier(n_estimators=1, verbosity=0)
    HAS_XGB = True
except Exception:
    HAS_XGB = False
    print("WARNING: xgboost not available. XGBoost classifier will be skipped.")

SAVE_DIR = 'figures'
os.makedirs(SAVE_DIR, exist_ok=True)

N_FOLDS = 5
RANDOM_STATE = 42


# ===========================================================================
# HYPERPARAMETER GRIDS
# ===========================================================================

REGRESSION_GRIDS = {
    'LinearReg (Poly+RBF)': {
        'alpha': [0.01, 0.1, 1.0, 10.0],
        'poly_degree': [2],                     # fixed (memory)
        'rbf_gamma': [0.01, 0.1, 1.0],
        'rbf_n_components': [50, 100, 200],
    },
    'RF Regressor': {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20],
        'min_samples_leaf': [3, 5, 10],
    },
    'DNN Regressor': {
        'hidden_dims': [[256, 128, 64], [512, 256, 128], [128, 64]],
        'lr': [1e-3, 5e-4, 1e-4],
        'dropout': [0.2, 0.3, 0.4],
    },
}

CLASSIFICATION_GRIDS = {
    'Logistic Regression': {
        'C': [0.01, 0.1, 1.0, 10.0],
        'solver': ['lbfgs', 'saga'],
    },
    'RF Classifier': {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20],
        'min_samples_leaf': [3, 5, 10],
    },
    'XGBoost': {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1],
    },
    'DNN Classifier': {
        'hidden_dims': [[256, 128, 64], [512, 256, 128], [128, 64]],
        'lr': [1e-3, 5e-4, 1e-4],
        'dropout': [0.2, 0.3, 0.4],
    },
}


# ===========================================================================
# HELPER: DNN TRAINING (reusable for both regressor & classifier)
# ===========================================================================

class _DNNModel(nn.Module):
    """Configurable DNN for regression or classification."""
    def __init__(self, input_dim, hidden_dims, dropout, task='classification'):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h),
                           nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        self.task = task

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _train_dnn_fold(X_tr, y_tr, X_val, y_val, hidden_dims, lr, dropout,
                    task='classification', epochs=20, batch_size=1024):
    """Train a DNN on one fold, return predictions on validation set."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X_tr_t = torch.FloatTensor(X_tr).to(device)
    y_tr_t = torch.FloatTensor(y_tr).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)

    ds = TensorDataset(X_tr_t, y_tr_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = _DNNModel(X_tr.shape[1], hidden_dims, dropout, task).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    if task == 'classification':
        n_pos = y_tr.sum()
        n_neg = len(y_tr) - n_pos
        pos_w = torch.FloatTensor([n_neg / max(n_pos, 1)]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    else:
        criterion = nn.MSELoss()

    best_state = None
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, torch.FloatTensor(y_val).to(device)).item()
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds = model(X_val_t).cpu().numpy()

    if task == 'classification':
        preds = 1 / (1 + np.exp(-preds))   # sigmoid

    return preds


# ===========================================================================
# REGRESSION CROSS-VALIDATION
# ===========================================================================

def _cv_linear_reg(X, y, params, skf):
    """Cross-validate Linear Regression with basis expansions."""
    scores = {'MAE': [], 'RMSE': [], 'R2': []}
    n_feat = min(50, X.shape[1])

    for train_idx, val_idx in skf.split(X, (y > np.median(y)).astype(int)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # Poly
        poly = PolynomialFeatures(degree=params['poly_degree'],
                                  interaction_only=True, include_bias=False)
        X_tr_p = poly.fit_transform(X_tr[:, :n_feat])
        X_val_p = poly.transform(X_val[:, :n_feat])

        # RBF
        rbf = RBFSampler(gamma=params['rbf_gamma'],
                         n_components=params['rbf_n_components'],
                         random_state=RANDOM_STATE)
        X_tr_r = rbf.fit_transform(X_tr[:, :n_feat])
        X_val_r = rbf.transform(X_val[:, :n_feat])

        X_tr_aug = np.hstack([X_tr, X_tr_p, X_tr_r])
        X_val_aug = np.hstack([X_val, X_val_p, X_val_r])

        model = Ridge(alpha=params['alpha'])
        model.fit(X_tr_aug, y_tr)
        pred = model.predict(X_val_aug)

        scores['MAE'].append(mean_absolute_error(y_val, pred))
        scores['RMSE'].append(np.sqrt(mean_squared_error(y_val, pred)))
        scores['R2'].append(r2_score(y_val, pred))

    return {k: (np.mean(v), np.std(v)) for k, v in scores.items()}


def _cv_rf_regressor(X, y, params, skf):
    """Cross-validate Random Forest Regressor."""
    scores = {'MAE': [], 'RMSE': [], 'R2': []}

    for train_idx, val_idx in skf.split(X, (y > np.median(y)).astype(int)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = RandomForestRegressor(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            min_samples_leaf=params['min_samples_leaf'],
            max_features='sqrt', n_jobs=-1, random_state=RANDOM_STATE
        )
        model.fit(X_tr, y_tr)
        pred = model.predict(X_val)

        scores['MAE'].append(mean_absolute_error(y_val, pred))
        scores['RMSE'].append(np.sqrt(mean_squared_error(y_val, pred)))
        scores['R2'].append(r2_score(y_val, pred))

    return {k: (np.mean(v), np.std(v)) for k, v in scores.items()}


def _cv_dnn_regressor(X, y, params, skf):
    """Cross-validate DNN Regressor."""
    scores = {'MAE': [], 'RMSE': [], 'R2': []}

    for train_idx, val_idx in skf.split(X, (y > np.median(y)).astype(int)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        pred = _train_dnn_fold(X_tr, y_tr, X_val, y_val,
                               hidden_dims=params['hidden_dims'],
                               lr=params['lr'], dropout=params['dropout'],
                               task='regression', epochs=15)

        scores['MAE'].append(mean_absolute_error(y_val, pred))
        scores['RMSE'].append(np.sqrt(mean_squared_error(y_val, pred)))
        scores['R2'].append(r2_score(y_val, pred))

    return {k: (np.mean(v), np.std(v)) for k, v in scores.items()}


# ===========================================================================
# CLASSIFICATION CROSS-VALIDATION
# ===========================================================================

def _eval_cls_fold(y_true, y_proba):
    """Compute classification metrics for one fold."""
    y_pred = (y_proba >= 0.5).astype(int)
    auc_roc = roc_auc_score(y_true, y_proba)
    prec_c, rec_c, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(rec_c, prec_c)
    return {
        'AUC_ROC': auc_roc,
        'PR_AUC': pr_auc,
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1': f1_score(y_true, y_pred, zero_division=0),
    }


def _cv_logistic(X, y, params, skf):
    """Cross-validate Logistic Regression."""
    fold_metrics = []
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = LogisticRegression(
            C=params['C'], solver=params['solver'],
            class_weight='balanced', max_iter=1000,
            n_jobs=-1, random_state=RANDOM_STATE
        )
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_val)[:, 1]
        fold_metrics.append(_eval_cls_fold(y_val, proba))

    return {k: (np.mean([m[k] for m in fold_metrics]),
                np.std([m[k] for m in fold_metrics]))
            for k in fold_metrics[0]}


def _cv_rf_classifier(X, y, params, skf):
    """Cross-validate Random Forest Classifier."""
    fold_metrics = []
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = RandomForestClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            min_samples_leaf=params['min_samples_leaf'],
            max_features='sqrt', class_weight='balanced_subsample',
            n_jobs=-1, random_state=RANDOM_STATE
        )
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_val)[:, 1]
        fold_metrics.append(_eval_cls_fold(y_val, proba))

    return {k: (np.mean([m[k] for m in fold_metrics]),
                np.std([m[k] for m in fold_metrics]))
            for k in fold_metrics[0]}


def _cv_xgboost(X, y, params, skf):
    """Cross-validate XGBoost."""
    if not HAS_XGB:
        return None
    fold_metrics = []
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        n_neg = (y_tr == 0).sum()
        n_pos = (y_tr == 1).sum()

        model = xgb.XGBClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=n_neg / max(n_pos, 1),
            eval_metric='auc', use_label_encoder=False,
            tree_method='hist', n_jobs=1,
            random_state=RANDOM_STATE, verbosity=0
        )
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        proba = model.predict_proba(X_val)[:, 1]
        fold_metrics.append(_eval_cls_fold(y_val, proba))

    return {k: (np.mean([m[k] for m in fold_metrics]),
                np.std([m[k] for m in fold_metrics]))
            for k in fold_metrics[0]}


def _cv_dnn_classifier(X, y, params, skf):
    """Cross-validate DNN Classifier."""
    fold_metrics = []
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        proba = _train_dnn_fold(X_tr, y_tr, X_val, y_val,
                                hidden_dims=params['hidden_dims'],
                                lr=params['lr'], dropout=params['dropout'],
                                task='classification', epochs=15)
        fold_metrics.append(_eval_cls_fold(y_val, proba))

    return {k: (np.mean([m[k] for m in fold_metrics]),
                np.std([m[k] for m in fold_metrics]))
            for k in fold_metrics[0]}


# ===========================================================================
# GRID SEARCH OVER HYPERPARAMETERS
# ===========================================================================

def _generate_param_combos(grid):
    """Generate all combinations from a hyperparameter grid."""
    from itertools import product
    keys = list(grid.keys())
    values = list(grid.values())
    for combo in product(*values):
        yield dict(zip(keys, combo))


def grid_search_cv(X, y, model_name, grid, cv_fn, skf, metric_key='R2'):
    """
    Run grid search with cross-validation for one model.

    Returns
    -------
    best_params : dict
    all_results : list[dict]  — params + mean/std for each metric
    """
    print(f"\n{'='*60}")
    print(f"  Grid Search: {model_name}")
    print(f"{'='*60}")

    combos = list(_generate_param_combos(grid))
    print(f"  Testing {len(combos)} hyperparameter combinations × {N_FOLDS} folds")

    all_results = []
    best_score = -float('inf')
    best_params = None

    for i, params in enumerate(combos):
        t0 = time.time()
        scores = cv_fn(X, y, params, skf)
        if scores is None:
            continue
        elapsed = time.time() - t0

        result = {'params': str(params)}
        for k, (mean, std) in scores.items():
            result[f'{k}_mean'] = mean
            result[f'{k}_std'] = std
        result['time'] = elapsed
        all_results.append(result)

        score_val = scores[metric_key][0]
        marker = " ★" if score_val > best_score else ""
        if score_val > best_score:
            best_score = score_val
            best_params = params

        if (i + 1) % 3 == 0 or (i + 1) == len(combos):
            print(f"  [{i+1}/{len(combos)}] Best {metric_key}: {best_score:.4f} | "
                  f"Current: {score_val:.4f} ({elapsed:.1f}s){marker}")

    print(f"\n  Best params: {best_params}")
    print(f"  Best {metric_key}: {best_score:.4f}")

    return best_params, all_results


# ===========================================================================
# HYPERPARAMETER SENSITIVITY PLOTS
# ===========================================================================

def plot_hyperparam_sensitivity(all_cv_results, model_name, param_name,
                                metric='AUC_ROC'):
    """Plot how one hyperparameter affects performance (line plot w/ error bars)."""
    df = pd.DataFrame(all_cv_results)

    # Extract the specific param value from the params string
    df['param_val'] = df['params'].apply(
        lambda s: eval(s).get(param_name, None)
    )
    df = df.dropna(subset=['param_val'])

    # Average over other hyperparams for each value of this param
    grouped = df.groupby('param_val').agg({
        f'{metric}_mean': 'mean',
        f'{metric}_std': 'mean'
    }).reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    x_vals = range(len(grouped))
    x_labels = [str(v) for v in grouped['param_val']]

    ax.errorbar(x_vals, grouped[f'{metric}_mean'],
                yerr=grouped[f'{metric}_std'],
                marker='o', capsize=5, linewidth=2, markersize=8,
                color='#2196F3')

    ax.set_xticks(x_vals)
    ax.set_xticklabels(x_labels, rotation=30, ha='right')
    ax.set_xlabel(param_name)
    ax.set_ylabel(f'{metric} (mean ± std)')
    ax.set_title(f'{model_name}: Effect of {param_name} on {metric}')
    ax.grid(True, alpha=0.3)

    fname = f'hp_sensitivity_{model_name.replace(" ", "_")}_{param_name}.png'
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {SAVE_DIR}/{fname}")


def plot_cv_fold_comparison(fold_results_dict):
    """Bar chart showing mean ± std across folds for each model."""
    models = []
    means = []
    stds = []

    for name, (mean, std) in fold_results_dict.items():
        models.append(name)
        means.append(mean)
        stds.append(std)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(models))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color='#2196F3',
                  edgecolor='white', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha='right')
    ax.set_ylabel('AUC-ROC (mean ± std over 5 folds)')
    ax.set_title('5-Fold Cross-Validation: Model Comparison')
    ax.grid(True, axis='y', alpha=0.3)

    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 0.003,
                f'{m:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'cv_model_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {SAVE_DIR}/cv_model_comparison.png")


# ===========================================================================
# MAIN ENTRY POINT
# ===========================================================================

def run_cross_validation(X_train, y_cls_train, y_reg_train):
    """
    Run full cross-validation & hyperparameter tuning for all models.

    Returns
    -------
    best_params       : dict  {model_name: best_param_dict}
    cv_summary        : dict  {model_name: (mean_score, std_score)}
    all_cv_results    : dict  {model_name: list[dict]}
    """
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION & HYPERPARAMETER TUNING")
    print(f"  {N_FOLDS}-Fold Stratified CV")
    print("=" * 70)

    skf_cls = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                               random_state=RANDOM_STATE)
    # For regression, we create pseudo-strata based on median split
    skf_reg = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                               random_state=RANDOM_STATE)

    best_params = {}
    cv_summary = {}        # model_name -> (mean_auc, std_auc)
    all_cv_results = {}

    # ------------------------------------------------------------------
    # REGRESSION CV
    # ------------------------------------------------------------------
    print("\n" + "─" * 50)
    print("  REGRESSION MODELS")
    print("─" * 50)

    # Linear Reg
    bp, res = grid_search_cv(
        X_train, y_reg_train, 'LinearReg (Poly+RBF)',
        REGRESSION_GRIDS['LinearReg (Poly+RBF)'],
        _cv_linear_reg, skf_reg, metric_key='R2'
    )
    best_params['LinearReg'] = bp
    all_cv_results['LinearReg'] = res
    best_r2 = max(r['R2_mean'] for r in res)
    best_std = [r['R2_std'] for r in res if r['R2_mean'] == best_r2][0]
    cv_summary['LinearReg (R²)'] = (best_r2, best_std)

    # RF Regressor
    bp, res = grid_search_cv(
        X_train, y_reg_train, 'RF Regressor',
        REGRESSION_GRIDS['RF Regressor'],
        _cv_rf_regressor, skf_reg, metric_key='R2'
    )
    best_params['RF_Reg'] = bp
    all_cv_results['RF_Reg'] = res
    best_r2 = max(r['R2_mean'] for r in res)
    best_std = [r['R2_std'] for r in res if r['R2_mean'] == best_r2][0]
    cv_summary['RF Reg (R²)'] = (best_r2, best_std)

    # DNN Regressor
    bp, res = grid_search_cv(
        X_train, y_reg_train, 'DNN Regressor',
        REGRESSION_GRIDS['DNN Regressor'],
        _cv_dnn_regressor, skf_reg, metric_key='R2'
    )
    best_params['DNN_Reg'] = bp
    all_cv_results['DNN_Reg'] = res
    best_r2 = max(r['R2_mean'] for r in res)
    best_std = [r['R2_std'] for r in res if r['R2_mean'] == best_r2][0]
    cv_summary['DNN Reg (R²)'] = (best_r2, best_std)

    # ------------------------------------------------------------------
    # CLASSIFICATION CV
    # ------------------------------------------------------------------
    print("\n" + "─" * 50)
    print("  CLASSIFICATION MODELS")
    print("─" * 50)

    cls_models = [
        ('Logistic Regression', CLASSIFICATION_GRIDS['Logistic Regression'],
         _cv_logistic, 'LogReg'),
        ('RF Classifier', CLASSIFICATION_GRIDS['RF Classifier'],
         _cv_rf_classifier, 'RF_Cls'),
        ('XGBoost', CLASSIFICATION_GRIDS['XGBoost'],
         _cv_xgboost, 'XGBoost'),
        ('DNN Classifier', CLASSIFICATION_GRIDS['DNN Classifier'],
         _cv_dnn_classifier, 'DNN_Cls'),
    ]

    for display_name, grid, cv_fn, key in cls_models:
        bp, res = grid_search_cv(
            X_train, y_cls_train, display_name, grid,
            cv_fn, skf_cls, metric_key='AUC_ROC'
        )
        if bp is not None:
            best_params[key] = bp
            all_cv_results[key] = res
            best_auc = max(r['AUC_ROC_mean'] for r in res)
            best_std = [r['AUC_ROC_std'] for r in res
                        if r['AUC_ROC_mean'] == best_auc][0]
            cv_summary[display_name] = (best_auc, best_std)

    # ------------------------------------------------------------------
    # GENERATE PLOTS
    # ------------------------------------------------------------------
    print("\n" + "─" * 50)
    print("  GENERATING HYPERPARAMETER SENSITIVITY PLOTS")
    print("─" * 50)

    # Key hyperparameter sensitivity plots
    if 'LogReg' in all_cv_results:
        plot_hyperparam_sensitivity(all_cv_results['LogReg'],
                                    'Logistic_Regression', 'C', 'AUC_ROC')
    if 'RF_Cls' in all_cv_results:
        plot_hyperparam_sensitivity(all_cv_results['RF_Cls'],
                                    'RF_Classifier', 'max_depth', 'AUC_ROC')
        plot_hyperparam_sensitivity(all_cv_results['RF_Cls'],
                                    'RF_Classifier', 'n_estimators', 'AUC_ROC')
    if 'XGBoost' in all_cv_results:
        plot_hyperparam_sensitivity(all_cv_results['XGBoost'],
                                    'XGBoost', 'learning_rate', 'AUC_ROC')
        plot_hyperparam_sensitivity(all_cv_results['XGBoost'],
                                    'XGBoost', 'max_depth', 'AUC_ROC')
    if 'DNN_Cls' in all_cv_results:
        plot_hyperparam_sensitivity(all_cv_results['DNN_Cls'],
                                    'DNN_Classifier', 'lr', 'AUC_ROC')

    # Regression sensitivity
    if 'LinearReg' in all_cv_results:
        plot_hyperparam_sensitivity(all_cv_results['LinearReg'],
                                    'LinearReg', 'alpha', 'R2')
    if 'RF_Reg' in all_cv_results:
        plot_hyperparam_sensitivity(all_cv_results['RF_Reg'],
                                    'RF_Regressor', 'max_depth', 'R2')

    # Cross-model fold comparison (classification only)
    cls_cv_summary = {k: v for k, v in cv_summary.items() if '(R²)' not in k}
    if cls_cv_summary:
        plot_cv_fold_comparison(cls_cv_summary)

    # ------------------------------------------------------------------
    # SUMMARY TABLE
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 70)

    rows = []
    for name, (mean, std) in cv_summary.items():
        rows.append({'Model': name, 'Score (mean)': mean, 'Score (std)': std,
                     'Score': f'{mean:.4f} ± {std:.4f}'})
    df_summary = pd.DataFrame(rows)
    print(df_summary.to_string(index=False))

    print("\nBest Hyperparameters:")
    for name, params in best_params.items():
        print(f"  {name:25s} → {params}")

    return best_params, cv_summary, all_cv_results


if __name__ == '__main__':
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    from data_preprocessing import load_and_preprocess
    (X_train, X_test, y_cls_train, y_cls_test,
     y_reg_train, y_reg_test, feature_names, scaler) = load_and_preprocess()

    best_params, cv_summary, all_cv_results = run_cross_validation(
        X_train, y_cls_train, y_reg_train
    )
