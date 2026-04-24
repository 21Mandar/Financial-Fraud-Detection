"""
stage1_regression.py — Stage 1: Predict TransactionAmt with 3 regressors.

Regressors:
    1. Linear Regression with polynomial + RBF basis expansions
    2. Random Forest Regressor
    3. DNN Regressor (PyTorch)

For each regressor, computes absolute residuals as anomaly scores.
These anomaly scores become features for Stage 2 classifiers.

Usage:
    from stage1_regression import run_stage1
    anomaly_train, anomaly_test, reg_results = run_stage1(
        X_train, X_test, y_reg_train, y_reg_test, feature_names
    )
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import make_pipeline
import time
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# EVALUATION HELPER
# ---------------------------------------------------------------------------

def eval_regressor(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"  {name:40s} | MAE: {mae:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}")
    return {'model': name, 'MAE': mae, 'RMSE': rmse, 'R2': r2}


# ---------------------------------------------------------------------------
# 1. LINEAR REGRESSION WITH BASIS EXPANSIONS
# ---------------------------------------------------------------------------

def train_linear_regression(X_train, X_test, y_train, y_test):
    """Ridge regression with polynomial (degree=2) + RBF basis expansion."""
    print("\n--- Linear Regression (Poly + RBF Basis) ---")

    # Use a subset of features for basis expansion (memory efficiency)
    n_features_basis = min(50, X_train.shape[1])

    # Polynomial features (degree 2, interaction only to manage dimensionality)
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_train_poly = poly.fit_transform(X_train[:, :n_features_basis])
    X_test_poly = poly.transform(X_test[:, :n_features_basis])
    print(f"  Poly features: {X_train_poly.shape[1]}")

    # RBF approximation
    rbf = RBFSampler(gamma=0.1, n_components=100, random_state=42)
    X_train_rbf = rbf.fit_transform(X_train[:, :n_features_basis])
    X_test_rbf = rbf.transform(X_test[:, :n_features_basis])

    # Concatenate original + poly + RBF
    X_train_aug = np.hstack([X_train, X_train_poly, X_train_rbf])
    X_test_aug = np.hstack([X_test, X_test_poly, X_test_rbf])
    print(f"  Total augmented features: {X_train_aug.shape[1]}")

    # Ridge regression (regularized linear regression)
    model = Ridge(alpha=1.0)
    model.fit(X_train_aug, y_train)

    y_pred_train = model.predict(X_train_aug)
    y_pred_test = model.predict(X_test_aug)

    metrics = eval_regressor(y_test, y_pred_test, "Linear Reg (Poly+RBF)")

    residuals_train = np.abs(y_train - y_pred_train)
    residuals_test = np.abs(y_test - y_pred_test)

    return model, residuals_train, residuals_test, metrics


# ---------------------------------------------------------------------------
# 2. RANDOM FOREST REGRESSOR
# ---------------------------------------------------------------------------

def train_rf_regressor(X_train, X_test, y_train, y_test):
    """Random Forest Regressor with tuned hyperparameters."""
    print("\n--- Random Forest Regressor ---")

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        n_jobs=-1,
        random_state=42,
        verbose=0
    )

    t0 = time.time()
    model.fit(X_train, y_train)
    print(f"  Training time: {time.time()-t0:.1f}s")

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    metrics = eval_regressor(y_test, y_pred_test, "Random Forest Regressor")

    residuals_train = np.abs(y_train - y_pred_train)
    residuals_test = np.abs(y_test - y_pred_test)

    return model, residuals_train, residuals_test, metrics


# ---------------------------------------------------------------------------
# 3. DNN REGRESSOR (PyTorch)
# ---------------------------------------------------------------------------

class DNNRegressor(nn.Module):
    """Deep feedforward network for regression."""
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_dnn_regressor(X_train, X_test, y_train, y_test,
                        epochs=30, batch_size=1024, lr=1e-3):
    """Train DNN regressor with early stopping."""
    print("\n--- DNN Regressor (PyTorch) ---")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    # Convert to tensors
    X_tr = torch.FloatTensor(X_train).to(device)
    y_tr = torch.FloatTensor(y_train).to(device)
    X_te = torch.FloatTensor(X_test).to(device)
    y_te = torch.FloatTensor(y_test).to(device)

    train_ds = TensorDataset(X_tr, y_tr)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = DNNRegressor(input_dim=X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)

        epoch_loss /= len(train_ds)

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_te)
            val_loss = criterion(val_pred, y_te).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:3d} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f}")

        if patience_counter >= 7:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # Restore best model
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        y_pred_train = model(X_tr).cpu().numpy()
        y_pred_test = model(X_te).cpu().numpy()

    metrics = eval_regressor(y_test, y_pred_test, "DNN Regressor")

    residuals_train = np.abs(y_train - y_pred_train)
    residuals_test = np.abs(y_test - y_pred_test)

    return model, residuals_train, residuals_test, metrics


# ---------------------------------------------------------------------------
# 4. ANOMALY SCORE CORRELATION WITH FRAUD
# ---------------------------------------------------------------------------

def analyze_residuals_vs_fraud(anomaly_scores, y_fraud, model_name):
    """Check if high residuals correlate with fraud (Research Question 2)."""
    from scipy.stats import pointbiserialr

    corr, pval = pointbiserialr(y_fraud, anomaly_scores)
    print(f"\n  {model_name} — Residual-Fraud correlation: {corr:.4f} (p={pval:.2e})")

    # Fraud rate in top 10% residuals vs overall
    threshold = np.percentile(anomaly_scores, 90)
    high_residual_mask = anomaly_scores >= threshold
    fraud_rate_high = y_fraud[high_residual_mask].mean()
    fraud_rate_overall = y_fraud.mean()
    print(f"  Fraud rate in top-10% residuals: {fraud_rate_high:.4f} vs overall: {fraud_rate_overall:.4f}")
    print(f"  Lift: {fraud_rate_high / fraud_rate_overall:.2f}x")

    return {'model': model_name, 'correlation': corr, 'p_value': pval,
            'fraud_rate_top10pct': fraud_rate_high, 'lift': fraud_rate_high / fraud_rate_overall}


# ---------------------------------------------------------------------------
# 5. MAIN ENTRY POINT
# ---------------------------------------------------------------------------

def run_stage1(X_train, X_test, y_reg_train, y_reg_test,
               y_cls_train=None, feature_names=None):
    """
    Run all three regressors and return anomaly scores.

    Returns
    -------
    anomaly_train : dict  {model_name: residuals_array}
    anomaly_test  : dict  {model_name: residuals_array}
    results       : list[dict]  metrics for each regressor
    """
    print("=" * 70)
    print("STAGE 1: REGRESSION — Predicting Log(TransactionAmt)")
    print("=" * 70)

    results = []
    anomaly_train = {}
    anomaly_test = {}
    correlation_results = []

    # 1. Linear Regression (Poly + RBF)
    _, res_tr, res_te, m = train_linear_regression(
        X_train, X_test, y_reg_train, y_reg_test
    )
    results.append(m)
    anomaly_train['LinearReg'] = res_tr
    anomaly_test['LinearReg'] = res_te

    # 2. Random Forest Regressor
    _, res_tr, res_te, m = train_rf_regressor(
        X_train, X_test, y_reg_train, y_reg_test
    )
    results.append(m)
    anomaly_train['RF_Reg'] = res_tr
    anomaly_test['RF_Reg'] = res_te

    # 3. DNN Regressor
    _, res_tr, res_te, m = train_dnn_regressor(
        X_train, X_test, y_reg_train, y_reg_test
    )
    results.append(m)
    anomaly_train['DNN_Reg'] = res_tr
    anomaly_test['DNN_Reg'] = res_te

    # Print summary
    print("\n" + "=" * 70)
    print("STAGE 1 SUMMARY")
    print("=" * 70)
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))

    # Research Question 2: residual-fraud correlation
    if y_cls_train is not None:
        print("\n--- Research Question 2: Residual-Fraud Correlation ---")
        for name in anomaly_train:
            cr = analyze_residuals_vs_fraud(anomaly_train[name], y_cls_train, name)
            correlation_results.append(cr)
        if correlation_results:
            print("\n" + pd.DataFrame(correlation_results).to_string(index=False))

    return anomaly_train, anomaly_test, results


if __name__ == '__main__':
    from data_preprocessing import load_and_preprocess
    (X_train, X_test, y_cls_train, y_cls_test,
     y_reg_train, y_reg_test, feature_names, scaler) = load_and_preprocess()

    anomaly_train, anomaly_test, results = run_stage1(
        X_train, X_test, y_reg_train, y_reg_test,
        y_cls_train=y_cls_train, feature_names=feature_names
    )