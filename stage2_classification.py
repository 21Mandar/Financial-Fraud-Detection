"""
stage2_classification.py — Stage 2: Fraud detection classifiers.

Classifiers:
    1. Logistic Regression
    2. Random Forest Classifier
    3. XGBoost
    4. DNN Classifier (PyTorch)
    5. GNN (PyTorch Geometric)

Each classifier is trained WITH and WITHOUT Stage 1 anomaly scores
to answer Research Question 3.

Usage:
    from stage2_classification import run_stage2
    results = run_stage2(X_train, X_test, y_cls_train, y_cls_test,
                         anomaly_train, anomaly_test, feature_names)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    classification_report, precision_recall_curve, auc
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import time
import warnings
warnings.filterwarnings("ignore")

# Try importing xgboost and torch_geometric
try:
    import xgboost as xgb
    _test = xgb.XGBClassifier(n_estimators=1, verbosity=0)
    HAS_XGB = True
except Exception:
    HAS_XGB = False
    print("WARNING: xgboost not available. XGBoost classifier will be skipped.")

try:
    import torch_geometric
    from torch_geometric.data import Data
    from torch_geometric.nn import SAGEConv, global_mean_pool
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print("WARNING: torch_geometric not installed. GNN classifier will be skipped.")


# ---------------------------------------------------------------------------
# EVALUATION HELPER
# ---------------------------------------------------------------------------

def eval_classifier(y_true, y_pred_proba, name):
    """Evaluate classifier with AUC-ROC, PR-AUC, Precision, Recall, F1."""
    y_pred = (y_pred_proba >= 0.5).astype(int)

    auc_roc = roc_auc_score(y_true, y_pred_proba)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # PR-AUC (more informative for imbalanced datasets)
    prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(rec_curve, prec_curve)

    print(f"  {name:45s} | AUC-ROC: {auc_roc:.4f} | PR-AUC: {pr_auc:.4f} | "
          f"P: {precision:.4f} | R: {recall:.4f} | F1: {f1:.4f}")

    return {
        'model': name, 'AUC_ROC': auc_roc, 'PR_AUC': pr_auc,
        'Precision': precision, 'Recall': recall, 'F1': f1
    }


# ---------------------------------------------------------------------------
# SMOTE RESAMPLING
# ---------------------------------------------------------------------------

def apply_smote(X_train, y_train):
    """Apply SMOTE to handle class imbalance."""
    sm = SMOTE(random_state=42, n_jobs=-1)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"  SMOTE: {X_train.shape[0]} → {X_res.shape[0]} samples "
          f"(fraud: {y_train.sum()} → {y_res.sum()})")
    return X_res, y_res


# ---------------------------------------------------------------------------
# 1. LOGISTIC REGRESSION
# ---------------------------------------------------------------------------

def train_logistic_regression(X_train, X_test, y_train, y_test, suffix=""):
    """Logistic Regression with class weights."""
    name = f"Logistic Regression{suffix}"
    print(f"\n--- {name} ---")

    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        C=0.1,
        solver='lbfgs',
        n_jobs=-1,
        random_state=42
    )

    t0 = time.time()
    model.fit(X_train, y_train)
    print(f"  Training time: {time.time()-t0:.1f}s")

    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = eval_classifier(y_test, y_proba, name)

    return model, y_proba, metrics


# ---------------------------------------------------------------------------
# 2. RANDOM FOREST CLASSIFIER
# ---------------------------------------------------------------------------

def train_rf_classifier(X_train, X_test, y_train, y_test, suffix=""):
    """Random Forest Classifier with balanced class weights."""
    name = f"Random Forest Classifier{suffix}"
    print(f"\n--- {name} ---")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        class_weight='balanced_subsample',
        n_jobs=-1,
        random_state=42,
        verbose=0
    )

    t0 = time.time()
    model.fit(X_train, y_train)
    print(f"  Training time: {time.time()-t0:.1f}s")

    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = eval_classifier(y_test, y_proba, name)

    return model, y_proba, metrics


# ---------------------------------------------------------------------------
# 3. XGBOOST
# ---------------------------------------------------------------------------

def train_xgboost(X_train, X_test, y_train, y_test, suffix=""):
    """XGBoost with scale_pos_weight for imbalance."""
    if not HAS_XGB:
        print("  Skipping XGBoost (not installed)")
        return None, None, None

    name = f"XGBoost{suffix}"
    print(f"\n--- {name} ---")

    # Calculate scale_pos_weight for imbalance
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric='auc',
        use_label_encoder=False,
        tree_method='hist',
        n_jobs=1,
        random_state=42,
        verbosity=0
    )

    t0 = time.time()
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)
    print(f"  Training time: {time.time()-t0:.1f}s")

    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = eval_classifier(y_test, y_proba, name)

    return model, y_proba, metrics


# ---------------------------------------------------------------------------
# 4. DNN CLASSIFIER (PyTorch)
# ---------------------------------------------------------------------------

class DNNClassifier(nn.Module):
    """Deep feedforward network for binary classification."""
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


def train_dnn_classifier(X_train, X_test, y_train, y_test,
                         suffix="", epochs=30, batch_size=1024, lr=1e-3):
    """Train DNN classifier with class-weighted BCE loss."""
    name = f"DNN Classifier{suffix}"
    print(f"\n--- {name} ---")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    # Class weights for loss
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    pos_weight = torch.FloatTensor([n_neg / n_pos]).to(device)

    # Convert to tensors
    X_tr = torch.FloatTensor(X_train).to(device)
    y_tr = torch.FloatTensor(y_train).to(device)
    X_te = torch.FloatTensor(X_test).to(device)

    train_ds = TensorDataset(X_tr, y_tr)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = DNNClassifier(input_dim=X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_auc = 0
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(train_ds)

        # Validation AUC
        model.eval()
        with torch.no_grad():
            val_logits = model(X_te)
            val_proba = torch.sigmoid(val_logits).cpu().numpy()
        val_auc = roc_auc_score(y_test, val_proba)

        scheduler.step(-val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:3d} | Loss: {epoch_loss:.4f} | Val AUC: {val_auc:.4f}")

        if patience_counter >= 7:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        y_proba = torch.sigmoid(model(X_te)).cpu().numpy()

    metrics = eval_classifier(y_test, y_proba, name)
    return model, y_proba, metrics


# ---------------------------------------------------------------------------
# 5. GNN (PyTorch Geometric)
# ---------------------------------------------------------------------------

class FraudGNN(nn.Module):
    """
    GraphSAGE-based GNN for fraud detection.
    Constructs a k-NN graph from feature similarity.
    """
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(SAGEConv(input_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = torch.relu(x)
            x = torch.dropout(x, p=self.dropout, train=self.training)
        return self.classifier(x).squeeze(-1)


def build_knn_graph(X, k=10, batch_size=5000):
    """Build k-NN graph from feature matrix using cosine similarity.
    Processes in batches for memory efficiency."""
    from sklearn.neighbors import NearestNeighbors

    print(f"  Building {k}-NN graph for {X.shape[0]} nodes ...")
    nn_model = NearestNeighbors(n_neighbors=k, metric='cosine',
                                 algorithm='brute', n_jobs=-1)
    nn_model.fit(X)
    _, indices = nn_model.kneighbors(X)

    # Convert to edge_index format
    rows = np.repeat(np.arange(X.shape[0]), k)
    cols = indices.flatten()
    edge_index = np.stack([rows, cols], axis=0)

    return edge_index


def train_gnn_classifier(X_train, X_test, y_train, y_test,
                         suffix="", epochs=30, lr=1e-3, k=10):
    """Train GNN classifier on k-NN graph."""
    if not HAS_PYG:
        print("  Skipping GNN (torch_geometric not installed)")
        return None, None, None

    name = f"GNN (GraphSAGE){suffix}"
    print(f"\n--- {name} ---")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    # Combine train + test for graph construction, then mask
    n_train = X_train.shape[0]
    n_total = n_train + X_test.shape[0]
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])

    train_mask = np.zeros(n_total, dtype=bool)
    train_mask[:n_train] = True
    test_mask = ~train_mask

    # Build graph (use subsample if dataset is very large)
    MAX_NODES_FOR_GRAPH = 100000
    if n_total > MAX_NODES_FOR_GRAPH:
        print(f"  Subsampling to {MAX_NODES_FOR_GRAPH} nodes for GNN ...")
        # Stratified subsample
        from sklearn.model_selection import train_test_split
        idx = np.arange(n_total)
        idx_sub, _ = train_test_split(
            idx, train_size=MAX_NODES_FOR_GRAPH,
            stratify=y_all, random_state=42
        )
        idx_sub = np.sort(idx_sub)
        X_all = X_all[idx_sub]
        y_all = y_all[idx_sub]
        # Recalculate masks
        train_mask_orig = train_mask[idx_sub]
        test_mask_orig = ~train_mask_orig
        train_mask = train_mask_orig
        test_mask = test_mask_orig
        n_total = len(idx_sub)
        print(f"  Subsampled: {n_total} nodes ({train_mask.sum()} train, {test_mask.sum()} test)")

    edge_index = build_knn_graph(X_all, k=k)

    # PyG Data object
    x = torch.FloatTensor(X_all).to(device)
    y = torch.FloatTensor(y_all).to(device)
    edge_idx = torch.LongTensor(edge_index).to(device)
    train_mask_t = torch.BoolTensor(train_mask).to(device)
    test_mask_t = torch.BoolTensor(test_mask).to(device)

    # Class weights
    n_pos = y_all[train_mask].sum()
    n_neg = train_mask.sum() - n_pos
    pos_weight = torch.FloatTensor([n_neg / max(n_pos, 1)]).to(device)

    model = FraudGNN(input_dim=X_all.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_auc = 0
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(x, edge_idx)
        loss = criterion(logits[train_mask_t], y[train_mask_t])
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            all_logits = model(x, edge_idx)
            test_proba = torch.sigmoid(all_logits[test_mask_t]).cpu().numpy()
        val_auc = roc_auc_score(y_all[test_mask], test_proba)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:3d} | Loss: {loss.item():.4f} | Val AUC: {val_auc:.4f}")

        if patience_counter >= 7:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        all_logits = model(x, edge_idx)
        y_proba = torch.sigmoid(all_logits[test_mask_t]).cpu().numpy()

    y_test_gnn = y_all[test_mask]
    metrics = eval_classifier(y_test_gnn, y_proba, name)

    return model, y_proba, metrics


# ---------------------------------------------------------------------------
# 6. AUGMENT FEATURES WITH ANOMALY SCORES
# ---------------------------------------------------------------------------

def augment_with_anomaly(X, anomaly_dict):
    """Append anomaly scores from all Stage 1 regressors as new features."""
    anomaly_cols = []
    for name, scores in anomaly_dict.items():
        anomaly_cols.append(scores.reshape(-1, 1))
    if anomaly_cols:
        return np.hstack([X] + anomaly_cols)
    return X


# ---------------------------------------------------------------------------
# 7. MAIN ENTRY POINT
# ---------------------------------------------------------------------------

def run_stage2(X_train, X_test, y_cls_train, y_cls_test,
               anomaly_train=None, anomaly_test=None, feature_names=None):
    """
    Run all classifiers with and without anomaly scores.

    Returns
    -------
    results : list[dict]  — metrics for all model variants
    """
    print("\n" + "=" * 70)
    print("STAGE 2: CLASSIFICATION — Fraud Detection")
    print("=" * 70)

    all_results = []
    classifiers = [
        ("LogReg", train_logistic_regression),
        ("RF", train_rf_classifier),
        ("XGBoost", train_xgboost),
        ("DNN", train_dnn_classifier),
        ("GNN", train_gnn_classifier),
    ]

    # ---- Run WITHOUT anomaly scores ----
    print("\n>>> Without Anomaly Scores <<<")
    for short_name, train_fn in classifiers:
        _, _, metrics = train_fn(X_train, X_test, y_cls_train, y_cls_test,
                                  suffix=" (base)")
        if metrics is not None:
            metrics['anomaly_scores'] = False
            all_results.append(metrics)

    # ---- Run WITH anomaly scores (Research Question 3) ----
    if anomaly_train is not None and anomaly_test is not None:
        X_train_aug = augment_with_anomaly(X_train, anomaly_train)
        X_test_aug = augment_with_anomaly(X_test, anomaly_test)
        print(f"\n>>> With Anomaly Scores (features: {X_train.shape[1]} → {X_train_aug.shape[1]}) <<<")

        for short_name, train_fn in classifiers:
            _, _, metrics = train_fn(X_train_aug, X_test_aug,
                                      y_cls_train, y_cls_test,
                                      suffix=" (+anomaly)")
            if metrics is not None:
                metrics['anomaly_scores'] = True
                all_results.append(metrics)

    # Summary
    print("\n" + "=" * 70)
    print("STAGE 2 SUMMARY")
    print("=" * 70)
    df_results = pd.DataFrame(all_results)
    print(df_results.to_string(index=False))

    return all_results


if __name__ == '__main__':
    from data_preprocessing import load_and_preprocess
    from stage1_regression import run_stage1

    (X_train, X_test, y_cls_train, y_cls_test,
     y_reg_train, y_reg_test, feature_names, scaler) = load_and_preprocess()

    anomaly_train, anomaly_test, _ = run_stage1(
        X_train, X_test, y_reg_train, y_reg_test,
        y_cls_train=y_cls_train, feature_names=feature_names
    )

    results = run_stage2(
        X_train, X_test, y_cls_train, y_cls_test,
        anomaly_train, anomaly_test, feature_names
    )