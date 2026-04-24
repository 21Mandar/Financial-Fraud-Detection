"""
model_utils.py — Model saving, loading, and experiment tracking utilities.

Provides:
    - Save/load for sklearn models (joblib)
    - Save/load for PyTorch models (state_dict)
    - Experiment result logging to CSV
    - Reproducibility seeding

Usage:
    from model_utils import save_model, load_model, log_experiment, set_seed
"""

import os
import json
import time
import numpy as np
import pandas as pd
import torch
import random
import warnings
warnings.filterwarnings("ignore")

MODEL_DIR = 'saved_models'
LOG_FILE = 'experiment_log.csv'

os.makedirs(MODEL_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. REPRODUCIBILITY
# ---------------------------------------------------------------------------

def set_seed(seed=42):
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"  Random seed set to {seed} for reproducibility.")


# ---------------------------------------------------------------------------
# 2. MODEL SAVING
# ---------------------------------------------------------------------------

def save_model(model, model_name, params=None, metrics=None):
    """
    Save a trained model to disk.

    Parameters
    ----------
    model       : sklearn model or PyTorch nn.Module
    model_name  : str — identifier used for filename
    params      : dict — hyperparameters (saved as JSON alongside)
    metrics     : dict — evaluation metrics (saved as JSON alongside)
    """
    safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '')
    base_path = os.path.join(MODEL_DIR, safe_name)

    if isinstance(model, torch.nn.Module):
        # PyTorch model
        path = base_path + '.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_class': model.__class__.__name__,
        }, path)
        print(f"  Saved PyTorch model: {path}")
    else:
        # Sklearn / XGBoost model
        try:
            import joblib
            path = base_path + '.joblib'
            joblib.dump(model, path)
            print(f"  Saved sklearn model: {path}")
        except ImportError:
            import pickle
            path = base_path + '.pkl'
            with open(path, 'wb') as f:
                pickle.dump(model, f)
            print(f"  Saved model (pickle): {path}")

    # Save hyperparameters
    if params is not None:
        params_path = base_path + '_params.json'
        # Convert non-serializable types
        serializable = {}
        for k, v in params.items():
            if isinstance(v, (list, tuple)):
                serializable[k] = str(v)
            elif isinstance(v, np.integer):
                serializable[k] = int(v)
            elif isinstance(v, np.floating):
                serializable[k] = float(v)
            else:
                serializable[k] = v
        with open(params_path, 'w') as f:
            json.dump(serializable, f, indent=2)

    # Save metrics
    if metrics is not None:
        metrics_path = base_path + '_metrics.json'
        serializable_m = {}
        for k, v in metrics.items():
            if isinstance(v, (np.floating, np.integer)):
                serializable_m[k] = float(v)
            else:
                serializable_m[k] = v
        with open(metrics_path, 'w') as f:
            json.dump(serializable_m, f, indent=2)


def load_sklearn_model(model_name):
    """Load a saved sklearn model."""
    safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '')
    joblib_path = os.path.join(MODEL_DIR, safe_name + '.joblib')
    pkl_path = os.path.join(MODEL_DIR, safe_name + '.pkl')

    if os.path.exists(joblib_path):
        import joblib
        model = joblib.load(joblib_path)
        print(f"  Loaded: {joblib_path}")
        return model
    elif os.path.exists(pkl_path):
        import pickle
        with open(pkl_path, 'rb') as f:
            model = pickle.load(f)
        print(f"  Loaded: {pkl_path}")
        return model
    else:
        raise FileNotFoundError(f"No saved model found for '{model_name}'")


def load_pytorch_model(model_class, model_name, **model_kwargs):
    """
    Load a saved PyTorch model.

    Parameters
    ----------
    model_class  : the nn.Module subclass (e.g. DNNClassifier)
    model_name   : str — must match what was used in save_model
    **model_kwargs : keyword arguments to instantiate model_class
    """
    safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '')
    path = os.path.join(MODEL_DIR, safe_name + '.pt')

    if not os.path.exists(path):
        raise FileNotFoundError(f"No saved model found at '{path}'")

    checkpoint = torch.load(path, map_location='cpu')
    model = model_class(**model_kwargs)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"  Loaded PyTorch model: {path}")
    return model


# ---------------------------------------------------------------------------
# 3. EXPERIMENT LOGGING
# ---------------------------------------------------------------------------

def log_experiment(model_name, stage, params, metrics, notes=''):
    """
    Append one row to the experiment log CSV.

    Parameters
    ----------
    model_name : str
    stage      : str — 'regression', 'classification', or 'cv'
    params     : dict
    metrics    : dict
    notes      : str
    """
    row = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'model': model_name,
        'stage': stage,
        'params': str(params),
        'notes': notes,
    }
    row.update(metrics)

    log_path = os.path.join(MODEL_DIR, LOG_FILE)
    df_new = pd.DataFrame([row])

    if os.path.exists(log_path):
        df_existing = pd.read_csv(log_path)
        df_all = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_all = df_new

    df_all.to_csv(log_path, index=False)
    print(f"  Logged experiment: {model_name} ({stage})")


def get_experiment_log():
    """Read the experiment log as a DataFrame."""
    log_path = os.path.join(MODEL_DIR, LOG_FILE)
    if os.path.exists(log_path):
        return pd.read_csv(log_path)
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# 4. QUICK SUMMARY
# ---------------------------------------------------------------------------

def print_saved_models():
    """List all saved models on disk."""
    print(f"\nSaved models in '{MODEL_DIR}/':")
    for f in sorted(os.listdir(MODEL_DIR)):
        if f.endswith(('.pt', '.joblib', '.pkl')):
            size = os.path.getsize(os.path.join(MODEL_DIR, f))
            print(f"  {f:45s}  ({size / 1024:.0f} KB)")


if __name__ == '__main__':
    set_seed(42)
    print_saved_models()