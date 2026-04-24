"""
main.py — Full pipeline orchestrator for the Financial Fraud Detection project.

CS 6140 Machine Learning — Course Project
"Financial Fraud Detection: Comparing Classical, Deep, and Graph-Enhanced
 Machine Learning Approaches"

Pipeline:
    0. EDA               (run separately: python eda.py)
    1. Preprocessing      → load, engineer, clean, split, scale
    2. Cross-validation   → hyperparameter tuning with 5-fold stratified CV
    3. Stage 1 Regression → 3 regressors, compute anomaly scores
    4. Stage 2 Classification → 5 classifiers, with/without anomaly scores
    5. Evaluation         → ROC, PR, confusion matrices, SHAP, report plots
    6. Model saving       → save all trained models + experiment log

Usage:
    python main.py                         # Full pipeline (with CV)
    python main.py --data_dir path/to/data
    python main.py --skip_cv               # Skip cross-validation (faster)
    python main.py --skip_gnn              # Skip GNN (faster)
"""

import argparse
import time
import numpy as np
import pandas as pd
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import load_and_preprocess
from stage1_regression import run_stage1
from stage2_classification import (
    run_stage2, augment_with_anomaly,
    train_logistic_regression, train_rf_classifier, train_xgboost,
    train_dnn_classifier, train_gnn_classifier
)
from evaluation import run_evaluation
from model_utils import set_seed, save_model, log_experiment, print_saved_models

import warnings
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description='Financial Fraud Detection Pipeline')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing CSV data files')
    parser.add_argument('--skip_cv', action='store_true',
                        help='Skip cross-validation (faster)')
    parser.add_argument('--skip_gnn', action='store_true',
                        help='Skip GNN training (faster)')
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("FINANCIAL FRAUD DETECTION PIPELINE")
    print("Comparing Classical, Deep, and Graph-Enhanced ML Approaches")
    print("=" * 70)
    print()

    t_start = time.time()
    set_seed(42)

    # ==================================================================
    # STEP 1: PREPROCESSING
    # ==================================================================
    (X_train, X_test,
     y_cls_train, y_cls_test,
     y_reg_train, y_reg_test,
     feature_names, scaler) = load_and_preprocess(data_dir=args.data_dir)

    # ==================================================================
    # STEP 2: CROSS-VALIDATION & HYPERPARAMETER TUNING
    # ==================================================================
    best_params = None
    if not args.skip_cv:
        from cross_validation import run_cross_validation
        best_params, cv_summary, all_cv_results = run_cross_validation(
            X_train, y_cls_train, y_reg_train
        )
    else:
        print("\n[SKIPPED] Cross-validation (use --skip_cv to skip)")

    # ==================================================================
    # STEP 3: STAGE 1 — REGRESSION
    # ==================================================================
    anomaly_train, anomaly_test, reg_results = run_stage1(
        X_train, X_test, y_reg_train, y_reg_test,
        y_cls_train=y_cls_train,
        feature_names=feature_names
    )

    # ==================================================================
    # STEP 4: STAGE 2 — CLASSIFICATION (collect predictions for plots)
    # ==================================================================
    print("\n" + "=" * 70)
    print("STAGE 2: CLASSIFICATION — Fraud Detection")
    print("=" * 70)

    # We run classifiers manually here so we can collect predictions
    # for ROC/PR/confusion matrix plots.

    all_cls_results = []
    predictions_base = {}
    predictions_aug = {}
    trained_models = {}

    X_train_aug = augment_with_anomaly(X_train, anomaly_train)
    X_test_aug = augment_with_anomaly(X_test, anomaly_test)

    classifiers = [
        ("Logistic Regression", train_logistic_regression),
        ("Random Forest Classifier", train_rf_classifier),
        ("XGBoost", train_xgboost),
        ("DNN Classifier", train_dnn_classifier),
    ]
    if not args.skip_gnn:
        classifiers.append(("GNN (GraphSAGE)", train_gnn_classifier))

    # --- WITHOUT anomaly scores ---
    print("\n>>> Without Anomaly Scores <<<")
    for name, train_fn in classifiers:
        model, y_proba, metrics = train_fn(
            X_train, X_test, y_cls_train, y_cls_test, suffix=" (base)"
        )
        if metrics is not None:
            metrics['anomaly_scores'] = False
            all_cls_results.append(metrics)
        if y_proba is not None:
            predictions_base[name] = y_proba

    # --- WITH anomaly scores ---
    print(f"\n>>> With Anomaly Scores (features: {X_train.shape[1]} → {X_train_aug.shape[1]}) <<<")
    for name, train_fn in classifiers:
        model, y_proba, metrics = train_fn(
            X_train_aug, X_test_aug, y_cls_train, y_cls_test, suffix=" (+anomaly)"
        )
        if metrics is not None:
            metrics['anomaly_scores'] = True
            all_cls_results.append(metrics)
        if y_proba is not None:
            predictions_aug[name + " +anomaly"] = y_proba
            trained_models[name] = model

    # Print Stage 2 summary
    print("\n" + "=" * 70)
    print("STAGE 2 SUMMARY")
    print("=" * 70)
    df_cls = pd.DataFrame(all_cls_results)
    print(df_cls.to_string(index=False))

    # ==================================================================
    # STEP 5: SAVE MODELS
    # ==================================================================
    print("\n" + "=" * 70)
    print("SAVING MODELS")
    print("=" * 70)
    for name, model in trained_models.items():
        if model is not None:
            save_model(model, name)
            # Log to experiment CSV
            matching = [r for r in all_cls_results
                        if r['model'].startswith(name) and r.get('anomaly_scores')]
            if matching:
                log_experiment(name, 'classification',
                               params=best_params.get(name, {}) if best_params else {},
                               metrics=matching[0])
    print_saved_models()

    # ==================================================================
    # STEP 6: EVALUATION & VISUALIZATION
    # ==================================================================
    # Merge all predictions for overlay plots
    all_predictions = {}
    all_predictions.update(predictions_base)
    all_predictions.update(predictions_aug)

    # Find best tree-based model for SHAP
    best_shap_model = None
    for name in ['XGBoost', 'Random Forest Classifier']:
        if name in trained_models and trained_models[name] is not None:
            best_shap_model = trained_models[name]
            shap_model_name = name
            break

    run_evaluation(
        reg_results=reg_results,
        cls_results=all_cls_results,
        anomaly_train=anomaly_train,
        y_cls_train=y_cls_train,
        best_model=best_shap_model,
        X_test=X_test_aug if best_shap_model else None,
        y_test=y_cls_test,
        feature_names=feature_names,
        predictions_dict=all_predictions,
    )

    # ==================================================================
    # RESEARCH QUESTION ANSWERS
    # ==================================================================
    elapsed = time.time() - t_start

    print("\n" + "=" * 70)
    print("RESEARCH QUESTION ANSWERS")
    print("=" * 70)

    # Q1: Best regressor
    df_reg = pd.DataFrame(reg_results)
    best_reg = df_reg.loc[df_reg['R2'].idxmax()]
    print(f"\nQ1: Best regressor → {best_reg['model']} "
          f"(R²={best_reg['R2']:.4f}, MAE={best_reg['MAE']:.4f})")

    # Q2: Residual-fraud correlation
    print("\nQ2: Residual-fraud correlation → see figures/residual_fraud_analysis.png")

    # Q3: Anomaly score impact
    df_aug = df_cls[df_cls['anomaly_scores'] == True]
    df_base_only = df_cls[df_cls['anomaly_scores'] == False]
    if len(df_aug) > 0 and len(df_base_only) > 0:
        mean_base = df_base_only['AUC_ROC'].mean()
        mean_aug = df_aug['AUC_ROC'].mean()
        print(f"\nQ3: Anomaly scores impact → mean AUC-ROC: "
              f"{mean_base:.4f} (base) → {mean_aug:.4f} (+anomaly) "
              f"[Δ={mean_aug - mean_base:+.4f}]")

    # Q4: GNN vs tabular
    gnn_rows = df_cls[df_cls['model'].str.contains('GNN', case=False)]
    if len(gnn_rows) > 0:
        best_gnn = gnn_rows.loc[gnn_rows['AUC_ROC'].idxmax()]
        best_tabular = df_cls[
            (~df_cls['model'].str.contains('GNN', case=False)) &
            (df_cls['anomaly_scores'] == best_gnn['anomaly_scores'])
        ]['AUC_ROC'].max()
        print(f"\nQ4: GNN AUC-ROC = {best_gnn['AUC_ROC']:.4f} vs "
              f"best tabular = {best_tabular:.4f}")
    else:
        print("\nQ4: GNN not trained — install torch_geometric to enable.")

    # CV summary
    if best_params and not args.skip_cv:
        print("\n" + "─" * 50)
        print("BEST HYPERPARAMETERS (from 5-fold CV)")
        print("─" * 50)
        for name, params in best_params.items():
            print(f"  {name:25s} → {params}")

    print(f"\nTotal runtime: {elapsed/60:.1f} minutes")
    print(f"Figures saved to: figures/")
    print(f"Models saved to: saved_models/")
    print("=" * 70)


if __name__ == '__main__':
    main()
