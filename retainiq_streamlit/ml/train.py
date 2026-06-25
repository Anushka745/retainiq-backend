"""
Churn prediction model training and inference.

Uses XGBoost if available, falls back to GradientBoostingClassifier
(same API, comparable accuracy) so the codebase works everywhere.
To switch to XGBoost on a production server: just `pip install xgboost`.
"""

import os
import sys
import json
import joblib
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
    _CLASSIFIER = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1.5,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    BACKEND = "xgboost"
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    _CLASSIFIER = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    BACKEND = "sklearn-gbm"

FEATURE_COLS = [
    "session_count",
    "avg_session_duration",
    "days_since_last_login",
    "support_tickets",
    "subscription_age",
    "feature_usage_score",
    "payment_failures",
]

MODEL_DIR = Path(__file__).parent.parent / "saved_models"
MODEL_PATH = MODEL_DIR / "churn_model.pkl"
METRICS_PATH = MODEL_DIR / "model_metrics.json"


def train(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Train churn model on the supplied DataFrame.
    Returns evaluation metrics dict and saves model to disk.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df["churn_label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", _CLASSIFIER),
        ]
    )

    print(f"[ML] Training with {BACKEND} on {len(X_train):,} samples…")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "backend": BACKEND,
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
    }

    joblib.dump(pipeline, MODEL_PATH)
    with open(METRICS_PATH, "w") as fh:
        json.dump(metrics, fh, indent=2)

    print(f"[ML] Model saved → {MODEL_PATH}")
    print(f"[ML] Metrics: {metrics}")
    return metrics


def load_model() -> Pipeline:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run `python ml/train.py` first."
        )
    return joblib.load(MODEL_PATH)


def get_metrics() -> Dict[str, Any]:
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as fh:
            return json.load(fh)
    return {}


def predict_single(features: Dict[str, float], model: Pipeline) -> Dict[str, Any]:
    """
    Predict churn for a single user.
    features must contain all FEATURE_COLS keys.
    Returns: churn_probability, risk_level, recommended_action
    """
    row = np.array(
        [[features[col] for col in FEATURE_COLS]], dtype=np.float32
    )
    prob = float(model.predict_proba(row)[0, 1])
    percent = round(prob * 100, 2)
    risk_level, action = _classify_risk(percent, features)
    return {
        "churn_probability": percent,
        "risk_level": risk_level,
        "recommended_action": action,
    }


def predict_batch(df: pd.DataFrame, model: Pipeline) -> pd.DataFrame:
    """Predict churn for all rows in df; expects FEATURE_COLS present."""
    X = df[FEATURE_COLS].values.astype(np.float32)
    probs = model.predict_proba(X)[:, 1]
    percents = (probs * 100).round(2)

    risk_levels = []
    actions = []
    for i, pct in enumerate(percents):
        row_features = {col: df[col].iloc[i] for col in FEATURE_COLS}
        rl, ac = _classify_risk(pct, row_features)
        risk_levels.append(rl)
        actions.append(ac)

    result = df[["user_id"]].copy()
    result["churn_probability"] = percents
    result["risk_level"] = risk_levels
    result["recommended_action"] = actions
    return result


def _classify_risk(prob_pct: float, features: Dict[str, float]) -> Tuple[str, str]:
    if prob_pct <= 40:
        level = "Low"
        action = _low_risk_action(features)
    elif prob_pct <= 70:
        level = "Medium"
        action = _medium_risk_action(features)
    else:
        level = "High"
        action = _high_risk_action(features)
    return level, action


def _low_risk_action(f: Dict[str, float]) -> str:
    if f.get("feature_usage_score", 100) < 60:
        return "Introduce advanced features via in-app walkthrough"
    if f.get("session_count", 30) < 10:
        return "Send engagement tips newsletter"
    return "Offer loyalty reward or referral bonus"


def _medium_risk_action(f: Dict[str, float]) -> str:
    if f.get("payment_failures", 0) >= 2:
        return "Proactive billing support call — prevent involuntary churn"
    if f.get("support_tickets", 0) >= 3:
        return "Assign dedicated CSM for white-glove onboarding review"
    if f.get("days_since_last_login", 0) > 14:
        return "Re-engagement campaign: personalized win-back email sequence"
    return "Schedule product demo to showcase underused features"


def _high_risk_action(f: Dict[str, float]) -> str:
    if f.get("payment_failures", 0) >= 3:
        return "URGENT: Finance escalation + retention offer with billing fix"
    if f.get("days_since_last_login", 0) > 30:
        return "URGENT: Executive outreach + 30-day extension offer"
    if f.get("support_tickets", 0) >= 5:
        return "URGENT: Escalate to engineering + offer SLA credit"
    return "URGENT: Personal call from Account Executive + retention discount"


if __name__ == "__main__":
    # Allow running directly: python ml/train.py
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from datasets.data_generator import generate_customers

    print("[ML] Generating dataset…")
    df = generate_customers(30_000)
    metrics = train(df)
    print("\n=== Final Metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
