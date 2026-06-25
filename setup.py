#!/usr/bin/env python3
"""
RetainIQ one-shot setup script.
Run this BEFORE starting the server:

    python setup.py

It will:
  1. Initialize the SQLite database
  2. Generate 30,000 synthetic customers
  3. Train and save the churn model
  4. Run batch predictions
  5. Trigger CRM workflows for high-risk users
  6. Seed audit logs
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from database import init_db, get_connection
from datasets.data_generator import generate_customers, seed_database
from ml.train import train, load_model, predict_batch, MODEL_PATH
from services import seed_audit_logs, trigger_retention_workflow


def main():
    t0 = time.time()
    print("\n" + "=" * 60)
    print("  RetainIQ Setup — AcmeFlow Churn Platform")
    print("=" * 60 + "\n")

    # ── 1. Database ──────────────────────────────────────────────
    print("Step 1/5: Initializing database…")
    init_db()
    conn = get_connection()
    print("  ✓ Tables created\n")

    # ── 2. Generate customers ────────────────────────────────────
    existing = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    if existing >= 100:
        print(f"Step 2/5: Database already has {existing:,} users — skipping generation.\n")
    else:
        print("Step 2/5: Generating 30,000 synthetic SaaS customers…")
        df = generate_customers(30_000)
        churn_rate = df["churn_label"].mean()
        print(f"  Generated {len(df):,} customers | churn rate: {churn_rate:.2%}")
        seed_database(conn, df)
        print("  ✓ Seeded into database\n")

    # ── 3. Train model ───────────────────────────────────────────
    if MODEL_PATH.exists():
        print(f"Step 3/5: Model already exists at {MODEL_PATH} — skipping training.")
        print("  Delete saved_models/churn_model.pkl to retrain.\n")
    else:
        print("Step 3/5: Training XGBoost/GBM churn model…")
        df = pd.read_sql_query("SELECT * FROM users", conn)
        metrics = train(df)
        print(f"  ✓ Model saved")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1_score']:.4f}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}\n")

    # ── 4. Batch predictions ─────────────────────────────────────
    pred_count = conn.execute("SELECT COUNT(*) FROM churn_predictions").fetchone()[0]
    if pred_count >= 100:
        print(f"Step 4/5: {pred_count:,} predictions already exist — skipping batch run.\n")
    else:
        print("Step 4/5: Running batch churn predictions for all users…")
        model = load_model()
        feat_df = pd.read_sql_query(
            "SELECT user_id, session_count, avg_session_duration, days_since_last_login, "
            "support_tickets, subscription_age, feature_usage_score, payment_failures FROM users",
            conn,
        )
        preds = predict_batch(feat_df, model)
        rows = preds.to_dict(orient="records")
        conn.executemany(
            "INSERT INTO churn_predictions "
            "(user_id, churn_probability, risk_level, recommended_action) "
            "VALUES (:user_id, :churn_probability, :risk_level, :recommended_action)",
            rows,
        )
        conn.commit()

        high_risk = preds[preds["risk_level"] == "High"]
        medium_risk = preds[preds["risk_level"] == "Medium"]
        low_risk = preds[preds["risk_level"] == "Low"]
        print(f"  ✓ {len(rows):,} predictions stored")
        print(f"  High risk: {len(high_risk):,} | Medium: {len(medium_risk):,} | Low: {len(low_risk):,}\n")

        # ── 5. CRM workflows ─────────────────────────────────────
        print(f"Step 5/5: Triggering CRM retention for {len(high_risk):,} high-risk users…")
        for r in rows:
            if r["churn_probability"] > 75:
                trigger_retention_workflow(
                    conn, r["user_id"], r["churn_probability"], r["recommended_action"]
                )
        conn.commit()

        retention_count = conn.execute("SELECT COUNT(*) FROM retention_logs").fetchone()[0]
        print(f"  ✓ {retention_count:,} retention actions logged\n")

    # Audit logs
    seed_audit_logs(conn, count=200)
    conn.close()

    elapsed = time.time() - t0
    print("=" * 60)
    print(f"  Setup complete in {elapsed:.1f}s")
    print("=" * 60)
    print("\nTo start the server:")
    print("  uvicorn app:app --host 0.0.0.0 --port 8000 --reload\n")


if __name__ == "__main__":
    main()
