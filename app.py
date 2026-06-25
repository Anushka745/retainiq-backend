"""
RetainIQ Backend — FastAPI + ML churn prediction platform for AcmeFlow.

Startup sequence:
  1. Create SQLite tables
  2. Generate & seed 30,000 synthetic customers (idempotent)
  3. Train XGBoost/GBM churn model (skipped if saved model exists)
  4. Run batch prediction for all users (skipped if predictions exist)
  5. Seed audit log samples
  6. Serve API
"""

import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Resolve imports when running from backend/ directory
sys.path.insert(0, str(Path(__file__).parent))

import app_state
from database import init_db, get_connection
from datasets.data_generator import generate_customers, seed_database
from ml.train import train, load_model, get_metrics, predict_batch, BACKEND, MODEL_PATH
from services import seed_audit_logs, trigger_retention_workflow
from routes import dashboard_router, churn_router, predict_router, analytics_router


# ── Startup / Shutdown ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n" + "=" * 60)
    print("  RetainIQ — AcmeFlow Churn Platform")
    print("=" * 60)

    t0 = time.time()

    # 1. DB tables
    init_db()

    conn = get_connection()

    # 2. Seed customers (idempotent)
    conn.execute("SELECT COUNT(*) FROM users")
    existing = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    if existing < 100:
        print("[STARTUP] Generating 30,000 synthetic customers…")
        df = generate_customers(30_000)
        seed_database(conn, df)
    else:
        print(f"[STARTUP] Found {existing:,} existing users — skipping data generation.")
        df = pd.read_sql("SELECT * FROM users", conn.__class__.__module__ and conn)

    # 3. Train model (skip if saved model exists)
    if not MODEL_PATH.exists():
        print("[STARTUP] Training churn model…")
        if existing >= 100:
            # Load from DB
            import sqlite3
            raw_conn = conn
            df = pd.read_sql_query("SELECT * FROM users", raw_conn)
        metrics = train(df)
        print(f"[STARTUP] Model trained. ROC-AUC: {metrics['roc_auc']}")
    else:
        print(f"[STARTUP] Saved model found — loading from {MODEL_PATH}")

    app_state.MODEL = load_model()
    app_state.MODEL_BACKEND = BACKEND
    print(f"[STARTUP] Model loaded ({BACKEND}). ✓")

    # 4. Batch predictions (skip if already computed)
    pred_count = conn.execute("SELECT COUNT(*) FROM churn_predictions").fetchone()[0]
    if pred_count < 100:
        print("[STARTUP] Running batch churn predictions…")
        feat_df = pd.read_sql_query(
            "SELECT user_id, session_count, avg_session_duration, days_since_last_login, "
            "support_tickets, subscription_age, feature_usage_score, payment_failures "
            "FROM users",
            conn,
        )
        preds = predict_batch(feat_df, app_state.MODEL)

        rows = preds.to_dict(orient="records")
        conn.executemany(
            "INSERT INTO churn_predictions (user_id, churn_probability, risk_level, recommended_action) "
            "VALUES (:user_id, :churn_probability, :risk_level, :recommended_action)",
            rows,
        )
        conn.commit()
        print(f"[STARTUP] Stored {len(rows):,} churn predictions.")

        # CRM: trigger retention for high-risk users
        high_risk = [r for r in rows if r["churn_probability"] > 75]
        print(f"[STARTUP] Triggering CRM workflow for {len(high_risk):,} high-risk users…")
        for r in high_risk[:500]:   # cap to avoid blocking startup too long
            trigger_retention_workflow(
                conn, r["user_id"], r["churn_probability"], r["recommended_action"]
            )
        conn.commit()
    else:
        print(f"[STARTUP] {pred_count:,} predictions already exist — skipping batch run.")

    # 5. Seed audit logs
    seed_audit_logs(conn, count=200)

    conn.close()

    elapsed = time.time() - t0
    print(f"\n[STARTUP] Ready in {elapsed:.1f}s ✓")
    print("=" * 60 + "\n")

    yield  # ← app runs here

    print("[SHUTDOWN] RetainIQ shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="RetainIQ — AcmeFlow Churn API",
    description=(
        "ML-powered churn prediction and retention analytics backend. "
        "XGBoost / GradientBoosting model trained on 30,000 synthetic SaaS customers."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow frontend on any origin (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────────────────────

app.include_router(dashboard_router, tags=["Dashboard"])
app.include_router(churn_router, tags=["Churn"])
app.include_router(predict_router, tags=["Predict"])
app.include_router(analytics_router, tags=["Analytics"])


@app.get("/", tags=["Health"])
def root():
    return {
        "service": "RetainIQ Churn API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "ok",
    }


@app.get("/health", tags=["Health"])
def health():
    conn = get_connection()
    total = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    conn.close()
    return {
        "status": "ok",
        "model_loaded": app_state.MODEL is not None,
        "model_backend": app_state.MODEL_BACKEND,
        "total_users": total,
        "metrics": get_metrics(),
    }
