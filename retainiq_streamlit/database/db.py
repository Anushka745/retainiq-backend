"""
Database setup and connection management.
Uses SQLite via SQLAlchemy (sync) for simplicity and portability.
Swap DATABASE_URL to PostgreSQL for production.
"""

import os
import sqlite3
from pathlib import Path

DATABASE_PATH = Path(__file__).parent.parent / "datasets" / "retainiq.db"
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"


def get_connection() -> sqlite3.Connection:
    """Return a raw SQLite connection (used in startup scripts)."""
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create all tables if they don't exist."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id         TEXT UNIQUE NOT NULL,
            name            TEXT NOT NULL,
            email           TEXT UNIQUE NOT NULL,
            signup_date     TEXT NOT NULL,
            plan_type       TEXT NOT NULL,
            activity_score  REAL NOT NULL,
            churn_label     INTEGER NOT NULL DEFAULT 0,
            session_count           INTEGER NOT NULL DEFAULT 0,
            avg_session_duration    REAL NOT NULL DEFAULT 0,
            days_since_last_login   INTEGER NOT NULL DEFAULT 0,
            support_tickets         INTEGER NOT NULL DEFAULT 0,
            subscription_age        INTEGER NOT NULL DEFAULT 0,
            feature_usage_score     REAL NOT NULL DEFAULT 0,
            payment_failures        INTEGER NOT NULL DEFAULT 0,
            created_at      TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS churn_predictions (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id             TEXT NOT NULL,
            churn_probability   REAL NOT NULL,
            risk_level          TEXT NOT NULL,
            recommended_action  TEXT NOT NULL,
            predicted_at        TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        );

        CREATE TABLE IF NOT EXISTS retention_logs (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id         TEXT NOT NULL,
            action_type     TEXT NOT NULL,
            action_detail   TEXT NOT NULL,
            status          TEXT NOT NULL DEFAULT 'sent',
            triggered_at    TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        );

        CREATE TABLE IF NOT EXISTS audit_logs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            event       TEXT NOT NULL,
            user_id     TEXT,
            ip_address  TEXT,
            severity    TEXT NOT NULL DEFAULT 'info',
            details     TEXT,
            logged_at   TEXT DEFAULT (datetime('now'))
        );
        """
    )

    conn.commit()
    conn.close()
    print(f"[DB] Tables initialized at {DATABASE_PATH}")
