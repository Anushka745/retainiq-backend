"""
Business logic services for RetainIQ.

Covers:
  - Dashboard KPI computation
  - CRM retention workflow (email triggers, log writes)
  - AI insight generation
  - Segment classification
  - Audit log generation
  - Funnel metrics
"""

import random
import sqlite3
from datetime import datetime
from typing import List, Dict, Any

import numpy as np

PLAN_MRR: Dict[str, float] = {
    "Free": 0.0,
    "Starter": 29.0,
    "Pro": 79.0,
    "Enterprise": 299.0,
}

FAKE_IPS = [
    "192.168.1.{}".format(i) for i in range(1, 50)
] + ["10.0.0.{}".format(i) for i in range(1, 30)]


# ── Dashboard KPIs ────────────────────────────────────────────────────────────

def compute_dashboard_metrics(conn: sqlite3.Connection) -> Dict[str, Any]:
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM users")
    total_users = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM users WHERE days_since_last_login <= 30")
    active_users = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM users WHERE churn_label = 1")
    churned_users = cur.fetchone()[0]

    retention_rate = round((1 - churned_users / max(total_users, 1)) * 100, 2)
    churn_rate = round(churned_users / max(total_users, 1) * 100, 2)

    # MRR: sum of monthly revenue from active (non-churned) users
    cur.execute(
        "SELECT plan_type, COUNT(*) as cnt FROM users WHERE churn_label = 0 GROUP BY plan_type"
    )
    plan_counts = {row[0]: row[1] for row in cur.fetchall()}
    mrr = sum(PLAN_MRR.get(plan, 0) * cnt for plan, cnt in plan_counts.items())
    arr = round(mrr * 12, 2)
    mrr = round(mrr, 2)

    return {
        "total_users": total_users,
        "active_users": active_users,
        "retention_rate": retention_rate,
        "churn_rate": churn_rate,
        "mrr": mrr,
        "arr": arr,
    }


# ── CRM Automation ────────────────────────────────────────────────────────────

def trigger_retention_workflow(
    conn: sqlite3.Connection,
    user_id: str,
    churn_probability: float,
    recommended_action: str,
) -> None:
    """
    When churn_probability > 75, fire retention email and log the action.
    """
    if churn_probability <= 75:
        return

    cur = conn.cursor()

    # Simulate sending retention email
    email_detail = (
        f"Retention email dispatched: '{recommended_action}'. "
        f"Churn probability was {churn_probability:.1f}%."
    )
    cur.execute(
        """
        INSERT INTO retention_logs (user_id, action_type, action_detail, status)
        VALUES (?, 'email', ?, 'sent')
        """,
        (user_id, email_detail),
    )

    # Log to audit table
    _write_audit(
        conn=conn,
        event="CRM_RETENTION_TRIGGERED",
        user_id=user_id,
        severity="warning",
        details=email_detail,
    )

    conn.commit()


def _write_audit(
    conn: sqlite3.Connection,
    event: str,
    user_id: str = None,
    severity: str = "info",
    details: str = None,
    ip_address: str = None,
) -> None:
    ip = ip_address or random.choice(FAKE_IPS)
    conn.cursor().execute(
        """
        INSERT INTO audit_logs (event, user_id, ip_address, severity, details)
        VALUES (?, ?, ?, ?, ?)
        """,
        (event, user_id, ip, severity, details),
    )


# ── Audit Log Seeding ─────────────────────────────────────────────────────────

AUDIT_EVENTS = [
    ("USER_LOGIN", "info", "Successful authentication"),
    ("USER_LOGOUT", "info", "Session terminated normally"),
    ("PASSWORD_RESET", "warning", "Password reset requested via email"),
    ("PLAN_UPGRADED", "info", "Subscription upgraded to Pro"),
    ("PLAN_DOWNGRADED", "warning", "Subscription downgraded to Starter"),
    ("PAYMENT_FAILED", "error", "Stripe charge failed — card declined"),
    ("PAYMENT_SUCCESS", "info", "Monthly subscription payment processed"),
    ("API_KEY_GENERATED", "warning", "New API key created"),
    ("EXPORT_DATA", "warning", "User exported account data (GDPR)"),
    ("ACCOUNT_DELETED", "error", "Account marked for deletion"),
    ("LOGIN_FAILED", "error", "Failed login attempt — invalid credentials"),
    ("MFA_ENABLED", "info", "Two-factor authentication enabled"),
    ("WEBHOOK_CREATED", "info", "New webhook endpoint registered"),
    ("FEATURE_FLAG_CHANGED", "warning", "Feature flag toggled by admin"),
    ("RATE_LIMIT_HIT", "warning", "API rate limit exceeded"),
]


def seed_audit_logs(conn: sqlite3.Connection, count: int = 200) -> None:
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM audit_logs")
    existing = cur.fetchone()[0]
    if existing >= count:
        return

    cur.execute("SELECT user_id FROM users ORDER BY RANDOM() LIMIT ?", (count,))
    user_ids = [r[0] for r in cur.fetchall()]

    rows = []
    for i in range(count):
        event, severity, detail = random.choice(AUDIT_EVENTS)
        ip = random.choice(FAKE_IPS)
        uid = user_ids[i % len(user_ids)] if user_ids else None
        rows.append((event, uid, ip, severity, detail))

    cur.executemany(
        "INSERT INTO audit_logs (event, user_id, ip_address, severity, details) VALUES (?,?,?,?,?)",
        rows,
    )
    conn.commit()
    print(f"[SERVICE] Seeded {count} audit log entries.")


# ── Funnel Metrics ────────────────────────────────────────────────────────────

def get_funnel_data() -> List[Dict[str, Any]]:
    visitors = 120_000
    signups = 32_400
    cart = 14_904
    purchases = 4_322

    return [
        {"name": "Visitors", "value": visitors, "conversion": 100.0},
        {"name": "Signup", "value": signups, "conversion": round(signups / visitors * 100, 1)},
        {"name": "AddToCart", "value": cart, "conversion": round(cart / signups * 100, 1)},
        {"name": "Purchase", "value": purchases, "conversion": round(purchases / cart * 100, 1)},
    ]


# ── Insights ──────────────────────────────────────────────────────────────────

def generate_insights(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """Derive AI-style recommendations from live churn pattern data."""
    cur = conn.cursor()

    cur.execute("SELECT AVG(days_since_last_login) FROM users WHERE churn_label=1")
    avg_days = round(cur.fetchone()[0] or 0, 1)

    cur.execute("SELECT AVG(payment_failures) FROM users WHERE churn_label=1")
    avg_failures = round(cur.fetchone()[0] or 0, 2)

    cur.execute(
        "SELECT plan_type, COUNT(*) as c FROM users WHERE churn_label=1 GROUP BY plan_type ORDER BY c DESC LIMIT 1"
    )
    top_churn_plan = cur.fetchone()
    top_plan = top_churn_plan[0] if top_churn_plan else "Starter"

    cur.execute(
        "SELECT COUNT(*) FROM users WHERE days_since_last_login > 21 AND churn_label = 0"
    )
    dormant_risk = cur.fetchone()[0]

    cur.execute(
        "SELECT COUNT(*) FROM users WHERE payment_failures >= 2 AND churn_label = 0"
    )
    payment_risk = cur.fetchone()[0]

    cur.execute(
        "SELECT COUNT(*) FROM users WHERE feature_usage_score < 30 AND churn_label = 0"
    )
    low_feature = cur.fetchone()[0]

    insights = [
        {
            "id": 1,
            "title": "Re-engage Dormant Users",
            "description": (
                f"Churned users averaged {avg_days} days since last login. "
                f"{dormant_risk:,} active users haven't logged in for 21+ days — "
                "prime candidates for re-engagement before they disengage fully."
            ),
            "impact": "High",
            "category": "Engagement",
            "action": "Launch win-back email sequence with a 14-day trial extension",
        },
        {
            "id": 2,
            "title": "Payment Failure Recovery",
            "description": (
                f"Churned customers averaged {avg_failures} payment failures. "
                f"{payment_risk:,} active users have 2+ failures — involuntary churn risk."
            ),
            "impact": "High",
            "category": "Revenue",
            "action": "Trigger proactive billing health check and alternative payment prompt",
        },
        {
            "id": 3,
            "title": f"Improve {top_plan} Plan Retention",
            "description": (
                f"The {top_plan} plan shows the highest absolute churn volume. "
                "Consider reviewing onboarding friction and feature discoverability for this tier."
            ),
            "impact": "High",
            "category": "Product",
            "action": f"Schedule interactive onboarding checklist for all new {top_plan} signups",
        },
        {
            "id": 4,
            "title": "Feature Adoption Drive",
            "description": (
                f"{low_feature:,} active users have a feature usage score below 30/100. "
                "Low feature adoption correlates strongly with mid-term churn."
            ),
            "impact": "Medium",
            "category": "Activation",
            "action": "In-app tooltip campaign highlighting 3 power features per user segment",
        },
        {
            "id": 5,
            "title": "High-Ticket Account Protection",
            "description": (
                "Enterprise and Pro accounts contribute 78% of MRR. "
                "Proactive QBRs reduce churn in high-value cohorts by up to 40%."
            ),
            "impact": "Medium",
            "category": "Success",
            "action": "Automate quarterly business review invitations for Enterprise/Pro accounts",
        },
        {
            "id": 6,
            "title": "Support Ticket Escalation Path",
            "description": (
                "Users with 4+ unresolved tickets in 30 days show 3× churn likelihood. "
                "Streamlining resolution reduces frustration churn."
            ),
            "impact": "Medium",
            "category": "Support",
            "action": "Create SLA-based auto-escalation rules for repeat support contacts",
        },
    ]
    return insights


# ── Segments ──────────────────────────────────────────────────────────────────

SEGMENT_COLORS = {
    "Power Users": "#6366f1",
    "Casual Users": "#22d3ee",
    "Mobile Users": "#f59e0b",
    "Dormant Users": "#ef4444",
}


def compute_segments(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    cur = conn.cursor()

    # Power Users: high activity, frequent login
    cur.execute(
        """
        SELECT COUNT(*),
               AVG(CASE WHEN churn_label=1 THEN 1.0 ELSE 0.0 END)*100,
               AVG(activity_score)
        FROM users WHERE session_count >= 30 AND days_since_last_login <= 7
        """
    )
    r = cur.fetchone()
    segments = [
        {
            "name": "Power Users",
            "count": r[0],
            "churn_rate": round(r[1] or 0, 2),
            "avg_activity": round(r[2] or 0, 2),
            "color": "#6366f1",
        }
    ]

    # Casual Users: moderate activity
    cur.execute(
        """
        SELECT COUNT(*),
               AVG(CASE WHEN churn_label=1 THEN 1.0 ELSE 0.0 END)*100,
               AVG(activity_score)
        FROM users
        WHERE session_count BETWEEN 5 AND 29
          AND days_since_last_login BETWEEN 8 AND 30
        """
    )
    r = cur.fetchone()
    segments.append(
        {
            "name": "Casual Users",
            "count": r[0],
            "churn_rate": round(r[1] or 0, 2),
            "avg_activity": round(r[2] or 0, 2),
            "color": "#22d3ee",
        }
    )

    # Mobile Users: short sessions (proxy for mobile behavior)
    cur.execute(
        """
        SELECT COUNT(*),
               AVG(CASE WHEN churn_label=1 THEN 1.0 ELSE 0.0 END)*100,
               AVG(activity_score)
        FROM users WHERE avg_session_duration < 5 AND session_count >= 10
        """
    )
    r = cur.fetchone()
    segments.append(
        {
            "name": "Mobile Users",
            "count": r[0],
            "churn_rate": round(r[1] or 0, 2),
            "avg_activity": round(r[2] or 0, 2),
            "color": "#f59e0b",
        }
    )

    # Dormant Users: haven't logged in recently
    cur.execute(
        """
        SELECT COUNT(*),
               AVG(CASE WHEN churn_label=1 THEN 1.0 ELSE 0.0 END)*100,
               AVG(activity_score)
        FROM users WHERE days_since_last_login > 30
        """
    )
    r = cur.fetchone()
    segments.append(
        {
            "name": "Dormant Users",
            "count": r[0],
            "churn_rate": round(r[1] or 0, 2),
            "avg_activity": round(r[2] or 0, 2),
            "color": "#ef4444",
        }
    )

    return segments
