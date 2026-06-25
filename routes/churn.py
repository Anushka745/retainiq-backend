from typing import List, Optional
from fastapi import APIRouter, Query
from database import get_connection

router = APIRouter()


def _row_to_dict(row) -> dict:
    return dict(row)


@router.get("/api/churn/users")
def get_churn_users(
    limit: int = Query(default=500, le=5000),
    offset: int = Query(default=0),
):
    """All users with churn predictions, paginated."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                u.user_id,
                u.name,
                u.email,
                u.plan_type,
                u.days_since_last_login,
                u.payment_failures,
                COALESCE(p.churn_probability, 0)   AS churn_probability,
                COALESCE(p.risk_level, 'Low')       AS risk_level,
                COALESCE(p.recommended_action, 'Monitor account') AS recommended_action
            FROM users u
            LEFT JOIN (
                SELECT user_id, churn_probability, risk_level, recommended_action
                FROM churn_predictions
                WHERE id IN (
                    SELECT MAX(id) FROM churn_predictions GROUP BY user_id
                )
            ) p ON u.user_id = p.user_id
            ORDER BY churn_probability DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        )
        rows = [_row_to_dict(r) for r in cur.fetchall()]

        cur.execute("SELECT COUNT(*) FROM users")
        total = cur.fetchone()[0]
        return {"total": total, "offset": offset, "limit": limit, "users": rows}
    finally:
        conn.close()


@router.get("/api/churn/high-risk")
def get_high_risk_users(limit: int = Query(default=100, le=1000)):
    """Only users with risk_level = High."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                u.user_id,
                u.name,
                u.email,
                u.plan_type,
                u.days_since_last_login,
                u.payment_failures,
                p.churn_probability,
                p.risk_level,
                p.recommended_action
            FROM users u
            INNER JOIN (
                SELECT user_id, churn_probability, risk_level, recommended_action
                FROM churn_predictions
                WHERE id IN (
                    SELECT MAX(id) FROM churn_predictions GROUP BY user_id
                )
                  AND risk_level = 'High'
            ) p ON u.user_id = p.user_id
            ORDER BY p.churn_probability DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = [_row_to_dict(r) for r in cur.fetchall()]
        return {"count": len(rows), "users": rows}
    finally:
        conn.close()
