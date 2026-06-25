from fastapi import APIRouter, Query
from database import get_connection
from services import get_funnel_data, generate_insights, compute_segments

router = APIRouter()


@router.get("/api/funnel")
def get_funnel():
    return {"steps": get_funnel_data()}


@router.get("/api/insights")
def get_insights():
    conn = get_connection()
    try:
        return {"insights": generate_insights(conn)}
    finally:
        conn.close()


@router.get("/api/segments")
def get_segments():
    conn = get_connection()
    try:
        return {"segments": compute_segments(conn)}
    finally:
        conn.close()


@router.get("/api/security/logs")
def get_security_logs(limit: int = Query(default=50, le=500)):
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, event, user_id, ip_address, severity, details, logged_at
            FROM audit_logs
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = [dict(r) for r in cur.fetchall()]
        return {"count": len(rows), "logs": rows}
    finally:
        conn.close()
