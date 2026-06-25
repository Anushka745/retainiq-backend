from fastapi import APIRouter
from database import get_connection
from services import compute_dashboard_metrics

router = APIRouter()


@router.get("/api/dashboard")
def get_dashboard():
    conn = get_connection()
    try:
        return compute_dashboard_metrics(conn)
    finally:
        conn.close()
