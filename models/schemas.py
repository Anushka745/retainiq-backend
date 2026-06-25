"""
Pydantic request/response schemas.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


# ── Prediction ────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    user_id: Optional[str] = None
    session_count: int = Field(..., ge=0, le=2000)
    avg_session_duration: float = Field(..., ge=0, le=480)
    days_since_last_login: int = Field(..., ge=0, le=730)
    support_tickets: int = Field(..., ge=0, le=100)
    subscription_age: int = Field(..., ge=0, le=3650)
    feature_usage_score: float = Field(..., ge=0, le=100)
    payment_failures: int = Field(..., ge=0, le=50)

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "USR-TEST01",
                "session_count": 3,
                "avg_session_duration": 4.2,
                "days_since_last_login": 45,
                "support_tickets": 5,
                "subscription_age": 90,
                "feature_usage_score": 18.0,
                "payment_failures": 2,
            }
        }


class PredictResponse(BaseModel):
    user_id: Optional[str]
    churn_probability: float
    risk_level: str
    recommended_action: str


# ── Dashboard ─────────────────────────────────────────────────────────────────

class DashboardMetrics(BaseModel):
    total_users: int
    active_users: int
    retention_rate: float
    churn_rate: float
    mrr: float
    arr: float


# ── Churn Users ───────────────────────────────────────────────────────────────

class ChurnUser(BaseModel):
    user_id: str
    name: str
    email: str
    plan_type: str
    churn_probability: float
    risk_level: str
    recommended_action: str
    days_since_last_login: int
    payment_failures: int


# ── Funnel ────────────────────────────────────────────────────────────────────

class FunnelStep(BaseModel):
    name: str
    value: int
    conversion: Optional[float] = None


# ── Insights ──────────────────────────────────────────────────────────────────

class Insight(BaseModel):
    id: int
    title: str
    description: str
    impact: str        # High | Medium | Low
    category: str
    action: str


# ── Segments ──────────────────────────────────────────────────────────────────

class Segment(BaseModel):
    name: str
    count: int
    churn_rate: float
    avg_activity: float
    color: str


# ── Audit Logs ────────────────────────────────────────────────────────────────

class AuditLog(BaseModel):
    id: int
    event: str
    user_id: Optional[str]
    ip_address: Optional[str]
    severity: str
    details: Optional[str]
    logged_at: str


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_backend: str
    total_users: int
    metrics: dict
