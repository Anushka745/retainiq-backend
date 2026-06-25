from fastapi import APIRouter, HTTPException
from models import PredictRequest, PredictResponse
from database import get_connection
from ml import predict_single, FEATURE_COLS
from services import trigger_retention_workflow
import app_state  # module-level model holder

router = APIRouter()


@router.post("/api/predict", response_model=PredictResponse)
def predict_churn(req: PredictRequest):
    model = app_state.MODEL
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run setup first.")

    features = {
        "session_count": req.session_count,
        "avg_session_duration": req.avg_session_duration,
        "days_since_last_login": req.days_since_last_login,
        "support_tickets": req.support_tickets,
        "subscription_age": req.subscription_age,
        "feature_usage_score": req.feature_usage_score,
        "payment_failures": req.payment_failures,
    }

    result = predict_single(features, model)
    user_id = req.user_id or "ADHOC"

    # Persist prediction
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO churn_predictions (user_id, churn_probability, risk_level, recommended_action)
            VALUES (?, ?, ?, ?)
            """,
            (user_id, result["churn_probability"], result["risk_level"], result["recommended_action"]),
        )
        conn.commit()

        # CRM automation trigger
        trigger_retention_workflow(
            conn=conn,
            user_id=user_id,
            churn_probability=result["churn_probability"],
            recommended_action=result["recommended_action"],
        )
    finally:
        conn.close()

    return PredictResponse(user_id=user_id, **result)
