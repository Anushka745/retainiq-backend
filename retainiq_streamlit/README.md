# RetainIQ — AcmeFlow Churn Prediction Platform

ML-powered SaaS churn prediction backend built with **FastAPI + XGBoost/GBM + SQLite**.  
Serves live analytics to the RetainIQ frontend dashboard.

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI 0.111 + Uvicorn |
| ML | XGBoost (or sklearn GBM fallback) |
| Data | Pandas, NumPy, Scikit-Learn |
| Database | SQLite (swap URL for PostgreSQL) |
| Serialization | Pydantic v2 |

---

## Project Structure

```
backend/
├── app.py                    # FastAPI application entry point
├── app_state.py              # Global model holder
├── setup.py                  # One-shot setup script
├── retainiq_integration.js   # Frontend ↔ backend bridge script
│
├── routes/
│   ├── dashboard.py          # GET /api/dashboard
│   ├── churn.py              # GET /api/churn/users, /api/churn/high-risk
│   ├── predict.py            # POST /api/predict
│   └── analytics.py          # GET /api/funnel, /api/insights, /api/segments, /api/security/logs
│
├── models/
│   └── schemas.py            # Pydantic request/response models
│
├── services/
│   └── business.py           # KPI computation, CRM automation, insights, segments
│
├── database/
│   └── db.py                 # SQLite init, connection factory
│
├── datasets/
│   ├── data_generator.py     # 30,000 synthetic SaaS customers
│   └── retainiq.db           # Auto-created SQLite database
│
├── ml/
│   └── train.py              # Model training, prediction, risk classification
│
├── saved_models/
│   ├── churn_model.pkl       # Trained pipeline (auto-generated)
│   └── model_metrics.json    # Accuracy, precision, recall, F1, ROC-AUC
│
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Run setup (generates data + trains model)

```bash
python setup.py
```

This will:
- Create SQLite tables
- Generate 30,000 synthetic AcmeFlow customers
- Train XGBoost churn classifier (~ROC-AUC 0.79)
- Run batch predictions for all users
- Trigger CRM email workflows for high-risk users
- Seed 200 audit log entries

### 3. Start the server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

API docs available at: **http://localhost:8000/docs**

### 4. Connect the frontend

Add this line to your `index.html` just before `</body>`:

```html
<script src="retainiq_integration.js"></script>
```

Then update the `BASE_URL` at the top of `retainiq_integration.js`:

```js
const BASE_URL = "http://localhost:8000"; // or your deployed URL
```

---

## API Endpoints

### `GET /api/dashboard`
Returns KPI metrics for the top dashboard cards.

```json
{
  "total_users": 30000,
  "active_users": 21600,
  "retention_rate": 72.0,
  "churn_rate": 28.0,
  "mrr": 1489200.0,
  "arr": 17870400.0
}
```

---

### `GET /api/churn/users?limit=200&offset=0`
All users with churn predictions, paginated.

```json
{
  "total": 30000,
  "offset": 0,
  "limit": 200,
  "users": [
    {
      "user_id": "USR-391044AD",
      "name": "Diane Evans",
      "email": "diane.evans.0@gmail.com",
      "plan_type": "Pro",
      "churn_probability": 87.4,
      "risk_level": "High",
      "recommended_action": "URGENT: Personal call from Account Executive + retention discount",
      "days_since_last_login": 62,
      "payment_failures": 3
    }
  ]
}
```

---

### `GET /api/churn/high-risk?limit=100`
Only users with `risk_level = High`.

```json
{
  "count": 4821,
  "users": [...]
}
```

---

### `GET /api/funnel`
Marketing funnel conversion data.

```json
{
  "steps": [
    { "name": "Visitors",   "value": 120000, "conversion": 100.0 },
    { "name": "Signup",     "value": 32400,  "conversion": 27.0  },
    { "name": "AddToCart",  "value": 14904,  "conversion": 46.0  },
    { "name": "Purchase",   "value": 4322,   "conversion": 29.0  }
  ]
}
```

---

### `GET /api/insights`
AI-generated retention recommendations based on live churn patterns.

```json
{
  "insights": [
    {
      "id": 1,
      "title": "Re-engage Dormant Users",
      "description": "Churned users averaged 42.3 days since last login...",
      "impact": "High",
      "category": "Engagement",
      "action": "Launch win-back email sequence with a 14-day trial extension"
    }
  ]
}
```

---

### `GET /api/segments`
User segment statistics.

```json
{
  "segments": [
    { "name": "Power Users",  "count": 4210, "churn_rate": 4.2,  "avg_activity": 91.3, "color": "#6366f1" },
    { "name": "Casual Users", "count": 12400,"churn_rate": 26.1, "avg_activity": 54.7, "color": "#22d3ee" },
    { "name": "Mobile Users", "count": 6800, "churn_rate": 32.6, "avg_activity": 48.2, "color": "#f59e0b" },
    { "name": "Dormant Users","count": 7200, "churn_rate": 41.3, "avg_activity": 22.1, "color": "#ef4444" }
  ]
}
```

---

### `GET /api/security/logs?limit=50`
Audit trail for compliance and monitoring.

```json
{
  "count": 50,
  "logs": [
    {
      "id": 247,
      "event": "PAYMENT_FAILED",
      "user_id": "USR-7A2C1E09",
      "ip_address": "192.168.1.14",
      "severity": "error",
      "details": "Stripe charge failed — card declined",
      "logged_at": "2024-05-01 14:32:11"
    }
  ]
}
```

---

### `POST /api/predict`
Real-time churn prediction for a single user.

**Request:**
```json
{
  "user_id": "USR-TEST01",
  "session_count": 3,
  "avg_session_duration": 4.2,
  "days_since_last_login": 45,
  "support_tickets": 5,
  "subscription_age": 90,
  "feature_usage_score": 18.0,
  "payment_failures": 2
}
```

**Response:**
```json
{
  "user_id": "USR-TEST01",
  "churn_probability": 84.3,
  "risk_level": "High",
  "recommended_action": "URGENT: Personal call from Account Executive + retention discount"
}
```

**Risk Levels:**
| Range | Level |
|---|---|
| 0–40% | Low |
| 41–70% | Medium |
| 71–100% | High |

---

### `GET /health`
Backend + model health check.

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_backend": "sklearn-gbm",
  "total_users": 30000,
  "metrics": {
    "accuracy": 0.7648,
    "precision": 0.6237,
    "recall": 0.4036,
    "f1_score": 0.4901,
    "roc_auc": 0.7867
  }
}
```

---

## Model Performance

Trained on 30,000 synthetic AcmeFlow customers (24k train / 6k test):

| Metric | Score |
|---|---|
| Accuracy | 0.765 |
| Precision | 0.624 |
| Recall | 0.404 |
| F1 Score | 0.490 |
| ROC-AUC | **0.787** |

> **To use real XGBoost:** `pip install xgboost` — the code auto-detects it.

---

## CRM Automation

When `churn_probability > 75%`:
1. Retention email is simulated and logged to `retention_logs` table
2. Audit entry is written with `severity = warning`
3. Recommended action is personalized based on user behavior signals

---

## Database Schema

```sql
users               -- 30,000 AcmeFlow customers with behavioral features
churn_predictions   -- Latest ML prediction per user
retention_logs      -- CRM automation history
audit_logs          -- Security and compliance events
```

---

## Deployment

### Render / Railway

```yaml
# render.yaml
services:
  - type: web
    name: retainiq-api
    env: python
    buildCommand: "pip install -r requirements.txt && python setup.py"
    startCommand: "uvicorn app:app --host 0.0.0.0 --port $PORT"
```

### PythonAnywhere

1. Upload the `backend/` folder via Files tab
2. Open a Bash console:
   ```bash
   pip install -r requirements.txt --user
   python setup.py
   ```
3. Create a Web App → Manual config → WSGI file:
   ```python
   import sys
   sys.path.insert(0, '/home/yourusername/backend')
   from app import app as application
   ```

### Environment Variables (Production)

```env
PORT=8000
DATABASE_URL=sqlite:///datasets/retainiq.db
# For PostgreSQL: DATABASE_URL=postgresql://user:pass@host/dbname
CORS_ORIGINS=https://your-frontend-domain.com
```

---

## Retrain the Model

```bash
# Delete existing model and re-run setup
rm saved_models/churn_model.pkl saved_models/model_metrics.json
python setup.py
```

---

## Frontend Integration

The included `retainiq_integration.js` script:
- Replaces all hardcoded JS arrays (`churnUsers`, `insightsData`, `auditLog`, `funnelSteps`) with live API calls
- Calls `renderChurnTable()`, `renderInsights()`, `renderAuditLog()`, `renderFunnel()` if those functions exist in your frontend
- Falls back to directly updating DOM elements by ID if they don't
- Wires up any `#predict-form` for real-time predictions
- Runs all fetches in parallel via `Promise.all`

**One-line integration:**
```html
<script src="retainiq_integration.js"></script>
```
