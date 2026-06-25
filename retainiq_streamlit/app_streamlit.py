"""
RetainIQ — AcmeFlow Churn Analytics Dashboard
Built with Streamlit — deploy anywhere in one command!
"""

import sys
import json
import time
import sqlite3
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RetainIQ — AcmeFlow",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f172a; }
    .stApp { background-color: #0f172a; color: #e2e8f0; }

    .metric-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 5px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #6366f1;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-top: 4px;
    }
    .risk-high   { color: #ef4444; font-weight: 700; }
    .risk-medium { color: #f59e0b; font-weight: 700; }
    .risk-low    { color: #22c55e; font-weight: 700; }

    .section-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #e2e8f0;
        margin: 20px 0 10px 0;
        border-left: 4px solid #6366f1;
        padding-left: 12px;
    }
    .insight-card {
        background: #1e293b;
        border-radius: 10px;
        padding: 15px;
        margin: 8px 0;
        border-left: 4px solid #6366f1;
    }
    div[data-testid="stMetricValue"] { color: #6366f1; font-size: 1.8rem; }
    div[data-testid="stMetricLabel"] { color: #94a3b8; }
    .stDataFrame { background: #1e293b; }
    [data-testid="stSidebar"] { background-color: #1e293b; }
</style>
""", unsafe_allow_html=True)


# ── Startup / Data Loading ────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def initialize_app():
    """Run setup once and cache everything."""
    from database import init_db, get_connection
    from datasets.data_generator import generate_customers, seed_database
    from ml.train import train, load_model, predict_batch, MODEL_PATH
    from services import seed_audit_logs, trigger_retention_workflow

    init_db()
    conn = get_connection()

    existing = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    if existing < 100:
        df = generate_customers(30_000)
        seed_database(conn, df)

    if not MODEL_PATH.exists():
        df = pd.read_sql_query("SELECT * FROM users", conn)
        train(df)

    model = load_model()

    pred_count = conn.execute("SELECT COUNT(*) FROM churn_predictions").fetchone()[0]
    if pred_count < 100:
        feat_df = pd.read_sql_query(
            "SELECT user_id, session_count, avg_session_duration, days_since_last_login,"
            " support_tickets, subscription_age, feature_usage_score, payment_failures FROM users",
            conn,
        )
        preds = predict_batch(feat_df, model)
        rows = preds.to_dict(orient="records")
        conn.executemany(
            "INSERT INTO churn_predictions (user_id,churn_probability,risk_level,recommended_action)"
            " VALUES (:user_id,:churn_probability,:risk_level,:recommended_action)",
            rows,
        )
        conn.commit()
        for r in rows:
            if r["churn_probability"] > 75:
                trigger_retention_workflow(conn, r["user_id"], r["churn_probability"], r["recommended_action"])
        conn.commit()

    seed_audit_logs(conn, count=200)
    conn.close()
    return model


@st.cache_data(ttl=300)
def load_dashboard_data():
    from database import get_connection
    from services import (compute_dashboard_metrics, get_funnel_data,
                          generate_insights, compute_segments)
    conn = get_connection()
    dashboard  = compute_dashboard_metrics(conn)
    funnel     = get_funnel_data()
    insights   = generate_insights(conn)
    segments   = compute_segments(conn)
    conn.close()
    return dashboard, funnel, insights, segments


@st.cache_data(ttl=300)
def load_churn_users():
    from database import get_connection
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT u.user_id, u.name, u.email, u.plan_type,
               u.days_since_last_login, u.payment_failures,
               COALESCE(p.churn_probability, 0)          AS churn_probability,
               COALESCE(p.risk_level, 'Low')              AS risk_level,
               COALESCE(p.recommended_action,'Monitor')   AS recommended_action
        FROM users u
        LEFT JOIN (
            SELECT user_id, churn_probability, risk_level, recommended_action
            FROM churn_predictions
            WHERE id IN (SELECT MAX(id) FROM churn_predictions GROUP BY user_id)
        ) p ON u.user_id = p.user_id
        ORDER BY churn_probability DESC
    """, conn)
    conn.close()
    return df


@st.cache_data(ttl=300)
def load_audit_logs():
    from database import get_connection
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT id, event, user_id, ip_address, severity, details, logged_at "
        "FROM audit_logs ORDER BY id DESC LIMIT 100",
        conn,
    )
    conn.close()
    return df


@st.cache_data(ttl=300)
def load_model_metrics():
    from ml.train import get_metrics
    return get_metrics()


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar():
    st.sidebar.image("https://img.icons8.com/fluency/48/combo-chart.png", width=40)
    st.sidebar.title("RetainIQ")
    st.sidebar.caption("AcmeFlow Churn Platform")
    st.sidebar.divider()

    page = st.sidebar.radio(
        "Navigation",
        ["📊 Dashboard", "👥 Churn Users", "🔮 Predict Churn",
         "💡 Insights", "🔒 Security Logs", "🤖 Model Info"],
        label_visibility="collapsed",
    )

    st.sidebar.divider()
    metrics = load_model_metrics()
    st.sidebar.markdown("**Model Performance**")
    st.sidebar.metric("ROC-AUC",  f"{metrics.get('roc_auc', 0):.4f}")
    st.sidebar.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
    st.sidebar.caption(f"Backend: {metrics.get('backend','—')}")
    return page


# ── Pages ─────────────────────────────────────────────────────────────────────

def page_dashboard():
    st.title("📊 RetainIQ Dashboard")
    st.caption("AcmeFlow — Live Churn Analytics")

    dashboard, funnel, insights, segments = load_dashboard_data()

    # ── KPI Cards ─────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1: st.metric("Total Users",     f"{dashboard['total_users']:,}")
    with c2: st.metric("Active Users",    f"{dashboard['active_users']:,}")
    with c3: st.metric("Retention Rate",  f"{dashboard['retention_rate']}%")
    with c4: st.metric("Churn Rate",      f"{dashboard['churn_rate']}%")
    with c5: st.metric("MRR",             f"${dashboard['mrr']:,.0f}")
    with c6: st.metric("ARR",             f"${dashboard['arr']:,.0f}")

    st.divider()

    # ── Charts Row ────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Conversion Funnel</div>', unsafe_allow_html=True)
        funnel_df = pd.DataFrame(funnel)
        fig = go.Figure(go.Funnel(
            y=funnel_df["name"],
            x=funnel_df["value"],
            textinfo="value+percent initial",
            marker=dict(color=["#6366f1", "#818cf8", "#a5b4fc", "#c7d2fe"]),
        ))
        fig.update_layout(
            paper_bgcolor="#1e293b", plot_bgcolor="#1e293b",
            font=dict(color="#e2e8f0"), margin=dict(l=10, r=10, t=10, b=10),
            height=320,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">User Segments</div>', unsafe_allow_html=True)
        seg_df = pd.DataFrame(segments)
        fig2 = px.bar(
            seg_df, x="name", y="count", color="churn_rate",
            color_continuous_scale=["#22c55e", "#f59e0b", "#ef4444"],
            labels={"count": "Users", "churn_rate": "Churn %"},
        )
        fig2.update_layout(
            paper_bgcolor="#1e293b", plot_bgcolor="#1e293b",
            font=dict(color="#e2e8f0"), margin=dict(l=10, r=10, t=10, b=10),
            height=320, showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Segment Stats Table ───────────────────────────────────────────────
    st.markdown('<div class="section-title">Segment Statistics</div>', unsafe_allow_html=True)
    seg_display = seg_df[["name", "count", "churn_rate", "avg_activity"]].copy()
    seg_display.columns = ["Segment", "Users", "Churn Rate (%)", "Avg Activity"]
    st.dataframe(seg_display, use_container_width=True, hide_index=True)

    # ── Top Insights Preview ──────────────────────────────────────────────
    st.markdown('<div class="section-title">Top AI Insights</div>', unsafe_allow_html=True)
    for ins in insights[:3]:
        color = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#22c55e"}.get(ins["impact"], "#6366f1")
        st.markdown(f"""
        <div class="insight-card" style="border-left-color:{color}">
            <strong style="color:{color}">[{ins['impact']}]</strong>
            <strong style="color:#e2e8f0"> {ins['title']}</strong><br/>
            <span style="color:#94a3b8;font-size:0.9rem">{ins['description']}</span><br/>
            <span style="color:#6366f1;font-size:0.85rem">→ {ins['action']}</span>
        </div>
        """, unsafe_allow_html=True)


def page_churn_users():
    st.title("👥 Churn Users")
    st.caption("All users ranked by churn probability")

    df = load_churn_users()

    # ── Filters ───────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        risk_filter = st.multiselect("Risk Level", ["High", "Medium", "Low"],
                                     default=["High", "Medium", "Low"])
    with col2:
        plan_filter = st.multiselect("Plan Type",
                                     df["plan_type"].unique().tolist(),
                                     default=df["plan_type"].unique().tolist())
    with col3:
        search = st.text_input("Search by name or email", "")

    filtered = df[
        df["risk_level"].isin(risk_filter) &
        df["plan_type"].isin(plan_filter)
    ]
    if search:
        filtered = filtered[
            filtered["name"].str.contains(search, case=False) |
            filtered["email"].str.contains(search, case=False)
        ]

    # ── Summary Chips ─────────────────────────────────────────────────────
    high   = (filtered["risk_level"] == "High").sum()
    medium = (filtered["risk_level"] == "Medium").sum()
    low    = (filtered["risk_level"] == "Low").sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Showing",       f"{len(filtered):,} users")
    c2.metric("🔴 High Risk",  f"{high:,}")
    c3.metric("🟡 Medium Risk",f"{medium:,}")
    c4.metric("🟢 Low Risk",   f"{low:,}")

    # ── Churn Probability Distribution ───────────────────────────────────
    fig = px.histogram(
        filtered, x="churn_probability", nbins=40,
        color="risk_level",
        color_discrete_map={"High":"#ef4444","Medium":"#f59e0b","Low":"#22c55e"},
        title="Churn Probability Distribution",
    )
    fig.update_layout(
        paper_bgcolor="#1e293b", plot_bgcolor="#1e293b",
        font=dict(color="#e2e8f0"), height=280,
        margin=dict(l=10,r=10,t=40,b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Table ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">User Table</div>', unsafe_allow_html=True)

    display_df = filtered[[
        "user_id", "name", "email", "plan_type",
        "churn_probability", "risk_level",
        "days_since_last_login", "payment_failures",
        "recommended_action"
    ]].copy()
    display_df["churn_probability"] = display_df["churn_probability"].map("{:.1f}%".format)

    st.dataframe(
        display_df.head(500),
        use_container_width=True,
        hide_index=True,
        column_config={
            "churn_probability": st.column_config.TextColumn("Churn %"),
            "risk_level": st.column_config.TextColumn("Risk"),
            "recommended_action": st.column_config.TextColumn("Action", width="large"),
        }
    )

    # ── Download Button ───────────────────────────────────────────────────
    csv = filtered.to_csv(index=False)
    st.download_button("⬇️ Download CSV", csv, "churn_users.csv", "text/csv")


def page_predict():
    st.title("🔮 Predict Churn")
    st.caption("Enter user behavior data to get an instant churn prediction")

    model = initialize_app()

    with st.form("predict_form"):
        st.markdown("#### User Behavior Features")
        c1, c2 = st.columns(2)

        with c1:
            session_count         = st.slider("Session Count (last 30 days)", 0, 200, 10)
            avg_session_duration  = st.slider("Avg Session Duration (mins)", 0.0, 60.0, 12.0)
            days_since_last_login = st.slider("Days Since Last Login", 0, 365, 15)
            support_tickets       = st.slider("Support Tickets Raised", 0, 20, 1)

        with c2:
            subscription_age    = st.slider("Subscription Age (days)", 1, 1460, 180)
            feature_usage_score = st.slider("Feature Usage Score (0–100)", 0.0, 100.0, 50.0)
            payment_failures    = st.slider("Payment Failures", 0, 10, 0)
            user_id_input       = st.text_input("User ID (optional)", "USR-DEMO01")

        submitted = st.form_submit_button("🚀 Predict Churn", use_container_width=True)

    if submitted:
        from ml.train import predict_single

        features = {
            "session_count":         session_count,
            "avg_session_duration":  avg_session_duration,
            "days_since_last_login": days_since_last_login,
            "support_tickets":       support_tickets,
            "subscription_age":      subscription_age,
            "feature_usage_score":   feature_usage_score,
            "payment_failures":      payment_failures,
        }

        result = predict_single(features, model)
        prob   = result["churn_probability"]
        risk   = result["risk_level"]
        action = result["recommended_action"]

        color = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#22c55e"}.get(risk, "#6366f1")

        st.divider()
        st.markdown("### Prediction Result")

        c1, c2, c3 = st.columns(3)
        c1.metric("Churn Probability", f"{prob:.1f}%")
        c2.metric("Risk Level", risk)
        c3.metric("User ID", user_id_input or "ADHOC")

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Churn Risk Meter", "font": {"color": "#e2e8f0"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#94a3b8"},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 40],  "color": "#1a2e1a"},
                    {"range": [40, 70], "color": "#2e2a1a"},
                    {"range": [70, 100],"color": "#2e1a1a"},
                ],
                "threshold": {"line": {"color": "white","width": 2}, "value": prob},
            },
            number={"suffix": "%", "font": {"color": color}},
        ))
        fig.update_layout(
            paper_bgcolor="#1e293b", font=dict(color="#e2e8f0"),
            height=280, margin=dict(l=20,r=20,t=40,b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"""
        <div class="insight-card" style="border-left-color:{color}">
            <strong style="color:{color}">Recommended Action</strong><br/>
            <span style="color:#e2e8f0;font-size:1rem">{action}</span>
        </div>
        """, unsafe_allow_html=True)

        # CRM log simulation
        if prob > 75:
            st.warning(f"⚠️ CRM Triggered: Retention email queued for {user_id_input}")


def page_insights():
    st.title("💡 AI Insights")
    st.caption("Recommendations generated from live churn patterns")

    _, _, insights, _ = load_dashboard_data()

    for ins in insights:
        color = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#22c55e"}.get(ins["impact"], "#6366f1")
        with st.expander(f"{'🔴' if ins['impact']=='High' else '🟡' if ins['impact']=='Medium' else '🟢'} {ins['title']} — {ins['category']}", expanded=(ins["impact"]=="High")):
            st.markdown(f"**Impact:** :{('red' if ins['impact']=='High' else 'orange' if ins['impact']=='Medium' else 'green')}[{ins['impact']}]")
            st.markdown(f"**Insight:** {ins['description']}")
            st.markdown(f"""
            <div style="background:#0f172a;border-radius:8px;padding:12px;margin-top:8px">
                <strong style="color:#6366f1">→ Recommended Action:</strong><br/>
                <span style="color:#e2e8f0">{ins['action']}</span>
            </div>
            """, unsafe_allow_html=True)


def page_security():
    st.title("🔒 Security & Audit Logs")
    st.caption("System events and compliance trail")

    df = load_audit_logs()

    # ── Filters ───────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        sev_filter = st.multiselect("Severity", ["error", "warning", "info"],
                                    default=["error", "warning", "info"])
    with col2:
        event_search = st.text_input("Search events", "")

    filtered = df[df["severity"].isin(sev_filter)]
    if event_search:
        filtered = filtered[filtered["event"].str.contains(event_search, case=False)]

    # ── Severity Breakdown ────────────────────────────────────────────────
    sev_counts = filtered["severity"].value_counts().reset_index()
    sev_counts.columns = ["severity", "count"]
    fig = px.pie(
        sev_counts, values="count", names="severity",
        color="severity",
        color_discrete_map={"error":"#ef4444","warning":"#f59e0b","info":"#6366f1"},
        title="Events by Severity",
    )
    fig.update_layout(
        paper_bgcolor="#1e293b", font=dict(color="#e2e8f0"),
        height=250, margin=dict(l=10,r=10,t=40,b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Table ─────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Events",   len(filtered))
    c2.metric("🔴 Errors",      (filtered["severity"]=="error").sum())
    c3.metric("🟡 Warnings",    (filtered["severity"]=="warning").sum())

    st.dataframe(
        filtered[["logged_at","severity","event","user_id","ip_address","details"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "severity": st.column_config.TextColumn("Severity"),
            "details":  st.column_config.TextColumn("Details", width="large"),
        }
    )


def page_model():
    st.title("🤖 Model Information")
    st.caption("XGBoost / GBM churn classifier details")

    metrics = load_model_metrics()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy",  f"{metrics.get('accuracy',0):.4f}")
    c2.metric("Precision", f"{metrics.get('precision',0):.4f}")
    c3.metric("Recall",    f"{metrics.get('recall',0):.4f}")
    c4.metric("F1 Score",  f"{metrics.get('f1_score',0):.4f}")
    c5.metric("ROC-AUC",   f"{metrics.get('roc_auc',0):.4f}")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Model Details")
        st.json({
            "Backend":        metrics.get("backend", "—"),
            "Train Samples":  f"{metrics.get('train_samples',0):,}",
            "Test Samples":   f"{metrics.get('test_samples',0):,}",
            "Features":       [
                "session_count", "avg_session_duration",
                "days_since_last_login", "support_tickets",
                "subscription_age", "feature_usage_score",
                "payment_failures"
            ],
        })

    with col2:
        st.markdown("#### Risk Classification Rules")
        rules_df = pd.DataFrame([
            {"Churn Probability", "Risk Level", "Action Type"},
        ])
        st.markdown("""
        | Probability | Risk Level | Action |
        |---|---|---|
        | 0% – 40%   | 🟢 Low    | Monitor / Reward |
        | 41% – 70%  | 🟡 Medium | Engage / Re-activate |
        | 71% – 100% | 🔴 High   | URGENT Intervention |
        """)

    # ── Metric Bar Chart ──────────────────────────────────────────────────
    metric_names  = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    metric_values = [
        metrics.get("accuracy",0), metrics.get("precision",0),
        metrics.get("recall",0),   metrics.get("f1_score",0),
        metrics.get("roc_auc",0),
    ]
    fig = px.bar(
        x=metric_names, y=metric_values,
        color=metric_values,
        color_continuous_scale=["#334155","#6366f1"],
        title="Model Performance Metrics",
        labels={"x":"Metric","y":"Score"},
    )
    fig.update_layout(
        paper_bgcolor="#1e293b", plot_bgcolor="#1e293b",
        font=dict(color="#e2e8f0"), height=320,
        margin=dict(l=10,r=10,t=40,b=10),
        showlegend=False, coloraxis_showscale=False,
        yaxis=dict(range=[0,1]),
    )
    fig.add_hline(y=0.8, line_dash="dot", line_color="#94a3b8",
                  annotation_text="Target 0.8", annotation_position="right")
    st.plotly_chart(fig, use_container_width=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Initialize once
    with st.spinner("🚀 Loading RetainIQ... (first run may take 1–2 mins to train model)"):
        initialize_app()

    page = render_sidebar()

    if   page == "📊 Dashboard":      page_dashboard()
    elif page == "👥 Churn Users":     page_churn_users()
    elif page == "🔮 Predict Churn":  page_predict()
    elif page == "💡 Insights":        page_insights()
    elif page == "🔒 Security Logs":   page_security()
    elif page == "🤖 Model Info":      page_model()


if __name__ == "__main__":
    main()
