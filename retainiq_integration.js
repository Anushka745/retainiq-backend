/**
 * RetainIQ Frontend ↔ Backend Integration
 * =========================================
 * Drop this <script> block into your existing index.html just before </body>.
 * It replaces all hardcoded arrays with live API calls.
 *
 * Change BASE_URL to match wherever you host the FastAPI backend.
 */

const BASE_URL = "http://localhost:8000"; // ← update for production

// ── Utility ──────────────────────────────────────────────────────────────────

async function apiFetch(path) {
  try {
    const res = await fetch(`${BASE_URL}${path}`);
    if (!res.ok) throw new Error(`HTTP ${res.status} from ${path}`);
    return await res.json();
  } catch (err) {
    console.error(`[RetainIQ API] ${path}:`, err);
    return null;
  }
}

// ── Dashboard KPIs ────────────────────────────────────────────────────────────

async function loadDashboard() {
  const data = await apiFetch("/api/dashboard");
  if (!data) return;

  // Map to existing frontend element IDs (adjust selectors to match your HTML)
  const set = (id, val) => {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
  };

  set("total-users",     data.total_users?.toLocaleString());
  set("active-users",    data.active_users?.toLocaleString());
  set("retention-rate",  `${data.retention_rate}%`);
  set("churn-rate",      `${data.churn_rate}%`);
  set("mrr",             `$${data.mrr?.toLocaleString()}`);
  set("arr",             `$${data.arr?.toLocaleString()}`);

  // Alternative: data attributes pattern
  document.querySelectorAll("[data-metric]").forEach(el => {
    const key = el.dataset.metric;
    if (data[key] !== undefined) el.textContent = data[key];
  });
}

// ── Churn Users Table ─────────────────────────────────────────────────────────

async function loadChurnUsers() {
  const data = await apiFetch("/api/churn/users?limit=200");
  if (!data) return;

  // Replace the global churnUsers array used by the frontend
  window.churnUsers = data.users;

  // If frontend uses renderChurnTable(), call it; otherwise rebuild tbody
  if (typeof window.renderChurnTable === "function") {
    window.renderChurnTable(data.users);
    return;
  }

  const tbody = document.querySelector("#churn-table tbody, #churnTable tbody, table tbody");
  if (!tbody) return;

  tbody.innerHTML = data.users.map(u => `
    <tr>
      <td>${u.user_id}</td>
      <td>${u.name}</td>
      <td>${u.email}</td>
      <td>${u.plan_type}</td>
      <td>
        <span class="risk-badge risk-${u.risk_level.toLowerCase()}">
          ${u.risk_level}
        </span>
      </td>
      <td>${u.churn_probability.toFixed(1)}%</td>
      <td>${u.recommended_action}</td>
    </tr>
  `).join("");
}

// ── High-Risk Users ───────────────────────────────────────────────────────────

async function loadHighRiskUsers() {
  const data = await apiFetch("/api/churn/high-risk?limit=50");
  if (!data) return;
  window.highRiskUsers = data.users;

  const container = document.getElementById("high-risk-list");
  if (!container) return;

  container.innerHTML = data.users.slice(0, 10).map(u => `
    <div class="risk-item">
      <strong>${u.name}</strong> (${u.plan_type})
      <span class="prob">${u.churn_probability.toFixed(1)}%</span>
      <small>${u.recommended_action}</small>
    </div>
  `).join("");
}

// ── Funnel ────────────────────────────────────────────────────────────────────

async function loadFunnel() {
  const data = await apiFetch("/api/funnel");
  if (!data) return;

  // Replace hardcoded funnelSteps array
  window.funnelSteps = data.steps;

  if (typeof window.renderFunnel === "function") {
    window.renderFunnel(data.steps);
  }

  // Recharts / Chart.js: update data source if a chart is already rendered
  data.steps.forEach(step => {
    const el = document.getElementById(`funnel-${step.name.toLowerCase()}`);
    if (el) el.textContent = step.value.toLocaleString();
  });
}

// ── Insights ──────────────────────────────────────────────────────────────────

async function loadInsights() {
  const data = await apiFetch("/api/insights");
  if (!data) return;

  // Replace hardcoded insightsData
  window.insightsData = data.insights;

  if (typeof window.renderInsights === "function") {
    window.renderInsights(data.insights);
    return;
  }

  const container = document.getElementById("insights-container, #insights-list");
  const target = document.querySelector("#insights-container, #insights-list, .insights");
  if (!target) return;

  target.innerHTML = data.insights.map(ins => `
    <div class="insight-card insight-${ins.impact.toLowerCase()}" data-id="${ins.id}">
      <div class="insight-header">
        <span class="badge impact-${ins.impact.toLowerCase()}">${ins.impact}</span>
        <span class="category">${ins.category}</span>
      </div>
      <h4>${ins.title}</h4>
      <p>${ins.description}</p>
      <div class="insight-action">
        <strong>Recommended:</strong> ${ins.action}
      </div>
    </div>
  `).join("");
}

// ── Segments ──────────────────────────────────────────────────────────────────

async function loadSegments() {
  const data = await apiFetch("/api/segments");
  if (!data) return;

  window.segmentData = data.segments;

  if (typeof window.renderSegments === "function") {
    window.renderSegments(data.segments);
  }
}

// ── Audit / Security Logs ─────────────────────────────────────────────────────

async function loadAuditLog() {
  const data = await apiFetch("/api/security/logs?limit=50");
  if (!data) return;

  // Replace hardcoded auditLog array
  window.auditLog = data.logs;

  if (typeof window.renderAuditLog === "function") {
    window.renderAuditLog(data.logs);
    return;
  }

  const tbody = document.querySelector("#audit-table tbody, #auditTable tbody, #security-table tbody");
  if (!tbody) return;

  const severityClass = s => ({ error: "danger", warning: "warn", info: "info" }[s] || "info");

  tbody.innerHTML = data.logs.map(log => `
    <tr>
      <td>${log.logged_at}</td>
      <td><span class="severity-badge ${severityClass(log.severity)}">${log.severity.toUpperCase()}</span></td>
      <td>${log.event}</td>
      <td>${log.user_id || "—"}</td>
      <td>${log.ip_address || "—"}</td>
      <td>${log.details || "—"}</td>
    </tr>
  `).join("");
}

// ── Predict (on-demand form) ──────────────────────────────────────────────────

async function predictChurn(formData) {
  try {
    const res = await fetch(`${BASE_URL}/api/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(formData),
    });
    return await res.json();
  } catch (err) {
    console.error("[RetainIQ API] /api/predict:", err);
    return null;
  }
}

// Wire prediction form if it exists
document.addEventListener("DOMContentLoaded", () => {
  const predictForm = document.getElementById("predict-form");
  if (predictForm) {
    predictForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const fd = new FormData(predictForm);
      const payload = Object.fromEntries(
        [...fd.entries()].map(([k, v]) => [k, isNaN(v) ? v : Number(v)])
      );
      const result = await predictChurn(payload);
      const output = document.getElementById("predict-result");
      if (output && result) {
        output.innerHTML = `
          <div class="prediction-result risk-${result.risk_level?.toLowerCase()}">
            <strong>Risk: ${result.risk_level}</strong>
            — Churn probability: ${result.churn_probability?.toFixed(1)}%<br/>
            <em>${result.recommended_action}</em>
          </div>`;
      }
    });
  }
});

// ── Bootstrap ─────────────────────────────────────────────────────────────────

async function initRetainIQ() {
  console.log("[RetainIQ] Loading live data from", BASE_URL);
  await Promise.all([
    loadDashboard(),
    loadChurnUsers(),
    loadHighRiskUsers(),
    loadFunnel(),
    loadInsights(),
    loadSegments(),
    loadAuditLog(),
  ]);
  console.log("[RetainIQ] Live data loaded ✓");
}

// Run on page load
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initRetainIQ);
} else {
  initRetainIQ();
}
