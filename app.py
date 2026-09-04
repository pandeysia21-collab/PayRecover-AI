import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="PayRecover AI",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# FILE NAMES
# =========================================================
MODEL_FILE = "payrecover_model.pkl"
HISTORY_FILE = "prediction_history.csv"
BATCH_FILE = "batch_recovery_results.csv"

# =========================================================
# SIMPLE CSS
# =========================================================
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    .big-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }
    .subtitle {
        font-size: 20px;
        opacity: 0.85;
        margin-bottom: 10px;
    }
    .description {
        font-size: 16px;
        line-height: 1.6;
        opacity: 0.8;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)

try:
    model = load_model()
    model_loaded = True
    model_error = None
except Exception as e:
    model = None
    model_loaded = False
    model_error = str(e)

# =========================================================
# LOAD PREDICTION HISTORY
# =========================================================
if "history" not in st.session_state:
    try:
        history_df = pd.read_csv(HISTORY_FILE)
        st.session_state.history = (
            [] if history_df.empty else history_df.to_dict("records")
        )
    except (FileNotFoundError, pd.errors.EmptyDataError):
        st.session_state.history = []

# =========================================================
# FEATURE ENGINEERING
# =========================================================
def create_features(data):
    data = data.copy()

    data["total_previous_payments"] = (
        data["previous_successful_payments"]
        + data["previous_failed_payments"]
    )

    data["customer_success_rate"] = (
        data["previous_successful_payments"]
        / (data["total_previous_payments"] + 1)
    )

    data["customer_failure_rate"] = (
        data["previous_failed_payments"]
        / (data["total_previous_payments"] + 1)
    )

    data["amount_deviation"] = (
        abs(data["amount"] - data["customer_avg_amount"])
        / (data["customer_avg_amount"] + 1)
    )

    data["retry_pressure"] = (
        data["retry_count"]
        / (data["time_since_failure_hours"] + 1)
    )

    return data

# =========================================================
# BOUNDED RECOVERY AGENT
# =========================================================
MAX_RETRIES = 3
LOW_PROBABILITY_THRESHOLD = 0.35
HIGH_PROBABILITY_THRESHOLD = 0.70

def recovery_agent_policy(probability, retry_count):
    """Single source of truth for bounded recovery decisions."""
    if retry_count >= MAX_RETRIES:
        return {
            "priority": "LOW",
            "action": "Stop repeated retries",
            "retry_allowed": False,
            "policy_reason": "Maximum retry limit reached",
            "execution": "BLOCKED",
            "escalation": "Stop automation"
        }
    if probability < LOW_PROBABILITY_THRESHOLD:
        return {
            "priority": "LOW",
            "action": "Avoid repeated retry",
            "retry_allowed": False,
            "policy_reason": "Recovery probability below 35% threshold",
            "execution": "BLOCKED",
            "escalation": "No automatic recovery action"
        }
    if probability >= HIGH_PROBABILITY_THRESHOLD:
        return {
            "priority": "HIGH",
            "action": "Retry payment immediately",
            "retry_allowed": True,
            "policy_reason": "High recovery probability",
            "execution": "SANDBOX RETRY",
            "escalation": "Retry payment"
        }
    return {
        "priority": "MEDIUM",
        "action": "Suggest alternative payment method",
        "retry_allowed": True,
        "policy_reason": "Moderate recovery probability",
        "execution": "SANDBOX ALTERNATIVE",
        "escalation": "Alternative payment method"
    }

def execute_recovery_agent(row):
    """Run a bounded offline/sandbox workflow; never moves real money."""
    tx_id = str(row.get("transaction_id", row.get("payment_id", "TX-DEMO")))
    probability = float(row.get("recovery_probability", 0.0))
    retry_count = int(row.get("retry_count", 0))
    amount = float(row.get("amount", 0.0))
    policy = recovery_agent_policy(probability, retry_count)
    events = [
        ("Detect", "Failed payment event received"),
        ("Score", f"Recovery probability = {probability:.2%}"),
        ("Decide", f"{policy['priority']} → {policy['action']}"),
        ("Policy Gate", policy["policy_reason"]),
    ]
    observed = float(row.get("measured_recovery_amount", 0.0) or 0.0)
    if policy["retry_allowed"]:
        events.append(("Execute", f"{policy['execution']} executed in offline sandbox"))
        events.append(("Observe", f"Offline observed outcome ₹{observed:,.2f}"))
    else:
        observed = 0.0
        events.append(("Execute", "BLOCKED by recovery policy"))
        events.append(("Observe", "Safely stopped; no automated recovery action executed"))
    events.append(("Audit", f"{tx_id} decision, guardrail and outcome recorded"))
    return {"transaction_id": tx_id, "amount": amount, "probability": probability,
            "retry_count": retry_count, **policy, "observed_recovery": observed,
            "events": events}

# =========================================================
# DECISION ENGINE
# =========================================================
def get_decision(probability):
    if probability >= 0.70:
        return "HIGH", "Retry payment immediately"
    elif probability >= 0.35:
        return "MEDIUM", "Suggest alternative payment method"
    else:
        return "LOW", "Avoid repeated retry"

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("💳 PayRecover AI")
st.sidebar.caption("Revenue Recovery Intelligence")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "🤖 Recovery Agent",
        "🔮 Recovery Predictor",
        "📊 Analytics",
        "📋 Prediction History"
    ]
)

st.sidebar.divider()

if model_loaded:
    st.sidebar.success("🤖 ML Model Online")
    st.sidebar.write("🌳 Gradient Boosting")
    st.sidebar.write("🧠 15 Engineered Features")
    st.sidebar.write("🎯 Decision Engine Active")
else:
    st.sidebar.error("❌ ML Model Offline")
    st.sidebar.caption(model_error)

# =========================================================
# DASHBOARD
# =========================================================
if page == "🏠 Dashboard":

    st.markdown(
        '<div class="big-title">💳 PayRecover AI</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="subtitle">AI-Powered Payment Failure & Revenue Recovery Intelligence</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="description">'
        'PayRecover AI analyzes failed payments, predicts recovery probability, '
        'recommends the next best recovery action, and measures recovery outcomes '
        'in an offline batch evaluation.'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    st.subheader("🤖 Agentic Recovery Loop")
    loop = st.columns(6)
    for col, label in zip(loop, ["Detect", "Score", "Decide", "Gate", "Execute", "Audit"]):
        col.success(label)
    st.caption("The agent converts an ML recovery score into a bounded action, executes only approved sandbox actions, observes the outcome and records an audit trail.")

    # =====================================================
    # SYSTEM STATUS
    # =====================================================
    st.header("🤖 System Status")
    s1, s2, s3, s4 = st.columns(4)

    with s1:
        st.write("🤖 **ML Model**")
        st.success("Online" if model_loaded else "Offline")

    with s2:
        st.write("🌳 **Algorithm**")
        st.write("**Gradient Boosting**")

    with s3:
        st.write("🧠 **Engineered Features**")
        st.write("**15**")

    with s4:
        st.write("🎯 **Decision Engine**")
        st.success("Active")

    st.divider()

    # =====================================================
    # LOAD BATCH DATA
    # =====================================================
    try:
        batch_df = pd.read_csv(BATCH_FILE)

        required_cols = [
            "amount",
            "recovery_probability",
            "expected_recovery",
            "retry_allowed"
        ]

        missing = [c for c in required_cols if c not in batch_df.columns]
        if missing:
            st.error(f"Missing columns in batch file: {', '.join(missing)}")
            st.stop()

        # -------------------------------------------------
        # CLEAN DATA
        # -------------------------------------------------
        batch_df["amount"] = pd.to_numeric(
            batch_df["amount"], errors="coerce"
        ).fillna(0)

        batch_df["recovery_probability"] = pd.to_numeric(
            batch_df["recovery_probability"], errors="coerce"
        ).fillna(0).clip(0, 1)

        batch_df["expected_recovery"] = pd.to_numeric(
            batch_df["expected_recovery"], errors="coerce"
        ).fillna(0)

        # -------------------------------------------------
        # RETRY ALLOWED
        # -------------------------------------------------
        batch_df["retry_allowed"] = (
            batch_df["retry_allowed"]
            .astype(str)
            .str.lower()
            .isin(["true", "1", "yes"])
        )

        # =================================================
        # BASIC BATCH METRICS
        # =================================================
        total_failed = len(batch_df)
        revenue_at_risk = batch_df["amount"].sum()
        avg_probability = batch_df["recovery_probability"].mean() * 100

        eligible_df = batch_df[
            batch_df["retry_allowed"] == True
        ].copy()

        eligible_payments = len(eligible_df)
        eligible_revenue = eligible_df["amount"].sum()

        expected_recovery = eligible_df["expected_recovery"].sum()
        expected_recovery = min(expected_recovery, eligible_revenue)

        # =================================================
        # MEASURED / OBSERVED RECOVERY
        # =================================================
        if "measured_recovery_amount" in eligible_df.columns:
            measured_recovery = pd.to_numeric(
                eligible_df["measured_recovery_amount"],
                errors="coerce"
            ).fillna(0).sum()
        elif "observed_recovered_amount" in eligible_df.columns:
            measured_recovery = pd.to_numeric(
                eligible_df["observed_recovered_amount"],
                errors="coerce"
            ).fillna(0).sum()
        else:
            measured_recovery = 0.0

        measured_recovery = min(measured_recovery, eligible_revenue)

        measured_recovery_rate = (
            measured_recovery / eligible_revenue * 100
            if eligible_revenue > 0 else 0
        )

        # =================================================
        # REVENUE RECOVERY INTELLIGENCE
        # =================================================
        st.header("🎯 Revenue Recovery Intelligence")

        r1, r2, r3, r4 = st.columns(4)

        with r1:
            st.metric("💳 Failed Payments", f"{total_failed:,}")

        with r2:
            st.metric("💰 Revenue At Risk", f"₹{revenue_at_risk:,.0f}")

        with r3:
            st.metric("🎯 Expected Recovery", f"₹{expected_recovery:,.0f}")

        with r4:
            st.metric("📈 Avg Recovery Probability",
                      f"{avg_probability:.2f}%")

        st.caption(
            "Model-based estimates from the current failed-payment batch."
        )

        st.divider()

        # =================================================
        # OFFLINE / SYNTHETIC BATCH EVALUATION
        # =================================================
        st.header("💵 Offline / Synthetic Batch Evaluation")

        e1, e2, e3, e4 = st.columns(4)

        with e1:
            st.metric(
                "Eligible Payments",
                f"{eligible_payments:,}"
            )

        with e2:
            st.metric(
                "Eligible Revenue",
                f"₹{eligible_revenue:,.0f}"
            )

        with e3:
            st.metric(
                "Measured Recovery",
                f"₹{measured_recovery:,.0f}"
            )

        with e4:
            st.metric(
                "Measured Recovery Rate",
                f"{measured_recovery_rate:.2f}%"
            )

        st.success(
            f"💰 Offline batch outcome: **₹{measured_recovery:,.2f}** "
            f"observed recovery from **{eligible_payments:,}** eligible payments."
        )

        st.caption(
            "Measured observed recovery is calculated from the dataset's "
            "recovered outcome. This is an offline/synthetic evaluation, "
            "not live money recovered from real customers."
        )

        st.divider()

        # =================================================
        # RECOVERY OPPORTUNITY
        # =================================================
        st.header("🚀 Recovery Opportunity")

        q1, q2, q3 = st.columns(3)

        with q1:
            st.metric(
                "Potential Recovery Opportunities",
                f"{eligible_payments:,}"
            )

        with q2:
            st.metric(
                "Eligible Revenue",
                f"₹{eligible_revenue:,.0f}"
            )

        with q3:
            st.metric(
                "Model-Based Recovery",
                f"₹{expected_recovery:,.0f}"
            )

        st.info(
            "💡 PayRecover AI focuses recovery efforts on payments where "
            "the model identifies meaningful recovery potential, while "
            "limiting unnecessary repeated retries."
        )

        st.divider()

        # =================================================
        # SMART RECOVERY QUEUE
        # =================================================
        st.header("🧠 Smart Recovery Queue")

        if len(eligible_df) > 0:

            eligible_df["recovery_score"] = (
                eligible_df["amount"]
                * eligible_df["recovery_probability"]
            )

            recovery_queue = (
                eligible_df
                .sort_values("recovery_score", ascending=False)
                .copy()
            )

            if "recommended_action" in recovery_queue.columns:
                recovery_queue["Recommended Action"] = (
                    recovery_queue["recommended_action"]
                )
            else:
                recovery_queue["Recommended Action"] = (
                    recovery_queue["recovery_probability"]
                    .apply(
                        lambda p:
                        "🚀 Retry Payment" if p >= 0.70
                        else "🔄 Alternative Payment" if p >= 0.35
                        else "🛑 Do Not Retry"
                    )
                )

            recovery_queue["Expected Recovery"] = (
                recovery_queue["amount"]
                * recovery_queue["recovery_probability"]
            )

            display_columns = [
                "amount",
                "recovery_probability",
                "Expected Recovery",
                "Recommended Action"
            ]

            display_df = recovery_queue[
                display_columns
            ].head(10).copy()

            display_df = display_df.rename(
                columns={
                    "amount": "Payment Amount",
                    "recovery_probability": "Recovery Probability"
                }
            )

            display_df["Recovery Probability"] = (
                display_df["Recovery Probability"] * 100
            ).round(2).astype(str) + "%"

            display_df["Payment Amount"] = display_df[
                "Payment Amount"
            ].apply(lambda x: f"₹{x:,.0f}")

            display_df["Expected Recovery"] = display_df[
                "Expected Recovery"
            ].apply(lambda x: f"₹{x:,.2f}")

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "Payments are ranked by estimated recovery impact: "
                "payment amount × recovery probability."
            )

        else:
            st.info("No eligible recovery opportunities found.")

        st.divider()

        # =================================================
        # AI RECOVERY PIPELINE
        # =================================================
        st.header("⚙️ AI Recovery Pipeline")

        p1, p2, p3 = st.columns(3)

        with p1:
            st.subheader("1️⃣ 💳 Payment Data")
            st.write(
                "Transaction amount, payment method, failure reason "
                "and customer context."
            )

        with p2:
            st.subheader("2️⃣ 🔧 Feature Engineering")
            st.write(
                "Customer success rate, failure rate, amount deviation "
                "and retry pressure."
            )

        with p3:
            st.subheader("3️⃣ 🤖 Gradient Boosting")
            st.write(
                "Machine-learning model analyzes the engineered "
                "payment signals."
            )

        p4, p5, p6 = st.columns(3)

        with p4:
            st.subheader("4️⃣ 📈 Recovery Probability")
            st.write(
                "Model estimates the likelihood that the payment "
                "can be recovered."
            )

        with p5:
            st.subheader("5️⃣ 🎯 Decision Engine")
            st.write(
                "Converts probability into the next best recovery action."
            )

        with p6:
            st.subheader("6️⃣ 💰 Revenue Impact")
            st.write(
                "Expected recovery value helps prioritize financially "
                "valuable payments."
            )

        st.divider()

        # =================================================
        # REVENUE IMPACT
        # =================================================
        st.header("💰 Revenue Impact")

        revenue_chart = pd.DataFrame(
            {
                "Category": [
                    "Revenue At Risk",
                    "Eligible Revenue",
                    "Expected Recovery",
                    "Measured Recovery"
                ],
                "Amount": [
                    revenue_at_risk,
                    eligible_revenue,
                    expected_recovery,
                    measured_recovery
                ]
            }
        )

        st.bar_chart(
            revenue_chart.set_index("Category")
        )

        st.caption(
            "Comparison of failed-payment value, eligible revenue, "
            "model-estimated recovery and observed offline recovery."
        )

        st.divider()

        # =================================================
        # AI DECISION COUNTS
        # =================================================
        st.header("📌 Batch AI Decisions")

        decision_counts = pd.DataFrame(
            {
                "Decision": [
                    "HIGH / Retry",
                    "MEDIUM / Alternative",
                    "LOW / Stop"
                ],
                "Payments": [
                    int(
                        (batch_df.get("priority", pd.Series(dtype=str))
                         .astype(str).str.upper() == "HIGH").sum()
                    ),
                    int(
                        (batch_df.get("priority", pd.Series(dtype=str))
                         .astype(str).str.upper() == "MEDIUM").sum()
                    ),
                    int(
                        (batch_df.get("priority", pd.Series(dtype=str))
                         .astype(str).str.upper() == "LOW").sum()
                    )
                ]
            }
        )

        # Fallback to recommended action if priority is absent.
        if "priority" not in batch_df.columns:
            probs = batch_df["recovery_probability"]
            decision_counts["Payments"] = [
                int((probs >= 0.70).sum()),
                int(((probs >= 0.35) & (probs < 0.70)).sum()),
                int((probs < 0.35).sum())
            ]

        st.bar_chart(
            decision_counts.set_index("Decision")
        )

        st.divider()

        # =================================================
        # RECOVERY AUDIT TRAIL
        # =================================================
        st.header("🧾 Recovery Audit Trail")

        audit_columns = [
            "transaction_id",
            "recovery_probability",
            "priority",
            "recommended_action",
            "retry_allowed",
            "stopping_reason",
            "measured_recovery_amount"
        ]

        available_audit = [
            c for c in audit_columns if c in batch_df.columns
        ]

        if available_audit:

            audit_df = batch_df[available_audit].copy()

            rename_map = {
                "transaction_id": "Transaction",
                "recovery_probability": "Recovery Probability",
                "priority": "Priority",
                "recommended_action": "Action",
                "retry_allowed": "Retry Allowed",
                "stopping_reason": "Policy Reason",
                "measured_recovery_amount": "Observed Recovery"
            }

            audit_df = audit_df.rename(columns=rename_map)

            if "Recovery Probability" in audit_df.columns:
                audit_df["Recovery Probability"] = (
                    pd.to_numeric(
                        audit_df["Recovery Probability"],
                        errors="coerce"
                    ).fillna(0) * 100
                ).round(2).astype(str) + "%"

            if "Observed Recovery" in audit_df.columns:
                audit_df["Observed Recovery"] = (
                    pd.to_numeric(
                        audit_df["Observed Recovery"],
                        errors="coerce"
                    ).fillna(0)
                    .apply(lambda x: f"₹{x:,.2f}")
                )

            st.dataframe(
                audit_df.head(20),
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "Each row records the model probability, policy decision, "
                "retry control and observed batch outcome."
            )

            audit_csv = batch_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "⬇️ Download Full Recovery Audit Trail",
                data=audit_csv,
                file_name="payrecover_audit_trail.csv",
                mime="text/csv"
            )

        else:
            st.warning(
                "Audit columns were not found in the current batch file."
            )

        st.divider()

        # =================================================
        # CONTROL STATUS
        # =================================================
        st.header("🛡️ Recovery Controls")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.success("✅ Audit Trail Enabled")

        with c2:
            st.success("🛑 Stopping Rules Enabled")

        with c3:
            st.success("🎯 Bounded Recovery Actions")

        st.caption(
            "The decision layer prevents uncontrolled retries. "
            "Retry eligibility is limited by probability thresholds "
            "and the retry-count stopping rule."
        )

        st.divider()

        # =================================================
        # DECISION POLICY
        # =================================================
        st.header("🎯 AI Decision Policy")

        d1, d2, d3 = st.columns(3)

        with d1:
            st.success(
                "### HIGH — 70%+\n\n"
                "Strong recovery opportunity.\n\n"
                "**Action:** Retry payment"
            )

        with d2:
            st.warning(
                "### MEDIUM — 35% to <70%\n\n"
                "Moderate recovery opportunity.\n\n"
                "**Action:** Suggest alternative payment"
            )

        with d3:
            st.error(
                "### LOW — <35%\n\n"
                "Low recovery opportunity.\n\n"
                "**Action:** Avoid repeated retry"
            )

        st.caption(
            "The ML model predicts probability; the decision engine "
            "converts that probability into an operational action. "
            "Payments reaching the retry limit are stopped."
        )

        st.divider()

        # =================================================
        # WHY PAYRECOVER
        # =================================================
        st.header("🏆 Why PayRecover AI?")

        w1, w2, w3 = st.columns(3)

        with w1:
            st.subheader("🧠 Intelligent")
            st.write(
                "Uses customer history, payment behaviour, retry "
                "activity and transaction patterns instead of a "
                "simple retry rule."
            )

        with w2:
            st.subheader("💰 Revenue Focused")
            st.write(
                "Converts recovery probability into expected "
                "recoverable value so businesses can prioritize "
                "financial impact."
            )

        with w3:
            st.subheader("🛑 Risk Aware")
            st.write(
                "Limits repeated retries and redirects uncertain "
                "cases toward alternative payment methods."
            )

        st.divider()

        # =================================================
        # PAYMENT PLATFORM INTELLIGENCE
        # =================================================
        st.header("💼 Proposed Payment-Platform Intelligence Layer")

        st.write(
            "PayRecover AI is designed as a proposed intelligence "
            "layer that could operate alongside a payment platform."
        )

        u1, u2, u3, u4 = st.columns(4)

        with u1:
            st.subheader("1️⃣ Failed Payment")
            st.write("Payment failure event is received.")

        with u2:
            st.subheader("2️⃣ AI Prediction")
            st.write("Recovery probability is estimated.")

        with u3:
            st.subheader("3️⃣ Smart Action")
            st.write("Retry, redirect or stop.")

        with u4:
            st.subheader("4️⃣ Revenue Recovery")
            st.write("Potential recovery is prioritized and measured.")

        st.info(
            "🎯 Business Goal: Reduce unnecessary retries, improve "
            "recovery prioritization and help payment platforms focus "
            "recovery efforts where they are most valuable."
        )

        st.caption(
            "This is a proposed intelligence layer and does not "
            "represent an actual payment-platform integration."
        )

        st.divider()

        # =================================================
        # ML METHODOLOGY
        # =================================================
        st.header("🧠 ML Methodology")

        st.write(
            "The system combines a machine-learning probability "
            "model with a rule-based decision engine."
        )

        m1, m2, m3 = st.columns(3)

        with m1:
            st.subheader("📥 Input")
            st.write(
                "Transaction data\n\n"
                "Customer behaviour\n\n"
                "Failure information\n\n"
                "Retry history"
            )

        with m2:
            st.subheader("🤖 ML Layer")
            st.write(
                "Gradient Boosting\n\n"
                "15 engineered features\n\n"
                "Recovery probability"
            )

        with m3:
            st.subheader("🎯 Decision Layer")
            st.write(
                "Probability thresholds\n\n"
                "Retry control\n\n"
                "Alternative payment\n\n"
                "Expected recovery"
            )

        st.divider()

        # =================================================
        # MODEL PERFORMANCE
        # =================================================
        st.header("📊 Model Performance")

        perf1, perf2, perf3 = st.columns(3)

        with perf1:
            st.write("**Algorithm**")
            st.write("### Gradient Boosting")

        with perf2:
            st.write("**Accuracy**")
            st.write("### 68.67%")

        with perf3:
            st.write("**ROC-AUC**")
            st.write("### 73.48%")

        st.caption(
            "Performance values are based on the current test evaluation."
        )

    except FileNotFoundError:
        st.warning(
            "⚠️ batch_recovery_results.csv was not found. "
            "Run batch_recovery.py first."
        )
    except Exception as e:
        st.error(f"Could not load dashboard data: {e}")

# =========================================================
# RECOVERY PREDICTOR
# =========================================================
elif page == "🤖 Recovery Agent":

    st.title("🤖 PayRecover Recovery Agent")
    st.write("Detect → Score → Decide → Gate → Execute → Observe → Audit")
    st.info("Offline/sandbox demonstration only — no real customer money is moved.")

    try:
        agent_df = pd.read_csv(BATCH_FILE)
    except Exception as e:
        st.error(f"Could not load batch recovery data: {e}")
        st.stop()

    if agent_df.empty or "recovery_probability" not in agent_df.columns:
        st.warning("No suitable batch transactions are available.")
        st.stop()

    if "transaction_id" not in agent_df.columns:
        agent_df["transaction_id"] = [f"TX-{i+1:05d}" for i in range(len(agent_df))]
    if "retry_count" not in agent_df.columns:
        agent_df["retry_count"] = 0
    if "amount" not in agent_df.columns:
        agent_df["amount"] = 0.0
    if "measured_recovery_amount" not in agent_df.columns:
        agent_df["measured_recovery_amount"] = 0.0

    agent_df["recovery_probability"] = pd.to_numeric(agent_df["recovery_probability"], errors="coerce").fillna(0.0)
    if agent_df["recovery_probability"].max() > 1:
        agent_df["recovery_probability"] /= 100.0

    selected_id = st.selectbox("Select a failed payment", agent_df["transaction_id"].astype(str).tolist())
    row = agent_df[agent_df["transaction_id"].astype(str) == selected_id].iloc[0].to_dict()
    preview = execute_recovery_agent(row)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Recovery Probability", f"{preview['probability']:.2%}")
    c2.metric("Retry Count", str(preview["retry_count"]))
    c3.metric("Priority", preview["priority"])
    c4.metric("Retry Allowed", "YES" if preview["retry_allowed"] else "NO")

    st.subheader("🛡️ Hard Guardrails")
    g1,g2,g3 = st.columns(3)
    g1.success("Maximum retries: 3")
    g2.success("No automatic retry below 35%")
    g3.success("Medium risk → alternative payment")

    st.subheader("🎯 Agent Decision")
    st.dataframe(pd.DataFrame([{
        "Transaction ID": preview["transaction_id"],
        "Priority": preview["priority"],
        "Next Best Action": preview["action"],
        "Policy Gate": preview["policy_reason"],
        "Execution": preview["execution"],
        "Escalation": preview["escalation"]
    }]), use_container_width=True, hide_index=True)

    if st.button("▶️ Run Bounded Recovery Agent", type="primary"):
        result = execute_recovery_agent(row)
        st.subheader("⚙️ Execution Trace")
        st.dataframe(pd.DataFrame([{
            "Stage": stage, "Agent Event": event
        } for stage,event in result["events"]]), use_container_width=True, hide_index=True)

        if result["retry_allowed"]:
            st.success(f"Sandbox action completed. Offline observed outcome: ₹{result['observed_recovery']:,.2f}")
        else:
            st.warning("Agent stopped safely — policy gate blocked automated recovery.")

        st.subheader("🧾 Audit Record")
        st.dataframe(pd.DataFrame([{
            "Transaction ID": result["transaction_id"],
            "Recovery Probability": f"{result['probability']:.2%}",
            "Priority": result["priority"],
            "Action": result["action"],
            "Retry Allowed": result["retry_allowed"],
            "Policy Reason": result["policy_reason"],
            "Execution": result["execution"],
            "Observed Recovery": f"₹{result['observed_recovery']:,.2f}",
            "Environment": "OFFLINE / SYNTHETIC SANDBOX"
        }]), use_container_width=True, hide_index=True)

    st.caption("Production boundary: connect the same bounded policy to payment-platform test mode only after authorization, idempotency, webhook verification, rate limits and a hard stop path.")

elif page == "🔮 Recovery Predictor":

    st.title("🔮 Recovery Predictor")
    st.write(
        "Enter payment and customer information to generate "
        "an AI-powered recovery prediction."
    )

    st.divider()

    left, right = st.columns(2)

    # =====================================================
    # PAYMENT INFORMATION
    # =====================================================
    with left:
        st.subheader("💳 Payment Information")

        amount = st.number_input(
            "Payment Amount (₹)",
            min_value=1.0,
            value=1000.0
        )

        payment_method = st.selectbox(
            "Payment Method",
            ["UPI", "Card", "Net Banking", "Wallet"]
        )

        failure_reason = st.selectbox(
            "Failure Reason",
            [
                "Insufficient Funds",
                "Bank Decline",
                "Network Error",
                "Expired Card",
                "Limit Exceeded",
                "Authentication Failed"
            ]
        )

        subscription = st.selectbox(
            "Subscription Payment?",
            ["Yes", "No"]
        )

    # =====================================================
    # CUSTOMER INFORMATION
    # =====================================================
    with right:
        st.subheader("👤 Customer Information")

        customer_age_days = st.number_input(
            "Customer Age (days)",
            min_value=1,
            value=500
        )

        previous_successful_payments = st.number_input(
            "Previous Successful Payments",
            min_value=0,
            value=20
        )

        previous_failed_payments = st.number_input(
            "Previous Failed Payments",
            min_value=0,
            value=4
        )

        retry_count = st.number_input(
            "Retry Count",
            min_value=0,
            value=1
        )

        time_since_failure_hours = st.number_input(
            "Hours Since Failure",
            min_value=0.1,
            value=2.0
        )

        customer_avg_amount = st.number_input(
            "Customer's Average Payment Amount (₹)",
            min_value=1.0,
            value=1000.0
        )

    st.divider()

    predict_button = st.button(
        "🚀 Predict Recovery",
        use_container_width=True,
        type="primary"
    )

    if predict_button:

        if not model_loaded:
            st.error("ML model is not available.")

        else:

            input_data = pd.DataFrame(
                [
                    {
                        "amount": amount,
                        "payment_method": payment_method,
                        "failure_reason": failure_reason,
                        "customer_age_days": customer_age_days,
                        "previous_successful_payments":
                            previous_successful_payments,
                        "previous_failed_payments":
                            previous_failed_payments,
                        "retry_count": retry_count,
                        "time_since_failure_hours":
                            time_since_failure_hours,
                        "customer_avg_amount":
                            customer_avg_amount,
                        "is_subscription":
                            1 if subscription == "Yes" else 0
                    }
                ]
            )

            input_data = create_features(input_data)

            # =================================================
            # PREDICTION
            # =================================================
            try:
                probability = model.predict_proba(
                    input_data
                )[0][1]
            except Exception as e:
                st.error(
                    "Prediction failed. Please check that the input "
                    "features match the trained model."
                )
                st.exception(e)
                st.stop()

            percentage = probability * 100

            policy = recovery_agent_policy(probability, retry_count)
            priority = policy["priority"]
            action = policy["action"]
            retry_allowed = policy["retry_allowed"]
            stopping_reason = policy["policy_reason"]

            expected_recovery = amount * probability

            # =================================================
            # SAVE HISTORY
            # =================================================
            new_prediction = {
                "Time": datetime.now().strftime(
                    "%d-%m-%Y %H:%M"
                ),
                "Amount": amount,
                "Payment Method": payment_method,
                "Failure Reason": failure_reason,
                "Recovery Probability": round(
                    percentage, 2
                ),
                "Expected Recovery": round(
                    expected_recovery, 2
                ),
                "Priority": priority,
                "Recommended Action": action,
                "Retry Allowed": (
                    "Yes" if retry_allowed else "No"
                ),
                "Policy Reason": stopping_reason
            }

            st.session_state.history.append(new_prediction)

            pd.DataFrame(
                st.session_state.history
            ).to_csv(
                HISTORY_FILE,
                index=False
            )

            # =================================================
            # RESULT
            # =================================================
            st.divider()
            st.header("📊 PayRecover AI Result")

            r1, r2, r3, r4 = st.columns(4)

            with r1:
                st.metric(
                    "Recovery Probability",
                    f"{percentage:.2f}%"
                )

            with r2:
                st.metric(
                    "Priority",
                    priority
                )

            with r3:
                st.metric(
                    "Expected Recovery",
                    f"₹{expected_recovery:,.2f}"
                )

            with r4:
                st.metric(
                    "Retry Allowed",
                    "YES" if retry_allowed else "NO"
                )

            st.progress(
                min(probability, 1.0),
                text=f"Recovery Probability: {percentage:.2f}%"
            )

            if priority == "HIGH":
                st.success(f"🔴 HIGH PRIORITY — {action}")
            elif priority == "MEDIUM":
                st.warning(f"🟡 MEDIUM PRIORITY — {action}")
            else:
                st.error(f"🟢 LOW PRIORITY — {action}")

            st.info(
                f"🧾 **Policy Reason:** {stopping_reason}"
            )

            # =================================================
            # EXPLAINABLE AI
            # =================================================
            st.divider()
            st.subheader(
                "🔍 Why did PayRecover recommend this?"
            )

            reasons = []

            if previous_successful_payments > previous_failed_payments:
                reasons.append(
                    "✅ Strong customer payment history"
                )
            else:
                reasons.append(
                    "⚠️ Customer has relatively high payment failures"
                )

            if retry_count == 0:
                reasons.append(
                    "✅ No previous retry attempt"
                )
            elif retry_count == 1:
                reasons.append(
                    "✅ Low retry pressure"
                )
            elif retry_count == 2:
                reasons.append(
                    "⚠️ Moderate retry pressure"
                )
            else:
                reasons.append(
                    "🛑 Retry limit has been reached"
                )

            amount_difference = abs(
                amount - customer_avg_amount
            )

            if amount_difference <= customer_avg_amount * 0.20:
                reasons.append(
                    "✅ Payment amount is close to customer's normal amount"
                )
            else:
                reasons.append(
                    "⚠️ Payment amount differs significantly from customer's average"
                )

            if probability >= 0.70:
                reasons.append(
                    "🚀 High predicted recovery probability"
                )
            elif probability >= 0.35:
                reasons.append(
                    "🔄 Moderate predicted recovery probability"
                )
            else:
                reasons.append(
                    "🛑 Low predicted recovery probability"
                )

            for reason in reasons:
                st.write(reason)

            st.caption(
                "These signals explain the operational recommendation. "
                "The final probability is generated by the trained ML model."
            )

            # =================================================
            # BUSINESS IMPACT
            # =================================================
            st.divider()
            st.subheader("💰 Business Impact")

            st.info(
                f"For a failed payment of **₹{amount:,.2f}**, "
                f"the model estimates approximately "
                f"**₹{expected_recovery:,.2f}** as expected recoverable value."
            )

            # =================================================
            # DECISION ENGINE
            # =================================================
            st.divider()
            st.subheader("🧠 AI Decision Engine")

            st.write(
                "PayRecover AI evaluates customer behaviour, payment "
                "history, retry activity and transaction amount to "
                "determine the most suitable recovery action."
            )

            d1, d2, d3 = st.columns(3)

            with d1:
                st.subheader("👤 Customer Behaviour")
                if previous_successful_payments > previous_failed_payments:
                    st.success("Positive payment history")
                else:
                    st.warning("Higher payment failure history")

                st.caption(
                    f"{previous_successful_payments} successful "
                    f"vs {previous_failed_payments} failed payments."
                )

            with d2:
                st.subheader("🔄 Retry Analysis")

                if retry_count == 0:
                    st.success("No previous retry")
                elif retry_count == 1:
                    st.success("Low retry pressure")
                elif retry_count == 2:
                    st.warning("Moderate retry pressure")
                else:
                    st.error("Retry limit reached")

                st.caption(
                    f"Current retry count: {retry_count}"
                )

            with d3:
                st.subheader("💰 Transaction Analysis")

                difference = abs(
                    amount - customer_avg_amount
                )

                if difference <= customer_avg_amount * 0.20:
                    st.success("Amount is within normal range")
                else:
                    st.warning("Amount differs from customer average")

                st.caption(
                    f"Current: ₹{amount:,.0f} | "
                    f"Average: ₹{customer_avg_amount:,.0f}"
                )

            # =================================================
            # FINAL DECISION
            # =================================================
            st.divider()
            st.subheader("🎯 AI Decision")

            if retry_count >= 3:
                st.error(
                    "🛑 **STOP RETRY** — Maximum retry limit reached."
                )
            elif probability < 0.35:
                st.error(
                    "🛑 **DO NOT RETRY** — Recovery probability "
                    "is below the automatic retry threshold."
                )
            elif probability >= 0.70:
                st.success(
                    "🚀 **RETRY PAYMENT** — High recovery probability "
                    "makes an immediate retry appropriate."
                )
            else:
                st.warning(
                    "🔄 **USE ALTERNATIVE PAYMENT METHOD** — "
                    "Recovery probability is moderate."
                )

            st.info(
                f"💡 **Final AI Recommendation:** {action}"
            )

# =========================================================
# ANALYTICS
# =========================================================
elif page == "📊 Analytics":

    st.title("📊 Recovery Analytics")

    st.write(
        "Analyze prediction performance, recovery probability, "
        "payment behaviour and potential revenue recovery."
    )

    st.divider()

    if len(st.session_state.history) == 0:
        st.info(
            "Make some predictions first to generate live analytics."
        )

    else:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        for col in [
            "Recovery Probability",
            "Expected Recovery",
            "Amount"
        ]:
            if col in history_df.columns:
                history_df[col] = pd.to_numeric(
                    history_df[col],
                    errors="coerce"
                ).fillna(0)

        total_predictions = len(history_df)

        average_probability = (
            history_df["Recovery Probability"].mean()
            if "Recovery Probability" in history_df.columns
            else 0
        )

        total_expected_recovery = (
            history_df["Expected Recovery"].sum()
            if "Expected Recovery" in history_df.columns
            else 0
        )

        total_transaction_value = (
            history_df["Amount"].sum()
            if "Amount" in history_df.columns
            else 0
        )

        high_count = (
            (history_df["Priority"] == "HIGH").sum()
            if "Priority" in history_df.columns
            else 0
        )

        medium_count = (
            (history_df["Priority"] == "MEDIUM").sum()
            if "Priority" in history_df.columns
            else 0
        )

        low_count = (
            (history_df["Priority"] == "LOW").sum()
            if "Priority" in history_df.columns
            else 0
        )

        a1, a2, a3, a4 = st.columns(4)

        with a1:
            st.metric("📊 Predictions", total_predictions)

        with a2:
            st.metric(
                "📈 Average Recovery",
                f"{average_probability:.2f}%"
            )

        with a3:
            st.metric(
                "💰 Expected Recovery",
                f"₹{total_expected_recovery:,.0f}"
            )

        with a4:
            st.metric(
                "🔴 High Priority",
                high_count
            )

        st.divider()

        # =================================================
        # TREND
        # =================================================
        st.header("📈 Recovery Probability Trend")

        trend = history_df[
            ["Recovery Probability"]
        ].copy()

        trend["Prediction"] = range(
            1, len(trend) + 1
        )

        st.line_chart(
            trend.set_index("Prediction")
        )

        st.divider()

        # =================================================
        # PRIORITY
        # =================================================
        st.header("🎯 Recovery Priority Distribution")

        priority_df = pd.DataFrame(
            {
                "Priority": [
                    "HIGH",
                    "MEDIUM",
                    "LOW"
                ],
                "Predictions": [
                    high_count,
                    medium_count,
                    low_count
                ]
            }
        )

        st.bar_chart(
            priority_df.set_index("Priority")
        )

        st.divider()

        # =================================================
        # RECOVERY BY PRIORITY
        # =================================================
        st.header("💰 Expected Recovery by Priority")

        recovery_priority_df = (
            history_df
            .groupby("Priority")["Expected Recovery"]
            .sum()
            .reindex(
                ["HIGH", "MEDIUM", "LOW"],
                fill_value=0
            )
        )

        st.bar_chart(recovery_priority_df)

        st.divider()

        # =================================================
        # PAYMENT METHODS
        # =================================================
        st.header("💳 Payment Method Analysis")

        if "Payment Method" in history_df.columns:
            payment_counts = (
                history_df["Payment Method"].value_counts()
            )
            st.bar_chart(payment_counts)

        st.divider()

        # =================================================
        # FAILURE REASONS
        # =================================================
        st.header("⚠️ Failure Reason Analysis")

        if "Failure Reason" in history_df.columns:
            failure_counts = (
                history_df["Failure Reason"].value_counts()
            )
            st.bar_chart(failure_counts)

        st.divider()

        # =================================================
        # BUSINESS INSIGHT
        # =================================================
        st.header("💼 Business Insight")

        if average_probability >= 70:
            st.success(
                "🚀 Strong recovery opportunity: current predictions "
                "show high average recovery potential."
            )
        elif average_probability >= 35:
            st.warning(
                "🔄 Moderate recovery opportunity: alternative payment "
                "methods can be considered for medium-probability cases."
            )
        else:
            st.error(
                "🛑 Low recovery opportunity: repeated retries should "
                "generally be avoided."
            )

        st.info(
            f"Across **{total_predictions} predictions**, PayRecover AI "
            f"estimates approximately **₹{total_expected_recovery:,.2f}** "
            f"in expected recoverable value from **₹{total_transaction_value:,.2f}** "
            f"of analyzed payment value."
        )

# =========================================================
# PREDICTION HISTORY
# =========================================================
elif page == "📋 Prediction History":

    st.title("📋 Prediction History")

    if len(st.session_state.history) == 0:
        st.info("No predictions available yet.")

    else:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        st.metric(
            "Total Predictions",
            len(history_df)
        )

        st.divider()

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        # =================================================
        # DOWNLOAD
        # =================================================
        csv_data = (
            history_df
            .to_csv(index=False)
            .encode("utf-8")
        )

        st.download_button(
            "⬇️ Download Prediction History",
            data=csv_data,
            file_name="payrecover_prediction_history.csv",
            mime="text/csv"
        )

        st.divider()

        # =================================================
        # CLEAR HISTORY
        # =================================================
        if st.button("🗑️ Clear Prediction History"):

            st.session_state.history = []

            empty_history = pd.DataFrame(
                columns=[
                    "Time",
                    "Amount",
                    "Payment Method",
                    "Failure Reason",
                    "Recovery Probability",
                    "Expected Recovery",
                    "Priority",
                    "Recommended Action",
                    "Retry Allowed",
                    "Policy Reason"
                ]
            )

            empty_history.to_csv(
                HISTORY_FILE,
                index=False
            )

            st.success("Prediction history cleared.")
            st.rerun()
