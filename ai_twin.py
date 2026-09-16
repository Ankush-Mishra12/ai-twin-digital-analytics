
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import textwrap
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="AI Twin Digital Analytics",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# STYLE
# ============================================================
css = """
<style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    .main-title {
        font-size: 2.4rem;
        font-weight: 750;
        letter-spacing: -0.5px;
        margin-bottom: 0.2rem;
    }

    .sub-title {
        color: #64748b;
        font-size: 1rem;
        margin-bottom: 1.2rem;
    }

    .section-title {
        font-size: 1.55rem;
        font-weight: 700;
        margin-top: 0.7rem;
        margin-bottom: 0.7rem;
    }

    .kpi-label {
        color: #64748b;
        font-size: 0.85rem;
        font-weight: 600;
    }

    .kpi-value {
        font-size: 1.75rem;
        font-weight: 750;
        margin-top: 0.1rem;
    }

    .kpi-card {
        padding: 1rem 1.1rem;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        background: rgba(248, 250, 252, 0.92);
        min-height: 105px;
    }

    .insight-box {
        padding: 0.9rem 1rem;
        border-radius: 12px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        margin: 0.5rem 0 1rem 0;
    }

    div[data-testid="stMetric"] {
        border: 1px solid #e2e8f0;
        padding: 0.8rem;
        border-radius: 12px;
        background: #f8fafc;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid #e2e8f0;
    }
</style>
"""
st.markdown(css, unsafe_allow_html=True)


# ============================================================
# LOGIN / APP ACCESS
# ============================================================
# Simple project login. Keep the GitHub repository PRIVATE because
# the credentials are intentionally stored in this file.
APP_USERNAME = "Ankush"
APP_PASSWORD = "ankush@999"

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    login_html = textwrap.dedent(
        """
        <style>
            .login-wrap {
                min-height: 72vh;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .login-card {
                width: min(520px, 92vw);
                padding: 2.2rem 2.3rem;
                border: 1px solid #dbeafe;
                border-radius: 22px;
                background: rgba(255,255,255,0.96);
                box-shadow: 0 18px 50px rgba(15,23,42,0.10);
                text-align: center;
                animation: loginEntrance 0.75s ease-out both;
            }
            .login-icon {
                font-size: 3.2rem;
                margin-bottom: 0.5rem;
                animation: pulseIcon 1.8s ease-in-out infinite;
            }
            .login-title {
                font-size: 2rem;
                font-weight: 750;
                color: #0f172a;
                margin-bottom: 0.25rem;
            }
            .login-subtitle {
                color: #64748b;
                font-size: 0.98rem;
                margin-bottom: 1rem;
            }
            @keyframes loginEntrance {
                from { opacity: 0; transform: translateY(22px) scale(0.98); }
                to { opacity: 1; transform: translateY(0) scale(1); }
            }
            @keyframes pulseIcon {
                0%, 100% { transform: scale(1); opacity: 0.95; }
                50% { transform: scale(1.08); opacity: 1; }
            }
        </style>

        <div class="login-wrap">
            <div class="login-card">
                <div class="login-icon">🩺</div>
                <div class="login-title">AI Twin Digital Analytics</div>
                <div class="login-subtitle">Secure project access</div>
            </div>
        </div>
        """
    )
    st.markdown(login_html, unsafe_allow_html=True)

    with st.form("login_form"):
        st.markdown("### Welcome back")
        username = st.text_input("User ID", placeholder="Enter your user ID")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        submit = st.form_submit_button("🔐 Enter Dashboard", use_container_width=True)

    if submit:
        if username == APP_USERNAME and password == APP_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Invalid User ID or Password.")

    st.stop()


# ============================================================
# DATA
# ============================================================
DATA_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "AI_twin_cleaned_folder")

FILES = [
    "ai_digital_twin_assessment",
    "daily_activity_log",
    "digital_twin_profile",
    "environmental_data",
    "future_health_predictions",
    "genomic_risk_profile",
    "healthcare_visits",
    "laboratory_test_results",
    "medical_history",
    "mental_health_timeline",
    "nutrition_log",
    "sleep_history",
    "smart_home_iot",
    "wearable_sensor_data",
]


@st.cache_data(show_spinner="Loading AI Digital Twin data...")
def load_data():
    data = {}
    errors = []

    for name in FILES:
        path = f"{DATA_FOLDER}/{name}.csv"
        try:
            df = pd.read_csv(path)
            df.columns = [str(c).strip().upper() for c in df.columns]
            data[name] = df
        except Exception as e:
            errors.append(f"{name}.csv -> {e}")

    return data, errors


data, load_errors = load_data()

if load_errors:
    st.error("Some files could not be loaded.")
    for err in load_errors:
        st.write(err)
    st.stop()

(
    ai_assessment,
    daily_activity,
    profile,
    environmental,
    future_predictions,
    genetic,
    visits,
    labs,
    medical,
    mental,
    nutrition,
    sleep,
    smart_home,
    wearable,
) = [data[n] for n in FILES]


# ============================================================
# HELPERS
# ============================================================
def pct(condition):
    s = pd.Series(condition).dropna()
    return float(s.mean() * 100) if len(s) else np.nan


def mean_or_nan(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.mean()) if len(s) else np.nan


def fmt_pct(x):
    return "—" if pd.isna(x) else f"{x:.1f}%"


def fmt_num(x, decimals=1):
    return "—" if pd.isna(x) else f"{x:,.{decimals}f}"


def kpi_cards(items):
    """Render KPI values inside native Streamlit bordered cards."""
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        with col:
            with st.container(border=True):
                st.markdown(f'<div class="kpi-label">{label}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="kpi-value">{value}</div>', unsafe_allow_html=True)


def section(title, subtitle=None):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.caption(subtitle)


def safe_bar(df, x, y, title, color=None, orientation=None, text=None):
    if df.empty:
        st.info("No data available for this chart.")
        return

    fig = px.bar(
        df,
        x=x,
        y=y,
        color=color,
        orientation=orientation,
        text=text,
        title=title,
    )
    fig.update_layout(height=390, margin=dict(l=20, r=20, t=55, b=20))
    st.plotly_chart(fig, use_container_width=True)


def safe_line(df, x, y, title, color=None):
    if df.empty:
        st.info("No data available for this chart.")
        return

    fig = px.line(
        df,
        x=x,
        y=y,
        color=color,
        markers=True,
        title=title,
    )
    fig.update_layout(height=390, margin=dict(l=20, r=20, t=55, b=20))
    st.plotly_chart(fig, use_container_width=True)


def safe_donut(df, names, values, title):
    if df.empty:
        st.info("No data available for this chart.")
        return

    fig = px.pie(df, names=names, values=values, hole=0.55, title=title)
    fig.update_layout(height=390, margin=dict(l=20, r=20, t=55, b=20))
    st.plotly_chart(fig, use_container_width=True)


def safe_100_stack(df, x, y, title, orientation=None):
    if df.empty:
        st.info("No data available for this chart.")
        return

    fig = px.bar(
        df,
        x=x,
        y=y,
        color="Category",
        barmode="relative",
        barnorm="percent",
        text_auto=".1f",
        orientation=orientation,
        title=title,
    )
    fig.update_layout(height=410, margin=dict(l=20, r=20, t=55, b=20))
    st.plotly_chart(fig, use_container_width=True)


def chart_data(cross, index_name, category_name="Category", value_name="Value"):
    out = cross.reset_index().melt(
        id_vars=[index_name],
        var_name=category_name,
        value_name=value_name
    )
    return out


# ============================================================
# COMMON MERGES
# ============================================================
# IMPORTANT: keep the actual analysis columns after merge.
# Earlier version merged only DIGITAL_ID, which caused KeyError
# when the analysis tried to use DIABETES_STATUS / risk fields.

risk_base = medical.merge(
    future_predictions,
    on="DIGITAL_ID",
    how="inner",
    suffixes=("", "_FUTURE")
)

genetic_priority = genetic.merge(
    future_predictions,
    on="DIGITAL_ID",
    how="inner",
    suffixes=("", "_FUTURE")
)

wellness_audit = ai_assessment.merge(
    medical,
    on="DIGITAL_ID",
    how="inner",
    suffixes=("", "_MEDICAL")
)

lab_ai = labs.merge(
    ai_assessment,
    on="DIGITAL_ID",
    how="inner",
    suffixes=("", "_AI")
)

profile_medical = profile.merge(
    medical,
    on="DIGITAL_ID",
    how="inner",
    suffixes=("", "_MEDICAL")
)

profile_future = profile.merge(
    future_predictions,
    on="DIGITAL_ID",
    how="inner",
    suffixes=("", "_FUTURE")
)


# ============================================================
# SIDEBAR MAIN NAVIGATION
# ============================================================
st.sidebar.title("🩺 AI Twin Analytics")
st.sidebar.caption("Health analytics & digital twin decision audit")

main_page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "🐍 Python Analysis", "📊 Power BI Analysis"],
    key="main_nav"
)


# ============================================================
# HOME
# ============================================================
if main_page == "🏠 Home":

    welcome_html = textwrap.dedent(
        """
        <style>
            .welcome-wrap {
                text-align: center;
                padding: 35px 10px 30px 10px;
            }

            .welcome-word {
                font-size: 2.4rem;
                font-weight: 750;
                margin: 6px 0;
                color: #64748b;
                opacity: 0.35;
                animation: wordHighlight 6s infinite;
            }

            .welcome-word:nth-child(1) { animation-delay: 0s; }
            .welcome-word:nth-child(2) { animation-delay: 1s; }
            .welcome-word:nth-child(3) { animation-delay: 2s; }
            .welcome-word:nth-child(4) { animation-delay: 3s; }
            .welcome-word:nth-child(5) { animation-delay: 4s; }
            .welcome-word:nth-child(6) { animation-delay: 5s; }

            @keyframes wordHighlight {
                0%, 100% {
                    color: #64748b;
                    opacity: 0.35;
                    transform: scale(1);
                }
                8%, 18% {
                    color: #ffffff;
                    opacity: 1;
                    transform: scale(1.08);
                    text-shadow: 0 0 18px rgba(255, 255, 255, 0.45);
                }
                25%, 100% {
                    color: #64748b;
                    opacity: 0.35;
                    transform: scale(1);
                }
            }
        </style>

        <div class="welcome-wrap">
            <div class="welcome-word">Welcome</div>
            <div class="welcome-word">to</div>
            <div class="welcome-word">AI</div>
            <div class="welcome-word">Twin</div>
            <div class="welcome-word">Digital</div>
            <div class="welcome-word">Analytics</div>
        </div>
        """
    )
    st.markdown(welcome_html, unsafe_allow_html=True)

    section("KPIs", "Current population and decision-layer indicators")

    total_patients = medical["DIGITAL_ID"].nunique() if "DIGITAL_ID" in medical else len(profile)
    high_physician = (
        (medical["PHYSICIAN_RISK_LEVEL"].astype(str).str.upper() == "HIGH").sum()
        if "PHYSICIAN_RISK_LEVEL" in medical else np.nan
    )
    critical_labs = (
        (labs["LAB_STATUS"].astype(str).str.upper() == "CRITICAL").sum()
        if "LAB_STATUS" in labs else np.nan
    )
    abnormal_blood = (
        (medical["BLOOD_TEST_STATUS"].astype(str).str.upper() == "ABNORMAL").sum()
        if "BLOOD_TEST_STATUS" in medical else np.nan
    )

    kpi_cards([
        ("Total Patients", f"{int(total_patients):,}"),
        ("High Physician Risk", f"{int(high_physician):,}" if not pd.isna(high_physician) else "—"),
        ("Critical Lab Records", f"{int(critical_labs):,}" if not pd.isna(critical_labs) else "—"),
        ("Abnormal Blood Tests", f"{int(abnormal_blood):,}" if not pd.isna(abnormal_blood) else "—"),
    ])

    st.divider()

    section("Project Coverage")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("🐍 **Python Analysis**\n\nDescriptive, diagnostic and predictive analysis.")
    with col2:
        st.info("📊 **Power BI Analysis**\n\nPopulation, wellness audit and risk translation.")
    with col3:
        st.info("🧠 **Decision Focus**\n\nCheck whether AI risk signals reach the final patient/physician output.")

    st.divider()
    st.markdown(
        '<div class="insight-box"><b>Project focus:</b> The app combines patient profile, medical history, genetic risk, labs, lifestyle and AI outputs to audit the digital twin from population-level KPIs through predictive modelling.</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# PYTHON ANALYSIS
# ============================================================
elif main_page == "🐍 Python Analysis":

    st.sidebar.markdown("---")
    python_bp = st.sidebar.radio(
        "Business Problem",
        [
            "Business Problem 1 — Risk Translation Gap",
            "Business Problem 2 — Recommendation & Adherence",
            "Predictive Analysis",
        ],
        key="python_bp"
    )

    if python_bp != "Predictive Analysis":
        analysis_type = st.sidebar.radio(
            "Analysis Type",
            ["Descriptive", "Diagnostic"],
            key=f"analysis_type_{python_bp}"
        )
    else:
        analysis_type = None

    # --------------------------------------------------------
    # BP1
    # --------------------------------------------------------
    if python_bp == "Business Problem 1 — Risk Translation Gap":

        st.title("Business Problem 1")
        st.caption("AI Risk-Score-to-Decision Translation Gap")

        section("KPIs")

        diagnosed_pct = np.nan
        diagnosed_avg_risk = np.nan
        diabetic_excellent = np.nan
        high_genetic_low = np.nan

        if {"DIABETES_STATUS", "ONE_YEAR_DIABETES_RISK"}.issubset(risk_base.columns):
            diagnosed = risk_base[
                risk_base["DIABETES_STATUS"].astype(str).str.upper() == "YES"
            ]
            diagnosed_pct = pct(
                risk_base["DIABETES_STATUS"].astype(str).str.upper() == "YES"
            )
            diagnosed_avg_risk = mean_or_nan(diagnosed["ONE_YEAR_DIABETES_RISK"])

        if {"DIABETES_STATUS", "AI_FUTURE_HEALTH_FORECAST"}.issubset(risk_base.columns):
            diabetic = risk_base[
                risk_base["DIABETES_STATUS"].astype(str).str.upper() == "YES"
            ]
            diabetic_excellent = pct(
                diabetic["AI_FUTURE_HEALTH_FORECAST"].astype(str).str.upper() == "EXCELLENT"
            )

        if {"DIABETES_GENETIC_RISK", "PREVENTIVE_CARE_PRIORITY"}.issubset(genetic_priority.columns):
            high_g = genetic_priority[
                genetic_priority["DIABETES_GENETIC_RISK"].astype(str).str.upper() == "HIGH"
            ]
            high_genetic_low = pct(
                high_g["PREVENTIVE_CARE_PRIORITY"].astype(str).str.upper() == "LOW"
            )

        kpi_cards([
            ("Diagnosed Diabetes Patients", fmt_pct(diagnosed_pct)),
            ("Avg Diabetes Risk — Diagnosed", fmt_num(diagnosed_avg_risk, 1) + "%"),
            ("Diabetic Patients with Excellent Forecast", fmt_pct(diabetic_excellent)),
            ("High Genetic Risk → Low Priority", fmt_pct(high_genetic_low)),
        ])

        section("Charts & Analysis")

        if analysis_type == "Descriptive":

            # Q1
            q1 = (
                risk_base.groupby("DIABETES_STATUS", dropna=False)["ONE_YEAR_DIABETES_RISK"]
                .mean()
                .reset_index(name="Average Risk")
            )
            col1, col2 = st.columns(2)
            with col1:
                safe_bar(q1, "DIABETES_STATUS", "Average Risk",
                         "Q1. Average One-Year Diabetes Risk by Diabetes Status")
            with col2:
                st.dataframe(q1.round(2), use_container_width=True, hide_index=True)

            # Q2
            chronic = risk_base.copy()
            if "CHRONIC_DISEASE" in chronic:
                chronic["HAS_CHRONIC_DISEASE"] = (
                    chronic["CHRONIC_DISEASE"].notna() &
                    (chronic["CHRONIC_DISEASE"].astype(str).str.strip().str.lower() != "none")
                )
                q2 = (
                    chronic.groupby("HAS_CHRONIC_DISEASE")["AI_FUTURE_HEALTH_FORECAST"]
                    .value_counts(normalize=True)
                    .unstack(fill_value=0)
                    .reset_index()
                )
                q2_melt = q2.melt("HAS_CHRONIC_DISEASE", var_name="Forecast", value_name="Percentage")
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.bar(
                        q2_melt,
                        x="HAS_CHRONIC_DISEASE",
                        y="Percentage",
                        color="Forecast",
                        barmode="stack",
                        title="Q2. Forecast Distribution by Chronic Disease Status",
                        text_auto=".1f",
                    )
                    fig.update_layout(height=390)
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    st.dataframe(q2.round(2), use_container_width=True, hide_index=True)

            # Q3
            if "DIABETES_GENETIC_RISK" in genetic_priority and "PREVENTIVE_CARE_PRIORITY" in genetic_priority:
                q3 = pd.crosstab(
                    genetic_priority["DIABETES_GENETIC_RISK"],
                    genetic_priority["PREVENTIVE_CARE_PRIORITY"],
                    normalize="index"
                ).mul(100)
                q3_melt = q3.reset_index().melt(
                    "DIABETES_GENETIC_RISK",
                    var_name="Priority",
                    value_name="Percentage"
                )
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.bar(
                        q3_melt,
                        x="DIABETES_GENETIC_RISK",
                        y="Percentage",
                        color="Priority",
                        barmode="stack",
                        title="Q3. Diabetes Genetic Risk vs Preventive Care Priority",
                        text_auto=".1f",
                    )
                    fig.update_layout(height=390)
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    st.dataframe(q3.round(2), use_container_width=True, hide_index=False)

            # Q5
            scores = risk_base[
                ["PHYSICIAN_RISK_LEVEL", "ONE_YEAR_DIABETES_RISK",
                 "FIVE_YEAR_CARDIOVASCULAR_RISK", "STROKE_RISK"]
            ].copy()
            q5 = scores.groupby("PHYSICIAN_RISK_LEVEL")[
                ["ONE_YEAR_DIABETES_RISK", "FIVE_YEAR_CARDIOVASCULAR_RISK", "STROKE_RISK"]
            ].mean().round(2)
            q5_melt = q5.reset_index().melt(
                "PHYSICIAN_RISK_LEVEL",
                var_name="Risk Score",
                value_name="Average"
            )
            col1, col2 = st.columns(2)
            with col1:
                fig = px.bar(
                    q5_melt, x="PHYSICIAN_RISK_LEVEL", y="Average",
                    color="Risk Score", barmode="group",
                    title="Q5. Numeric Risk Scores by Physician Risk Level"
                )
                fig.update_layout(height=390)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.dataframe(q5, use_container_width=True)

            # Q6
            q6 = medical[["DIGITAL_ID", "DIABETES_STATUS", "HYPERTENSION_STATUS"]].merge(
                labs[["DIGITAL_ID", "LAB_STATUS"]],
                on="DIGITAL_ID",
                how="inner"
            ).merge(
                genetic[["DIGITAL_ID", "DIABETES_GENETIC_RISK", "HEART_DISEASE_GENETIC_RISK"]],
                on="DIGITAL_ID",
                how="inner"
            ).merge(
                ai_assessment[["DIGITAL_ID", "OVERALL_AI_STATUS"]],
                on="DIGITAL_ID",
                how="inner"
            )
            q6["OBJECTIVELY_HIGH_RISK"] = (
                (q6["DIABETES_STATUS"].astype(str).str.upper() == "YES") |
                (q6["HYPERTENSION_STATUS"].astype(str).str.upper() == "YES") |
                (q6["LAB_STATUS"].astype(str).str.upper() == "CRITICAL") |
                (q6["DIABETES_GENETIC_RISK"].astype(str).str.upper() == "HIGH") |
                (q6["HEART_DISEASE_GENETIC_RISK"].astype(str).str.upper() == "HIGH")
            )
            high_group = q6[q6["OBJECTIVELY_HIGH_RISK"]]
            miss_pct = pct(high_group["OVERALL_AI_STATUS"].astype(str).str.upper() == "EXCELLENT")
            q6_summary = pd.DataFrame({
                "Metric": ["Objectively High-Risk Patients", "High-Risk patients with Excellent AI Output"],
                "Value": [len(high_group), f"{miss_pct:.2f}%"]
            })
            col1, col2 = st.columns(2)
            with col1:
                kpi_cards([
                    ("Objectively High-Risk", f"{len(high_group):,}"),
                    ("Excellent Final Output", fmt_pct(miss_pct))
                ])
            with col2:
                st.dataframe(q6_summary, use_container_width=True, hide_index=True)

        else:
            # Diagnostic Q1
            q1 = risk_base[
                ["AI_FUTURE_HEALTH_FORECAST", "ONE_YEAR_DIABETES_RISK"]
            ].groupby("AI_FUTURE_HEALTH_FORECAST")["ONE_YEAR_DIABETES_RISK"] \
             .agg(["count", "min", "max", "mean"]).round(2).reset_index()

            col1, col2 = st.columns(2)
            with col1:
                safe_bar(q1, "AI_FUTURE_HEALTH_FORECAST", "mean",
                         "Q1. Numeric Risk Distribution by AI Forecast")
            with col2:
                st.dataframe(q1, use_container_width=True, hide_index=True)

            # Diagnostic Q2
            q2a = pd.crosstab(
                genetic_priority["DIABETES_GENETIC_RISK"],
                genetic_priority["PREVENTIVE_CARE_PRIORITY"],
                normalize="index"
            ).mul(100)
            q2b = pd.crosstab(
                genetic_priority["HEART_DISEASE_GENETIC_RISK"],
                genetic_priority["PREVENTIVE_CARE_PRIORITY"],
                normalize="index"
            ).mul(100)

            col1, col2 = st.columns(2)
            with col1:
                q2a_melt = q2a.reset_index().melt(
                    "DIABETES_GENETIC_RISK",
                    var_name="Priority", value_name="Percentage"
                )
                fig = px.bar(
                    q2a_melt, x="DIABETES_GENETIC_RISK", y="Percentage",
                    color="Priority", barmode="stack",
                    title="Q2. Diabetes Genetic Risk → Preventive Priority",
                    text_auto=".1f"
                )
                fig.update_layout(height=390)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                q2b_melt = q2b.reset_index().melt(
                    "HEART_DISEASE_GENETIC_RISK",
                    var_name="Priority", value_name="Percentage"
                )
                fig = px.bar(
                    q2b_melt, x="HEART_DISEASE_GENETIC_RISK", y="Percentage",
                    color="Priority", barmode="stack",
                    title="Q2. Heart Genetic Risk → Preventive Priority",
                    text_auto=".1f"
                )
                fig.update_layout(height=390)
                st.plotly_chart(fig, use_container_width=True)

            # Diagnostic Q3
            age_data = profile_medical.merge(
                future_predictions[["DIGITAL_ID", "AI_FUTURE_HEALTH_FORECAST"]],
                on="DIGITAL_ID",
                how="inner"
            )
            age_data["AGE_GROUP"] = pd.cut(
                pd.to_numeric(age_data["AGE"], errors="coerce"),
                bins=[0, 30, 45, 60, 100],
                labels=["<30", "30-45", "45-60", "60+"]
            )
            diabetic = age_data[
                age_data["DIABETES_STATUS"].astype(str).str.upper() == "YES"
            ].copy()
            diabetic["MISMATCH"] = (
                diabetic["AI_FUTURE_HEALTH_FORECAST"].astype(str).str.upper() == "EXCELLENT"
            )
            q3 = diabetic.groupby("AGE_GROUP", observed=True)["MISMATCH"].mean().mul(100).reset_index(name="Mismatch Rate")
            safe_bar(q3, "AGE_GROUP", "Mismatch Rate",
                     "Q3. AI Forecast Mismatch Rate by Age Group")

            # Diagnostic Q4
            true_risk = risk_base[
                ["DIGITAL_ID", "DIABETES_STATUS", "HYPERTENSION_STATUS",
                 "ONE_YEAR_DIABETES_RISK", "AI_FUTURE_HEALTH_FORECAST"]
            ].copy()
            true_risk["GROUND_TRUTH_HIGH_RISK"] = (
                (true_risk["DIABETES_STATUS"].astype(str).str.upper() == "YES") |
                (true_risk["HYPERTENSION_STATUS"].astype(str).str.upper() == "YES") |
                (pd.to_numeric(true_risk["ONE_YEAR_DIABETES_RISK"], errors="coerce") >= 50)
            )
            total_true = true_risk["GROUND_TRUTH_HIGH_RISK"].sum()
            missed = true_risk[
                true_risk["GROUND_TRUTH_HIGH_RISK"] &
                (true_risk["AI_FUTURE_HEALTH_FORECAST"].astype(str).str.upper() == "EXCELLENT")
            ]
            miss_q4 = pct(
                missed["GROUND_TRUTH_HIGH_RISK"]
            )
            kpi_cards([
                ("Genuinely High-Risk Patients", f"{int(total_true):,}"),
                ("Shown as Excellent", f"{len(missed):,}"),
                ("Potentially Missed", fmt_pct(miss_q4)),
            ])

            st.dataframe(
                pd.DataFrame({
                    "Metric": [
                        "Total genuinely high-risk patients",
                        "High-risk patients shown as Excellent",
                        "Missed percentage"
                    ],
                    "Value": [
                        int(total_true),
                        len(missed),
                        f"{miss_q4:.1f}%"
                    ]
                }),
                use_container_width=True,
                hide_index=True
            )


    # --------------------------------------------------------
    # BP2
    # --------------------------------------------------------
    elif python_bp == "Business Problem 2 — Recommendation & Adherence":

        st.title("Business Problem 2")
        st.caption("AI Recommendation & Adherence Gap")

        # Build recommendation base
        rec = ai_assessment[
            ["DIGITAL_ID", "RECOMMENDED_SLEEP", "RECOMMENDED_CALORIES", "DAILY_WATER_GOAL", "AI_RECOMMENDATION"]
        ].merge(
            sleep[["DIGITAL_ID", "SLEEP_HOURS"]],
            on="DIGITAL_ID",
            how="inner"
        ).merge(
            nutrition[["DIGITAL_ID", "DAILY_CALORIES", "WATER_INTAKE_L"]],
            on="DIGITAL_ID",
            how="inner"
        )

        rec["SLEEP_GAP"] = (
            pd.to_numeric(rec["RECOMMENDED_SLEEP"], errors="coerce") -
            pd.to_numeric(rec["SLEEP_HOURS"], errors="coerce")
        ).abs()

        rec["CALORIE_GAP"] = (
            pd.to_numeric(rec["RECOMMENDED_CALORIES"], errors="coerce") -
            pd.to_numeric(rec["DAILY_CALORIES"], errors="coerce")
        ).abs()

        rec["WATER_GAP"] = (
            pd.to_numeric(rec["DAILY_WATER_GOAL"], errors="coerce") -
            pd.to_numeric(rec["WATER_INTAKE_L"], errors="coerce")
        ).abs()

        rec["MEETS_SLEEP"] = rec["SLEEP_GAP"] <= 0.5
        rec["MEETS_CALORIES"] = rec["CALORIE_GAP"] <= 200
        rec["MEETS_WATER"] = rec["WATER_GAP"] <= 0.2
        rec["MEETS_ALL_THREE"] = rec["MEETS_SLEEP"] & rec["MEETS_CALORIES"] & rec["MEETS_WATER"]

        section("KPIs")

        sleep_pct = pct(rec["MEETS_SLEEP"])
        calorie_pct = pct(rec["MEETS_CALORIES"])
        water_pct = pct(rec["MEETS_WATER"])
        all_pct = pct(rec["MEETS_ALL_THREE"])

        kpi_cards([
            ("Sleep Adherence ±0.5 hr", fmt_pct(sleep_pct)),
            ("Calorie Adherence ±200 kcal", fmt_pct(calorie_pct)),
            ("Water Adherence ±0.2 L", fmt_pct(water_pct)),
            ("All Three Together", fmt_pct(all_pct)),
        ])

        section("Charts & Analysis")

        if analysis_type == "Descriptive":

            # Q1
            q1 = pd.DataFrame({
                "Recommendation": ["Sleep", "Calories", "Water"],
                "Average Absolute Gap": [
                    mean_or_nan(rec["SLEEP_GAP"]),
                    mean_or_nan(rec["CALORIE_GAP"]),
                    mean_or_nan(rec["WATER_GAP"])
                ]
            })
            safe_bar(
                q1, "Recommendation", "Average Absolute Gap",
                "Q1. Average Gap Between AI Targets and Actual Behaviour"
            )
            st.dataframe(q1.round(2), use_container_width=True, hide_index=True)

            # Q2-Q4
            q24 = pd.DataFrame({
                "Adherence Metric": [
                    "Sleep ±0.5 hr",
                    "Calories ±200 kcal",
                    "Water ±0.2 L",
                    "All three"
                ],
                "Percentage": [
                    sleep_pct, calorie_pct, water_pct, all_pct
                ]
            })
            safe_bar(
                q24, "Adherence Metric", "Percentage",
                "Q2–Q4. Recommendation Adherence"
            )
            st.dataframe(q24.round(2), use_container_width=True, hide_index=True)

            # Q5
            bmi = profile[["DIGITAL_ID", "BMI"]].merge(
                ai_assessment[["DIGITAL_ID", "AI_RECOMMENDATION"]],
                on="DIGITAL_ID",
                how="inner"
            )
            q5 = bmi.groupby("AI_RECOMMENDATION", dropna=False)["BMI"].mean().sort_values().reset_index()
            safe_bar(
                q5, "BMI", "AI_RECOMMENDATION",
                "Q5. Average BMI by AI Recommendation",
                orientation="h"
            )
            st.dataframe(q5.round(2), use_container_width=True, hide_index=True)

            # Q6
            q6 = rec[["DIGITAL_ID", "MEETS_ALL_THREE"]].merge(
                ai_assessment[["DIGITAL_ID", "MENTAL_WELLBEING"]],
                on="DIGITAL_ID",
                how="inner"
            ).merge(
                mental[["DIGITAL_ID", "HAPPINESS_INDEX"]],
                on="DIGITAL_ID",
                how="inner"
            )
            q6_summary = q6.groupby("MEETS_ALL_THREE")[["MENTAL_WELLBEING", "HAPPINESS_INDEX"]].mean().reset_index()
            q6_melt = q6_summary.melt(
                "MEETS_ALL_THREE",
                var_name="Measure",
                value_name="Average"
            )
            q6_melt["MEETS_ALL_THREE"] = q6_melt["MEETS_ALL_THREE"].map({
                True: "Adherent",
                False: "Non-adherent"
            })
            fig = px.bar(
                q6_melt,
                x="MEETS_ALL_THREE",
                y="Average",
                color="Measure",
                barmode="group",
                title="Q6. Wellbeing by Full Recommendation Adherence",
                text_auto=".1f"
            )
            fig.update_layout(height=390)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(q6_summary.round(2), use_container_width=True, hide_index=True)

        else:

            # Q1
            demo = rec[["DIGITAL_ID", "MEETS_ALL_THREE"]].merge(
                profile[["DIGITAL_ID", "AGE", "OCCUPATION", "INCOME_USD"]],
                on="DIGITAL_ID",
                how="inner"
            )
            demo["AGE_GROUP"] = pd.cut(
                pd.to_numeric(demo["AGE"], errors="coerce"),
                bins=[0, 30, 45, 60, 100],
                labels=["<30", "30-45", "45-60", "60+"]
            )
            try:
                demo["INCOME_BRACKET"] = pd.qcut(
                    pd.to_numeric(demo["INCOME_USD"], errors="coerce"),
                    q=4,
                    labels=["Low", "Mid-Low", "Mid-High", "High"],
                    duplicates="drop"
                )
            except Exception:
                demo["INCOME_BRACKET"] = "Overall"

            age_rate = demo.groupby("AGE_GROUP", observed=True)["MEETS_ALL_THREE"].mean().mul(100).reset_index(name="Adherence %")
            occ_rate = demo.groupby("OCCUPATION")["MEETS_ALL_THREE"].mean().mul(100).sort_values(ascending=False).reset_index(name="Adherence %")
            income_rate = demo.groupby("INCOME_BRACKET", observed=True)["MEETS_ALL_THREE"].mean().mul(100).reset_index(name="Adherence %")

            st.subheader("Q1. Full-Adherence Rate by Demographic Segment")
            c1, c2 = st.columns(2)
            with c1:
                safe_bar(age_rate, "AGE_GROUP", "Adherence %", "Age Group")
                safe_bar(income_rate, "INCOME_BRACKET", "Adherence %", "Income Bracket")
            with c2:
                safe_bar(
                    occ_rate.sort_values("Adherence %", ascending=True),
                    "Adherence %",
                    "OCCUPATION",
                    "Occupation",
                    orientation="h"
                )

            # Q2
            behavior = profile[
                ["DIGITAL_ID", "BMI", "SCREEN_TIME_HOURS", "PHYSICAL_ACTIVITY"]
            ].merge(
                ai_assessment[["DIGITAL_ID", "AI_RECOMMENDATION"]],
                on="DIGITAL_ID",
                how="inner"
            ).merge(
                rec[["DIGITAL_ID", "SLEEP_GAP", "CALORIE_GAP"]],
                on="DIGITAL_ID",
                how="inner"
            )
            q2 = behavior.groupby("AI_RECOMMENDATION").agg(
                Avg_Sleep_Gap=("SLEEP_GAP", "mean"),
                Avg_Calorie_Gap=("CALORIE_GAP", "mean"),
                Avg_Screen_Time=("SCREEN_TIME_HOURS", "mean")
            ).reset_index()

            st.subheader("Q2. Behaviour Gap by AI Recommendation")
            q2_melt = q2.melt(
                "AI_RECOMMENDATION",
                var_name="Measure",
                value_name="Average"
            )
            fig = px.bar(
                q2_melt,
                x="AI_RECOMMENDATION",
                y="Average",
                color="Measure",
                barmode="group",
                title="Sleep, Calorie and Screen-Time Gap by Recommendation",
            )
            fig.update_layout(height=430, xaxis_tickangle=-20)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(q2.round(2), use_container_width=True, hide_index=True)

            # Q3
            recovery_avg = visits.groupby("DIGITAL_ID")["RECOVERY_SCORE"].mean().reset_index(name="AVG_RECOVERY_SCORE")
            q3 = rec[["DIGITAL_ID", "MEETS_ALL_THREE"]].merge(
                medical[["DIGITAL_ID", "HEALTH_RISK_SCORE"]],
                on="DIGITAL_ID",
                how="inner"
            ).merge(
                recovery_avg,
                on="DIGITAL_ID",
                how="left"
            )
            q3_summary = q3.groupby("MEETS_ALL_THREE")[["HEALTH_RISK_SCORE", "AVG_RECOVERY_SCORE"]].mean().reset_index()
            q3_melt = q3_summary.melt("MEETS_ALL_THREE", var_name="Measure", value_name="Average")
            q3_melt["MEETS_ALL_THREE"] = q3_melt["MEETS_ALL_THREE"].map({
                True: "Adherent",
                False: "Non-adherent"
            })
            fig = px.bar(
                q3_melt, x="MEETS_ALL_THREE", y="Average",
                color="Measure", barmode="group",
                title="Q3. Health Risk and Recovery by Recommendation Adherence",
                text_auto=".1f"
            )
            fig.update_layout(height=390)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(q3_summary.round(2), use_container_width=True, hide_index=True)

            # Q4
            q4 = rec[
                ["RECOMMENDED_SLEEP", "SLEEP_HOURS", "RECOMMENDED_CALORIES", "DAILY_CALORIES"]
            ].apply(pd.to_numeric, errors="coerce").describe().T

            target_means = pd.DataFrame({
                "Measure": ["Actual Sleep", "Recommended Sleep", "Actual Calories", "Recommended Calories"],
                "Mean": [
                    mean_or_nan(rec["SLEEP_HOURS"]),
                    mean_or_nan(rec["RECOMMENDED_SLEEP"]),
                    mean_or_nan(rec["DAILY_CALORIES"]),
                    mean_or_nan(rec["RECOMMENDED_CALORIES"])
                ]
            })
            col1, col2 = st.columns(2)
            with col1:
                safe_bar(target_means, "Measure", "Mean", "Q4. Actual vs Recommended Average")
            with col2:
                st.dataframe(target_means.round(2), use_container_width=True, hide_index=True)

            below_sleep = pct(
                pd.to_numeric(rec["SLEEP_HOURS"], errors="coerce") <
                pd.to_numeric(rec["RECOMMENDED_SLEEP"], errors="coerce")
            )
            below_cal = pct(
                pd.to_numeric(rec["DAILY_CALORIES"], errors="coerce") <
                pd.to_numeric(rec["RECOMMENDED_CALORIES"], errors="coerce")
            )

            kpi_cards([
                ("Actual Sleep Below Recommendation", fmt_pct(below_sleep)),
                ("Actual Calories Below Recommendation", fmt_pct(below_cal)),
            ])


    # --------------------------------------------------------
    # PREDICTIVE
    # --------------------------------------------------------
    else:

        st.title("Predictive Analysis")
        st.caption("Logistic Regression using pre-diagnostic features")

        model_df = profile[
            ["DIGITAL_ID", "AGE", "BMI", "BLOOD_PRESSURE_SYSTOLIC",
             "FAMILY_HISTORY", "PHYSICAL_ACTIVITY", "SMOKING", "ALCOHOL"]
        ].merge(
            medical[["DIGITAL_ID", "DIABETES_STATUS"]],
            on="DIGITAL_ID",
            how="inner"
        ).merge(
            genetic[["DIGITAL_ID", "DIABETES_GENETIC_RISK"]],
            on="DIGITAL_ID",
            how="inner"
        ).merge(
            nutrition[["DIGITAL_ID", "DAILY_CALORIES", "SUGAR_G"]],
            on="DIGITAL_ID",
            how="inner"
        ).merge(
            daily_activity[["DIGITAL_ID", "WALKING_MINUTES", "WORKOUT_MINUTES"]],
            on="DIGITAL_ID",
            how="inner"
        )

        model_df["TARGET"] = (
            model_df["DIABETES_STATUS"].astype(str).str.upper() == "YES"
        ).astype(int)

        cat_cols = ["FAMILY_HISTORY", "DIABETES_GENETIC_RISK", "PHYSICAL_ACTIVITY", "SMOKING", "ALCOHOL"]
        num_cols = [
            "AGE", "BMI", "BLOOD_PRESSURE_SYSTOLIC",
            "DAILY_CALORIES", "SUGAR_G", "WALKING_MINUTES", "WORKOUT_MINUTES"
        ]

        required_cols = cat_cols + num_cols + ["TARGET"]
        model_data = model_df[required_cols].dropna()

        X = model_data[cat_cols + num_cols]
        y = model_data["TARGET"]

        if y.nunique() < 2:
            st.error("Predictive model needs both diabetic and non-diabetic target classes.")
            st.stop()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        preprocessor = ColumnTransformer([
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ])

        model = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
        ])

        model.fit(X_train, y_train)

        proba_test = model.predict_proba(X_test)[:, 1]
        pred_test = model.predict(X_test)

        auc = roc_auc_score(y_test, proba_test)
        cm = confusion_matrix(y_test, pred_test)

        section("KPIs")
        kpi_cards([
            ("Model AUC-ROC", f"{auc:.3f}"),
            ("Test Patients", f"{len(y_test):,}"),
            ("Diabetes Test Cases", f"{int(y_test.sum()):,}"),
            ("Predicted Positive", f"{int(pred_test.sum()):,}"),
        ])

        section("Charts & Analysis")

        col1, col2 = st.columns(2)

        with col1:
            cm_df = pd.DataFrame(
                cm,
                index=["Actual 0", "Actual 1"],
                columns=["Predicted 0", "Predicted 1"]
            )
            fig = px.imshow(
                cm_df,
                text_auto=True,
                title="Confusion Matrix"
            )
            fig.update_layout(height=390)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            precision_recall = classification_report(
                y_test, pred_test, output_dict=True, zero_division=0
            )
            metrics_df = pd.DataFrame({
                "Metric": ["Precision (Diabetes)", "Recall (Diabetes)", "F1 (Diabetes)"],
                "Value": [
                    precision_recall["1"]["precision"],
                    precision_recall["1"]["recall"],
                    precision_recall["1"]["f1-score"],
                ]
            })
            fig = px.bar(
                metrics_df,
                x="Metric",
                y="Value",
                text_auto=".2f",
                title="Diabetes Class Performance"
            )
            fig.update_layout(height=390, yaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)

        # Full population scoring using the fitted model
        full_model = model_df.dropna(subset=required_cols).copy()
        full_X = full_model[cat_cols + num_cols]
        full_model["MODEL_DIABETES_RISK"] = model.predict_proba(full_X)[:, 1] * 100

        top_n = max(1, int(np.ceil(len(full_model) * 0.15)))
        top15 = full_model.sort_values("MODEL_DIABETES_RISK", ascending=False).head(top_n)
        rest = full_model.sort_values("MODEL_DIABETES_RISK", ascending=False).iloc[top_n:]

        top15_rate = pct(top15["TARGET"] == 1)
        rest_rate = pct(rest["TARGET"] == 1) if len(rest) else np.nan

        kpi_cards([
            ("Top 15% Model-Risk Diabetes Rate", fmt_pct(top15_rate)),
            ("Remaining Patients Diabetes Rate", fmt_pct(rest_rate)),
            ("Model Lift (Top 15% vs Rest)", fmt_num(top15_rate / rest_rate, 2) + "x" if rest_rate else "—"),
        ])

        risk_bins = pd.cut(
            full_model["MODEL_DIABETES_RISK"],
            bins=[-np.inf, 20, 40, 60, 80, np.inf],
            labels=["0–20", "20–40", "40–60", "60–80", "80+"]
        )
        risk_dist = full_model.assign(Risk_Band=risk_bins).groupby(
            "Risk_Band", observed=False
        )["TARGET"].mean().mul(100).reset_index(name="Diabetes Rate")

        safe_bar(
            risk_dist, "Risk_Band", "Diabetes Rate",
            "Observed Diabetes Rate across Model-Risk Bands"
        )


# ============================================================
# POWER BI ANALYSIS
# ============================================================
else:

    st.sidebar.markdown("---")

    pbi_page = st.sidebar.radio(
        "Power BI Pages",
        [
            "Population Health & Vitals",
            "AI Wellness Score Validity Audit",
            "AI Score-to-Decision",
            "Diabetes Risk Analysis",
        ],
        key="pbi_page"
    )

    # ========================================================
    # PBI PAGE 1 — POPULATION HEALTH
    # ========================================================
    if pbi_page == "Population Health & Vitals":

        st.title("Population Health & Vitals")
        st.caption("Power BI-style population overview recreated in Streamlit")

        section("KPIs")

        total_patient = profile["DIGITAL_ID"].nunique()
        avg_age = mean_or_nan(profile["AGE"])
        abnormal = (
            (medical["BLOOD_TEST_STATUS"].astype(str).str.upper() == "ABNORMAL").sum()
            if "BLOOD_TEST_STATUS" in medical else 0
        )
        high_risk = (
            (medical["PHYSICIAN_RISK_LEVEL"].astype(str).str.upper() == "HIGH").sum()
            if "PHYSICIAN_RISK_LEVEL" in medical else 0
        )

        kpi_cards([
            ("Total Patient", f"{int(total_patient):,}"),
            ("Average Age", fmt_num(avg_age, 1)),
            ("Abnormal Blood Test", f"{int(abnormal):,}"),
            ("Total High Risk Patient", f"{int(high_risk):,}"),
        ])

        section("Charts & Analysis")

        col1, col2 = st.columns(2)

        with col1:
            if "PHYSICIAN_RISK_LEVEL" in medical:
                pr = medical["PHYSICIAN_RISK_LEVEL"].value_counts(dropna=False).reset_index()
                pr.columns = ["Risk Level", "Patients"]
                safe_donut(pr, "Risk Level", "Patients", "Physician Risk Distribution")

        with col2:
            if "LAB_STATUS" in labs:
                ls = labs["LAB_STATUS"].value_counts(dropna=False).reset_index()
                ls.columns = ["Lab Status", "Patients"]
                safe_bar(ls, "Lab Status", "Patients", "Lab Status Distribution")

        col1, col2 = st.columns(2)

        with col1:
            if {"HEALTH_RISK_SCORE", "PHYSICIAN_RISK_LEVEL"}.issubset(medical.columns):
                h = medical.groupby("PHYSICIAN_RISK_LEVEL")["HEALTH_RISK_SCORE"].mean().reset_index()
                safe_bar(h, "PHYSICIAN_RISK_LEVEL", "HEALTH_RISK_SCORE",
                         "Average Health Score by Physician Risk Level")

        with col2:
            # Direct merge keeps both clinical columns available.
            comp = medical[["DIGITAL_ID", "PHYSICIAN_RISK_LEVEL"]].merge(
                labs[["DIGITAL_ID", "LAB_STATUS"]],
                on="DIGITAL_ID",
                how="inner"
            )
            if not comp.empty:
                ct = pd.crosstab(
                    comp["LAB_STATUS"],
                    comp["PHYSICIAN_RISK_LEVEL"],
                    normalize="index"
                ).mul(100)
                ct = ct.reset_index().melt("LAB_STATUS", var_name="Risk Level", value_name="Percentage")
                fig = px.bar(
                    ct, x="LAB_STATUS", y="Percentage",
                    color="Risk Level", barmode="stack",
                    title="Risk Composition by Lab Status",
                    text_auto=".1f"
                )
                fig.update_layout(height=410)
                st.plotly_chart(fig, use_container_width=True)

        age = profile_medical.copy()
        age["AGE_GROUP"] = pd.cut(
            pd.to_numeric(age["AGE"], errors="coerce"),
            bins=[0, 30, 45, 60, 100],
            labels=["<30", "30-45", "45-60", "60+"]
        )

        age_risk = pd.crosstab(
            age["AGE_GROUP"],
            age["PHYSICIAN_RISK_LEVEL"],
            normalize="index"
        ).mul(100).reset_index().melt(
            "AGE_GROUP", var_name="Risk Level", value_name="Percentage"
        )

        fig = px.bar(
            age_risk,
            x="AGE_GROUP",
            y="Percentage",
            color="Risk Level",
            barmode="stack",
            title="Risk Distribution Across Age Groups",
            text_auto=".1f"
        )
        fig.update_layout(height=410)
        st.plotly_chart(fig, use_container_width=True)


    # ========================================================
    # PBI PAGE 2 — WELLNESS AUDIT
    # ========================================================
    elif pbi_page == "AI Wellness Score Validity Audit":

        st.title("AI Wellness Score Validity Audit")
        st.caption("Does the visible AI status reflect underlying clinical risk?")

        section("KPIs")

        high_risk_count = (
            (medical["PHYSICIAN_RISK_LEVEL"].astype(str).str.upper() == "HIGH").sum()
            if "PHYSICIAN_RISK_LEVEL" in medical else 0
        )

        unique_wellness = ai_assessment["WELLNESS_SCORE"].nunique() if "WELLNESS_SCORE" in ai_assessment else np.nan

        critical_high = len(
            medical.loc[
                medical["PHYSICIAN_RISK_LEVEL"].astype(str).str.upper() == "HIGH",
                ["DIGITAL_ID"]
            ].merge(
                labs.loc[
                    labs["LAB_STATUS"].astype(str).str.upper() == "CRITICAL",
                    ["DIGITAL_ID"]
                ],
                on="DIGITAL_ID",
                how="inner"
            )
        )

        high_risk_excellent = np.nan
        if {"PHYSICIAN_RISK_LEVEL", "OVERALL_AI_STATUS"}.issubset(wellness_audit.columns):
            high_phys = wellness_audit[
                wellness_audit["PHYSICIAN_RISK_LEVEL"].astype(str).str.upper() == "HIGH"
            ]
            high_risk_excellent = pct(
                high_phys["OVERALL_AI_STATUS"].astype(str).str.upper() == "EXCELLENT"
            )

        kpi_cards([
            ("Unique Wellness Score", f"{int(unique_wellness):,}" if not pd.isna(unique_wellness) else "—"),
            ("Critical + High Physician Risk", f"{critical_high:,}"),
            ("High Physician Risk → Excellent", fmt_pct(high_risk_excellent)),
            ("High Physician-Risk Patients", f"{int(high_risk_count):,}"),
        ])

        section("Charts & Analysis")

        col1, col2 = st.columns(2)

        with col1:
            status = ai_assessment["OVERALL_AI_STATUS"].value_counts(dropna=False).reset_index()
            status.columns = ["AI Status", "Patients"]
            safe_donut(status, "AI Status", "Patients", "AI Status Distribution")

        with col2:
            if {"PHYSICIAN_RISK_LEVEL", "OVERALL_AI_STATUS"}.issubset(wellness_audit.columns):
                ct = pd.crosstab(
                    wellness_audit["PHYSICIAN_RISK_LEVEL"],
                    wellness_audit["OVERALL_AI_STATUS"],
                    normalize="index"
                ).mul(100).reset_index().melt(
                    "PHYSICIAN_RISK_LEVEL", var_name="AI Status", value_name="Percentage"
                )
                fig = px.bar(
                    ct, x="PHYSICIAN_RISK_LEVEL", y="Percentage",
                    color="AI Status", barmode="stack",
                    title="Physician Risk vs AI Status",
                    text_auto=".1f"
                )
                fig.update_layout(height=410)
                st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            if {"PHYSICIAN_RISK_LEVEL", "WELLNESS_SCORE"}.issubset(wellness_audit.columns):
                w = wellness_audit.groupby("PHYSICIAN_RISK_LEVEL")["WELLNESS_SCORE"].mean().reset_index()
                safe_bar(w, "PHYSICIAN_RISK_LEVEL", "WELLNESS_SCORE",
                         "Average Wellness Score by Physician Risk")

        with col2:
            if {"LAB_STATUS", "OVERALL_AI_STATUS"}.issubset(lab_ai.columns):
                ct = pd.crosstab(
                    lab_ai["LAB_STATUS"],
                    lab_ai["OVERALL_AI_STATUS"],
                    normalize="index"
                ).mul(100).reset_index().melt(
                    "LAB_STATUS", var_name="AI Status", value_name="Percentage"
                )
                fig = px.bar(
                    ct, x="LAB_STATUS", y="Percentage",
                    color="AI Status", barmode="stack",
                    title="AI Status Across Labs",
                    text_auto=".1f"
                )
                fig.update_layout(height=410)
                st.plotly_chart(fig, use_container_width=True)

        if {"AGE_GROUP", "HEALTH_RISK_SCORE", "PHYSICIAN_RISK_LEVEL"}.issubset(profile_medical.columns):
            pass

        trend = profile_medical.copy()
        trend["AGE_GROUP"] = pd.cut(
            pd.to_numeric(trend["AGE"], errors="coerce"),
            bins=[0, 30, 45, 60, 100],
            labels=["<30", "30-45", "45-60", "60+"]
        )
        if {"AGE_GROUP", "HEALTH_RISK_SCORE", "PHYSICIAN_RISK_LEVEL"}.issubset(trend.columns):
            t = trend.groupby(
                ["AGE_GROUP", "PHYSICIAN_RISK_LEVEL"], observed=True
            )["HEALTH_RISK_SCORE"].mean().reset_index()
            safe_line(
                t, "AGE_GROUP", "HEALTH_RISK_SCORE",
                "Clinical Risk vs AI Output", color="PHYSICIAN_RISK_LEVEL"
            )


    # ========================================================
    # PBI PAGE 3 — AI SCORE TO DECISION
    # ========================================================
    elif pbi_page == "AI Score-to-Decision":

        st.title("AI Score-to-Decision")
        st.caption("Risk-score-to-decision translation gap")

        section("KPIs")

        diagnosed_count = int(
            (risk_base["DIABETES_STATUS"].astype(str).str.upper() == "YES").sum()
        )

        diagnosed_avg = mean_or_nan(
            risk_base.loc[
                risk_base["DIABETES_STATUS"].astype(str).str.upper() == "YES",
                "ONE_YEAR_DIABETES_RISK"
            ]
        )

        diabetic_excellent = pct(
            risk_base.loc[
                risk_base["DIABETES_STATUS"].astype(str).str.upper() == "YES",
                "AI_FUTURE_HEALTH_FORECAST"
            ].astype(str).str.upper() == "EXCELLENT"
        )

        high_genetic_low = pct(
            genetic_priority.loc[
                genetic_priority["DIABETES_GENETIC_RISK"].astype(str).str.upper() == "HIGH",
                "PREVENTIVE_CARE_PRIORITY"
            ].astype(str).str.upper() == "LOW"
        )

        kpi_cards([
            ("Diagnosed Diabetes Patients", f"{diagnosed_count:,}"),
            ("Avg Diabetes Risk — Diagnosed", fmt_num(diagnosed_avg, 1) + "%"),
            ("Diabetic → AI Excellent", fmt_pct(diabetic_excellent)),
            ("High Genetic Risk → Low Priority", fmt_pct(high_genetic_low)),
        ])

        section("Charts & Analysis")

        # Exact PBIX visual concepts
        col1, col2 = st.columns(2)

        with col1:
            q1 = risk_base.groupby("DIABETES_STATUS")["ONE_YEAR_DIABETES_RISK"].mean().reset_index()
            safe_bar(q1, "DIABETES_STATUS", "ONE_YEAR_DIABETES_RISK",
                     "One-Year Diabetes Risk by Status")

        with col2:
            forecast = risk_base.groupby(
                ["DIABETES_STATUS", "AI_FUTURE_HEALTH_FORECAST"]
            ).size().reset_index(name="Patients")
            forecast["Percentage"] = forecast.groupby("DIABETES_STATUS")["Patients"].transform(
                lambda s: s / s.sum() * 100
            )
            fig = px.bar(
                forecast,
                x="DIABETES_STATUS",
                y="Percentage",
                color="AI_FUTURE_HEALTH_FORECAST",
                barmode="stack",
                title="AI Forecast Distribution",
                text_auto=".1f"
            )
            fig.update_layout(height=410)
            st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            priority = pd.crosstab(
                genetic_priority["DIABETES_GENETIC_RISK"],
                genetic_priority["PREVENTIVE_CARE_PRIORITY"],
                normalize="index"
            ).mul(100).reset_index().melt(
                "DIABETES_GENETIC_RISK", var_name="Priority", value_name="Percentage"
            )
            fig = px.bar(
                priority,
                x="DIABETES_GENETIC_RISK",
                y="Percentage",
                color="Priority",
                barmode="stack",
                title="Preventive Care Priority",
                text_auto=".1f"
            )
            fig.update_layout(height=410)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            physician_risk = risk_base.groupby(
                "PHYSICIAN_RISK_LEVEL"
            )["ONE_YEAR_DIABETES_RISK"].mean().reset_index()
            safe_bar(
                physician_risk,
                "PHYSICIAN_RISK_LEVEL",
                "ONE_YEAR_DIABETES_RISK",
                "Diabetic Risk Across Physician Risk Level"
            )

        col1, col2 = st.columns(2)

        with col1:
            pcp_genetic = pd.crosstab(
                genetic_priority["PREVENTIVE_CARE_PRIORITY"],
                genetic_priority["DIABETES_GENETIC_RISK"],
            ).reset_index()
            pcp_melt = pcp_genetic.melt(
                "PREVENTIVE_CARE_PRIORITY",
                var_name="Genetic Risk", value_name="Patients"
            )
            safe_bar(
                pcp_melt,
                "PREVENTIVE_CARE_PRIORITY",
                "Patients",
                "Preventive Care Priority vs Genetic Risk",
                color="Genetic Risk"
            )

        with col2:
            age_risk = profile_future.copy()
            age_risk["AGE_GROUP"] = pd.cut(
                pd.to_numeric(age_risk["AGE"], errors="coerce"),
                bins=[0, 30, 45, 60, 100],
                labels=["<30", "30-45", "45-60", "60+"]
            )
            a = age_risk.groupby("AGE_GROUP", observed=True)["ONE_YEAR_DIABETES_RISK"].mean().reset_index()
            safe_line(a, "AGE_GROUP", "ONE_YEAR_DIABETES_RISK",
                      "Age Group by Diabetes Risk")


    # ========================================================
    # PBI PAGE 4 — DIABETES RISK
    # ========================================================
    else:

        st.title("Diabetes Risk Analysis")
        st.caption("Detailed diabetes-risk view from the Power BI report")

        section("KPIs")

        avg_diabetes = mean_or_nan(future_predictions["ONE_YEAR_DIABETES_RISK"])

        diagnosed = risk_base[
            risk_base["DIABETES_STATUS"].astype(str).str.upper() == "YES"
        ]

        risk_gap = (
            mean_or_nan(diagnosed["ONE_YEAR_DIABETES_RISK"]) -
            mean_or_nan(
                risk_base.loc[
                    risk_base["DIABETES_STATUS"].astype(str).str.upper() != "YES",
                    "ONE_YEAR_DIABETES_RISK"
                ]
            )
        )

        genetic_diabetes = genetic.merge(
            medical[["DIGITAL_ID", "DIABETES_STATUS"]],
            on="DIGITAL_ID",
            how="inner"
        )
        high_genetic_with_diabetes = (
            (
                genetic_diabetes["DIABETES_GENETIC_RISK"].astype(str).str.upper() == "HIGH"
            ) &
            (
                genetic_diabetes["DIABETES_STATUS"].astype(str).str.upper() == "YES"
            )
        ).sum()

        kpi_cards([
            ("Average Diabetes Risk", fmt_num(avg_diabetes, 1) + "%"),
            ("Diabetes Risk Gap", fmt_num(risk_gap, 1) + " pts"),
            ("Total Prediction Records", f"{len(future_predictions):,}"),
            ("High Genetic Risk with Diabetes", f"{int(high_genetic_with_diabetes):,}" if not pd.isna(high_genetic_with_diabetes) else "—"),
        ])

        section("Charts & Analysis")

        col1, col2 = st.columns(2)

        with col1:
            q1 = risk_base.groupby("DIABETES_STATUS")["ONE_YEAR_DIABETES_RISK"].mean().reset_index()
            safe_bar(
                q1,
                "DIABETES_STATUS",
                "ONE_YEAR_DIABETES_RISK",
                "Actual Diabetes Risk"
            )

        with col2:
            diabetes_genetic_status = genetic.merge(
                medical[["DIGITAL_ID", "DIABETES_STATUS"]],
                on="DIGITAL_ID",
                how="inner"
            )
            q2 = pd.crosstab(
                diabetes_genetic_status["DIABETES_GENETIC_RISK"],
                diabetes_genetic_status["DIABETES_STATUS"],
                normalize="index"
            ).mul(100).reset_index().melt(
                "DIABETES_GENETIC_RISK",
                var_name="Diabetes Status",
                value_name="Percentage"
            )
            fig = px.bar(
                q2,
                x="DIABETES_GENETIC_RISK",
                y="Percentage",
                color="Diabetes Status",
                barmode="stack",
                title="Genetic Risk vs Diabetes Status",
                text_auto=".1f"
            )
            fig.update_layout(height=410)
            st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            physician = risk_base.groupby(
                "PHYSICIAN_RISK_LEVEL"
            )["ONE_YEAR_DIABETES_RISK"].mean().reset_index()
            safe_bar(
                physician,
                "PHYSICIAN_RISK_LEVEL",
                "ONE_YEAR_DIABETES_RISK",
                "Diabetes Risk Across Physician Risk"
            )

        with col2:
            age_risk = profile_future.copy()
            age_risk["AGE_GROUP"] = pd.cut(
                pd.to_numeric(age_risk["AGE"], errors="coerce"),
                bins=[0, 30, 45, 60, 100],
                labels=["<30", "30-45", "45-60", "60+"]
            )
            age_risk_avg = age_risk.groupby("AGE_GROUP", observed=True)["ONE_YEAR_DIABETES_RISK"].mean().reset_index()
            safe_line(
                age_risk_avg,
                "AGE_GROUP",
                "ONE_YEAR_DIABETES_RISK",
                "Age Group by Diabetes Risk"
            )

        forecast_status = risk_base.groupby(
            ["DIABETES_STATUS", "AI_FUTURE_HEALTH_FORECAST"]
        ).size().reset_index(name="Patients")
        forecast_status["Percentage"] = forecast_status.groupby(
            "DIABETES_STATUS"
        )["Patients"].transform(lambda s: s / s.sum() * 100)

        fig = px.bar(
            forecast_status,
            x="DIABETES_STATUS",
            y="Percentage",
            color="AI_FUTURE_HEALTH_FORECAST",
            barmode="stack",
            title="Diabetes Status vs AI Forecast",
            text_auto=".1f"
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            '<div class="insight-box"><b>Note:</b> This page reproduces the analytical concepts and field relationships visible in the supplied Power BI report, using the local cleaned CSVs so the Streamlit app remains self-contained.</div>',
            unsafe_allow_html=True,
        )
