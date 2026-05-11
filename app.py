import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
DATA_DIR = BASE_DIR / "data"
DATA_FILE = DATA_DIR / "PCOS_data_without_infertility.xlsx"

st.set_page_config(
    page_title="Prediksi PCOS - SVM Linear",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CUSTOM_CSS = """
<style>
:root {
    --primary: #8b5cf6;
    --primary-dark: #6d28d9;
    --pink: #ec4899;
    --soft-bg: #f8f5ff;
    --border: rgba(88, 28, 135, 0.14);
    --text-muted: #64748b;
}

.block-container {
    padding-top: 1.85rem;
    padding-bottom: 2.5rem;
    max-width: 1280px;
}

[data-testid="stSidebar"] {
    display: none;
}

[data-testid="collapsedControl"] {
    display: none;
}

.page-intro {
    position: relative;
    overflow: hidden;
    text-align: center;
    padding: 1.35rem 1rem 1.55rem 1rem;
    margin-bottom: 1.15rem;
    border-bottom: 1px solid rgba(88, 28, 135, 0.14);
}

.page-intro::before {
    content: "";
    position: absolute;
    width: 240px;
    height: 240px;
    left: 50%;
    top: -170px;
    transform: translateX(-50%);
    border-radius: 999px;
    background: radial-gradient(circle, rgba(219, 39, 119, 0.18), rgba(126, 34, 206, 0.08), transparent 70%);
    z-index: 0;
}

.page-intro-inner {
    position: relative;
    z-index: 1;
}

.page-icon {
    width: 66px;
    height: 66px;
    margin: 0 auto 0.75rem auto;
    border-radius: 22px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, #4c1d95 0%, #7e22ce 52%, #db2777 100%);
    box-shadow: 0 14px 34px rgba(88, 28, 135, 0.22);
    color: white;
    font-size: 1.8rem;
}

.page-kicker {
    display: inline-flex;
    align-items: center;
    gap: 0.38rem;
    padding: 0.38rem 0.72rem;
    border-radius: 999px;
    background: #f3e8ff;
    color: #6d28d9;
    font-size: 0.78rem;
    font-weight: 900;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}

.page-title {
    color: #22113d;
    font-size: 2.42rem;
    line-height: 1.12;
    font-weight: 950;
    letter-spacing: -0.035em;
    margin: 0;
}

.page-title-gradient {
    background: linear-gradient(135deg, #4c1d95 0%, #7e22ce 48%, #db2777 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.page-subtitle {
    color: var(--text-muted);
    font-size: 1.02rem;
    line-height: 1.75;
    margin: 0.75rem auto 0 auto;
    max-width: 850px;
}

.quick-stats {
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 0.7rem;
    margin-top: 1.05rem;
}

.quick-stat {
    display: inline-flex;
    align-items: center;
    gap: 0.48rem;
    padding: 0.58rem 0.86rem;
    border-radius: 999px;
    background: rgba(255,255,255,0.92);
    border: 1px solid rgba(88, 28, 135, 0.13);
    box-shadow: 0 8px 22px rgba(88, 28, 135, 0.06);
    color: #581c87;
    font-size: 0.84rem;
    font-weight: 800;
}

.quick-stat span {
    color: #334155;
    font-weight: 700;
}

.top-nav {
    margin-top: 0;
    margin-bottom: 1.35rem;
}

.st-key-top_nav .stButton button {
    background: rgba(255, 255, 255, 0.98) !important;
    color: #581c87 !important;
    border: 1px solid rgba(88, 28, 135, 0.16) !important;
    border-radius: 999px !important;
    min-height: 54px;
    padding: 0.72rem 0.95rem;
    font-weight: 800;
    font-size: 0.96rem;
    box-shadow: 0 8px 20px rgba(88, 28, 135, 0.06);
    transition: transform 0.18s ease-in-out, box-shadow 0.18s ease-in-out, background 0.18s ease-in-out, color 0.18s ease-in-out;
}

.st-key-top_nav .stButton button:hover {
    background: linear-gradient(135deg, #6d28d9 0%, #7e22ce 55%, #db2777 100%) !important;
    color: #ffffff !important;
    transform: translateY(-2px);
    box-shadow: 0 14px 28px rgba(88, 28, 135, 0.18);
}

.st-key-top_nav .stButton button:focus,
.st-key-top_nav .stButton button:active {
    box-shadow: none !important;
    outline: none !important;
}

@media (max-width: 900px) {
    .page-title {
        font-size: 1.82rem;
    }

    .page-subtitle {
        font-size: 0.95rem;
    }

    .quick-stat {
        font-size: 0.78rem;
    }
}

.hero {
    border-radius: 30px;
    padding: 2rem 2.2rem;
    background: linear-gradient(135deg, #4c1d95 0%, #7e22ce 48%, #db2777 100%);
    color: white;
    box-shadow: 0 20px 50px rgba(76, 29, 149, 0.25);
    margin-bottom: 1.35rem;
}

.hero h1 {
    font-size: 2.45rem;
    line-height: 1.08;
    margin-bottom: 0.7rem;
    color: white;
}

.hero p {
    font-size: 1.02rem;
    color: rgba(255, 255, 255, 0.92);
    max-width: 920px;
}

.glass-card {
    border: 1px solid var(--border);
    border-radius: 24px;
    padding: 1.35rem 1.45rem;
    background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(248,245,255,0.86));
    box-shadow: 0 12px 35px rgba(88, 28, 135, 0.08);
    min-height: unset;
    height: auto;
    box-sizing: border-box;
}

.glass-card h3 {
    font-size: 1.65rem;
    line-height: 1.25;
    margin-bottom: 0.7rem !important;
    word-break: normal;
    overflow-wrap: normal;
}

.glass-card p {
    line-height: 1.65;
    margin: 0 !important;
}

.metric-card {
    border-radius: 22px;
    padding: 1.05rem 1.1rem;
    background: white;
    border: 1px solid var(--border);
    box-shadow: 0 10px 28px rgba(88, 28, 135, 0.08);
    min-height: 145px;
    height: 100%;
    box-sizing: border-box;
}

.metric-card .label {
    color: var(--text-muted);
    font-size: 0.86rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}

.metric-card .value {
    color: #3b0764;
    font-size: 1.85rem;
    font-weight: 900;
    margin-top: 0.15rem;
}

.metric-card .note {
    color: var(--text-muted);
    font-size: 0.82rem;
    margin-top: 0.2rem;
    line-height: 1.45;
}

.section-title {
    font-size: 1.25rem;
    font-weight: 900;
    color: #3b0764;
    margin: 1.3rem 0 0.8rem 0;
}

.section-title-center {
    font-size: 1.25rem;
    font-weight: 900;
    color: #3b0764;
    margin: 1.3rem 0 0.9rem 0;
    text-align: center;
}

.pill {
    display: inline-block;
    padding: 0.36rem 0.62rem;
    border-radius: 999px;
    background: #f3e8ff;
    color: #6d28d9;
    font-weight: 800;
    font-size: 0.82rem;
    margin-right: 0.35rem;
    margin-bottom: 0.35rem;
}

.step-card {
    border-radius: 18px;
    padding: 1.25rem 1.25rem;
    border: 1px solid var(--border);
    background: #ffffff;
    height: 270px;
    min-height: 270px;
    width: 100%;
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
    box-shadow: 0 10px 28px rgba(88, 28, 135, 0.05);
    margin-bottom: 1.5rem;
}

.step-card h4 {
    min-height: 48px;
    display: flex;
    align-items: flex-start;
    margin: .1rem 0 .55rem 0 !important;
    line-height: 1.25;
    font-size: 1.35rem;
    word-break: normal;
    overflow-wrap: normal;
    white-space: normal;
}

.step-card p {
    line-height: 1.6;
    margin: 0 !important;
    font-size: 0.94rem;
}

.step-number {
    width: 34px;
    height: 34px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    border-radius: 999px;
    background: #7e22ce;
    color: white;
    font-weight: 900;
    margin-bottom: 0.65rem;
    flex-shrink: 0;
}

.result-positive {
    padding: 1.2rem 1.3rem;
    border-radius: 24px;
    background: linear-gradient(135deg, #fff1f2, #ffe4e6);
    border: 1px solid #fda4af;
    box-shadow: 0 12px 30px rgba(244, 63, 94, 0.10);
}

.result-negative {
    padding: 1.2rem 1.3rem;
    border-radius: 24px;
    background: linear-gradient(135deg, #ecfdf5, #dcfce7);
    border: 1px solid #86efac;
    box-shadow: 0 12px 30px rgba(34, 197, 94, 0.10);
}

.warning-card {
    padding: 1rem 1.1rem;
    border-radius: 18px;
    background: #fff7ed;
    border: 1px solid #fed7aa;
    color: #7c2d12;
}

.small-muted {
    color: var(--text-muted);
    font-size: 0.92rem;
}

hr.soft {
    border: none;
    height: 1px;
    background: rgba(88, 28, 135, 0.12);
    margin: 1.1rem 0;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 999px;
    padding: 0.55rem 1rem;
    background: #f3e8ff;
    color: #581c87;
    font-weight: 800;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #7e22ce, #db2777) !important;
    color: #ffffff !important;
}

.csv-rules-card {
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 1.1rem 1.25rem;
    background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(248,245,255,0.86));
    box-shadow: 0 10px 28px rgba(88, 28, 135, 0.06);
    height: auto;
    min-height: unset;
    margin-top: 1rem;
    margin-bottom: 1rem;
    line-height: 1.7;
}

.csv-rules-card b {
    color: #3b0764;
    font-size: 1rem;
}

.edu-card {
    border: 1px solid var(--border);
    border-radius: 24px;
    padding: 1.45rem 1.55rem;
    background: linear-gradient(180deg, rgba(255,255,255,0.97), rgba(248,245,255,0.88));
    box-shadow: 0 12px 35px rgba(88, 28, 135, 0.08);
    min-height: 300px;
    height: 100%;
    box-sizing: border-box;
}

.edu-card h3 {
    color: #3b0764;
    font-size: 1.55rem;
    line-height: 1.25;
    margin: 0 0 0.85rem 0 !important;
}

.edu-card p {
    color: var(--text-muted);
    font-size: 0.95rem;
    line-height: 1.75;
    margin: 0 !important;
}

.edu-row-space {
    height: 28px;
}

.edu-source-note {
    color: var(--text-muted);
    font-size: 0.88rem;
    line-height: 1.65;
    margin-top: 0.6rem;
    margin-bottom: 1.2rem;
}

.edu-source-note a {
    color: #6d28d9;
    font-weight: 800;
    text-decoration: none;
}

.edu-source-note a:hover {
    text-decoration: underline;
}

.myth-card-wrap {
    perspective: 1200px;
    width: 100%;
    margin-bottom: 1.3rem;
}

.myth-toggle {
    display: none;
}

.myth-card {
    display: block;
    width: 100%;
    height: 270px;
    cursor: pointer;
}

.myth-card-inner {
    position: relative;
    width: 100%;
    height: 100%;
    transition: transform 0.65s ease;
    transform-style: preserve-3d;
}

.myth-toggle:checked + .myth-card .myth-card-inner {
    transform: rotateY(180deg);
}

.myth-card-face {
    position: absolute;
    inset: 0;
    border-radius: 24px;
    padding: 1.35rem 1.4rem;
    backface-visibility: hidden;
    box-sizing: border-box;
    border: 1px solid var(--border);
    box-shadow: 0 12px 35px rgba(88, 28, 135, 0.08);
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}

.myth-card-front {
    background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(248,245,255,0.9));
}

.myth-card-back {
    background: linear-gradient(135deg, #4c1d95 0%, #7e22ce 52%, #db2777 100%);
    color: white;
    transform: rotateY(180deg);
}

.myth-label {
    display: inline-block;
    width: fit-content;
    padding: 0.35rem 0.65rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 900;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

.myth-card-front .myth-label {
    background: #f3e8ff;
    color: #6d28d9;
}

.myth-card-back .myth-label {
    background: rgba(255,255,255,0.18);
    color: #ffffff;
    border: 1px solid rgba(255,255,255,0.25);
}

.myth-title {
    color: #3b0764;
    font-size: 1.28rem;
    font-weight: 900;
    line-height: 1.35;
    margin-top: 0.9rem;
}

.fact-text {
    color: rgba(255,255,255,0.94);
    font-size: 0.92rem;
    line-height: 1.65;
    margin-top: 0.8rem;
}

.myth-hint {
    color: var(--text-muted);
    font-size: 0.82rem;
    margin-top: 0.9rem;
}

.fact-source {
    margin-top: 0.7rem;
    font-size: 0.82rem;
}

.fact-source a {
    color: #ffffff;
    font-weight: 900;
    text-decoration: underline;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    model = joblib.load(ARTIFACT_DIR / "svm_linear_bayes_ga_model.pkl")
    imputer = joblib.load(ARTIFACT_DIR / "median_imputer.pkl")
    scaler = joblib.load(ARTIFACT_DIR / "standard_scaler.pkl")
    metadata = json.loads((ARTIFACT_DIR / "metadata.json").read_text(encoding="utf-8"))
    metrics = json.loads((ARTIFACT_DIR / "metrics.json").read_text(encoding="utf-8"))
    return model, imputer, scaler, metadata, metrics


@st.cache_data
def load_tables():
    return {
        "bayes": pd.read_csv(ARTIFACT_DIR / "bayesian_svm_ga_metrics.csv"),
        "baseline": pd.read_csv(ARTIFACT_DIR / "baseline_svm_metrics.csv"),
        "ga": pd.read_csv(ARTIFACT_DIR / "svm_ga_metrics.csv"),
        "summary": pd.read_csv(ARTIFACT_DIR / "dataset_summary.csv"),
        "stats": pd.read_csv(ARTIFACT_DIR / "selected_feature_stats.csv"),
        "coef": pd.read_csv(ARTIFACT_DIR / "linear_svm_feature_coefficients.csv"),
    }


@st.cache_data
def load_dataset_preview():
    df = pd.read_excel(DATA_FILE, sheet_name="Full_new")
    df.columns = df.columns.str.strip()
    return df


model, imputer, scaler, metadata, metrics = load_artifacts()
tables = load_tables()

FEATURE_COLUMNS = metadata["feature_columns"]
SELECTED_FEATURES = metadata["selected_features"]
REMOVED_FEATURES = metadata["removed_features"]
feature_stats = tables["stats"].set_index("Feature")
TARGET = "PCOS (Y/N)"


def metric_card(label: str, value: str, note: str = ""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            <div class="note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def info_card(title: str, body: str, icon: str = "✨"):
    st.markdown(
        f"""
        <div class="glass-card">
            <div style="font-size:1.55rem; margin-bottom:0.25rem;">{icon}</div>
            <h3 style="margin:0 0 .45rem 0; color:#3b0764;">{title}</h3>
            <p class="small-muted" style="margin:0;">{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def step_card(number: int, title: str, body: str):
    st.markdown(
        f"""
        <div class="step-card">
            <div class="step-number">{number}</div>
            <h4 style="margin:.1rem 0 .35rem 0; color:#3b0764;">{title}</h4>
            <p class="small-muted" style="margin:0;">{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def eda_card(title, caption=""):
    st.markdown(
        f"""
        <div class="glass-card">
            <h3 style="margin:0 0 .35rem 0; color:#3b0764;">{title}</h3>
            <p class="small-muted" style="margin:0;">{caption}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def myth_fact_card(card_id, myth, fact, source_url, source_label):
    st.markdown(
        f"""
        <div class="myth-card-wrap">
            <input type="checkbox" id="{card_id}" class="myth-toggle">
            <label for="{card_id}" class="myth-card">
                <div class="myth-card-inner">
                    <div class="myth-card-face myth-card-front">
                        <div>
                            <span class="myth-label">Mitos</span>
                            <div class="myth-title">{myth}</div>
                        </div>
                        <div class="myth-hint">Klik kartu untuk melihat fakta</div>
                    </div>
                    <div class="myth-card-face myth-card-back">
                        <div>
                            <span class="myth-label">Fakta</span>
                            <div class="fact-text">{fact}</div>
                        </div>
                        <div class="fact-source">
                            Sumber:
                            <a href="{source_url}" target="_blank">{source_label}</a>
                        </div>
                    </div>
                </div>
            </label>
        </div>
        """,
        unsafe_allow_html=True,
    )


def default_for(feature):
    if feature in feature_stats.index and not pd.isna(feature_stats.loc[feature, "50%"]):
        return float(feature_stats.loc[feature, "50%"])
    return 0.0


def min_for(feature):
    if feature in feature_stats.index and not pd.isna(feature_stats.loc[feature, "min"]):
        return float(feature_stats.loc[feature, "min"])
    return 0.0


def max_for(feature):
    if feature in feature_stats.index and not pd.isna(feature_stats.loc[feature, "max"]):
        return float(feature_stats.loc[feature, "max"])
    return 9999.0


def make_number_input(feature, label=None, help_text=None, key=None):
    mn = min_for(feature)
    mx = max_for(feature)
    default = default_for(feature)
    span = mx - mn

    if span <= 0:
        low, high = 0.0, max(1.0, mx + 1.0)
    else:
        low = max(0.0, mn - 0.1 * span)
        high = mx + 0.1 * span

    step = 0.01 if "Ratio" in feature or "/" in feature else 0.1

    return st.number_input(
        label or feature,
        min_value=float(low),
        max_value=float(high),
        value=float(default),
        step=float(step),
        help=help_text,
        key=key or feature,
    )


def yes_no_input(label: str, key: str):
    label_value = st.radio(label, ["Tidak", "Ya"], horizontal=True, key=key)
    return 1 if label_value == "Ya" else 0


def preprocess_input(input_values: dict):
    full_input = {feature: np.nan for feature in FEATURE_COLUMNS}
    full_input.update(input_values)

    input_df = pd.DataFrame([full_input], columns=FEATURE_COLUMNS)

    for col in input_df.columns:
        input_df[col] = pd.to_numeric(input_df[col], errors="coerce")

    imputed = imputer.transform(input_df)
    scaled = scaler.transform(imputed)
    processed = pd.DataFrame(scaled, columns=FEATURE_COLUMNS)

    return processed[SELECTED_FEATURES]


def preprocess_batch_dataframe(raw_df: pd.DataFrame):
    df = raw_df.copy()
    df.columns = df.columns.astype(str).str.strip()

    missing_columns = [col for col in FEATURE_COLUMNS if col not in df.columns]
    missing_selected = [col for col in SELECTED_FEATURES if col not in df.columns]
    extra_columns = [col for col in df.columns if col not in FEATURE_COLUMNS]

    full_df = pd.DataFrame(index=df.index)

    for col in FEATURE_COLUMNS:
        if col in df.columns:
            full_df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            full_df[col] = np.nan

    imputed = imputer.transform(full_df)
    scaled = scaler.transform(imputed)
    processed = pd.DataFrame(scaled, columns=FEATURE_COLUMNS, index=df.index)

    return processed[SELECTED_FEATURES], missing_columns, missing_selected, extra_columns


@st.cache_data
def build_csv_template():
    row = {}
    for feature in SELECTED_FEATURES:
        row[feature] = default_for(feature)
    return pd.DataFrame([row])


def classify_margin(score: float):
    abs_score = abs(score)
    if abs_score < 0.25:
        return "Margin tipis"
    if abs_score < 1.0:
        return "Margin sedang"
    return "Margin kuat"


def plot_confusion_matrix(cm):
    cm = np.array(cm)

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    im = ax.imshow(cm, cmap="Purples", vmin=0, vmax=max(cm.max(), 1))

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Tidak PCOS", "PCOS"], fontsize=11, fontweight="bold")
    ax.set_yticklabels(["Tidak PCOS", "PCOS"], fontsize=11, fontweight="bold")

    ax.set_xlabel("Prediksi Model", fontsize=12, fontweight="bold", labelpad=12)
    ax.set_ylabel("Data Aktual", fontsize=12, fontweight="bold", labelpad=12)
    ax.set_title("Confusion Matrix Model Final", fontsize=15, fontweight="bold", color="#3b0764", pad=18)

    total = cm.sum()

    for i in range(2):
        for j in range(2):
            value = int(cm[i, j])
            percent = (value / total * 100) if total > 0 else 0
            text_color = "white" if value > cm.max() / 2 else "#3b0764"
            ax.text(
                j,
                i,
                f"{value}\n({percent:.1f}%)",
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                color=text_color,
            )

    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)
    cbar.outline.set_visible(False)

    fig.tight_layout()
    return fig


def plot_barh(df, x_col, y_col, title, top_n=12):
    plot_df = df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, max(4, 0.35 * len(plot_df))))
    ax.barh(plot_df[y_col], plot_df[x_col])
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel("")
    fig.tight_layout()
    return fig


def label_target(series: pd.Series):
    return series.map({0: "Tidak PCOS", 1: "PCOS"}).fillna(series.astype(str))


def apply_plot_layout(fig, height=420):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        margin=dict(l=28, r=28, t=58, b=30),
        font=dict(size=12, color="#334155"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )
    return fig


def prepare_eda_dataframe():
    raw_df = load_dataset_preview()
    raw_df.columns = raw_df.columns.astype(str).str.strip()

    eda_df = raw_df.drop(
        columns=["Sl. No", "Patient File No.", "Unnamed: 44"],
        errors="ignore",
    )

    if TARGET not in eda_df.columns:
        return None

    eda_df[TARGET] = pd.to_numeric(eda_df[TARGET], errors="coerce")
    eda_df = eda_df.dropna(subset=[TARGET]).copy()
    eda_df[TARGET] = eda_df[TARGET].astype(int)

    for col in eda_df.columns:
        if col != TARGET:
            eda_df[col] = pd.to_numeric(eda_df[col], errors="coerce")

    return eda_df


def target_donut_chart(y: pd.Series):
    target_counts = label_target(y).value_counts().reset_index()
    target_counts.columns = ["Kelas", "Jumlah"]

    fig = px.pie(
        target_counts,
        names="Kelas",
        values="Jumlah",
        hole=0.58,
        title="Komposisi Kelas Target PCOS",
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        pull=[0.03] * len(target_counts),
    )

    fig.add_annotation(
        text="PCOS",
        x=0.5,
        y=0.52,
        showarrow=False,
        font_size=23,
        font_color="#3b0764",
    )

    fig.add_annotation(
        text="Dataset",
        x=0.5,
        y=0.44,
        showarrow=False,
        font_size=12,
        font_color="#64748b",
    )

    return apply_plot_layout(fig, 430)


def numeric_histogram(eda_df: pd.DataFrame, feature: str):
    temp = eda_df[[feature, TARGET]].copy()
    temp[feature] = pd.to_numeric(temp[feature], errors="coerce")
    temp = temp.dropna(subset=[feature])
    temp["Kelas"] = label_target(temp[TARGET])

    fig = px.histogram(
        temp,
        x=feature,
        color="Kelas",
        nbins=28,
        marginal="box",
        barmode="overlay",
        opacity=0.72,
        title=f"Distribusi {feature} Berdasarkan Kelas",
    )

    return apply_plot_layout(fig, 460)


def grouped_boxplot(eda_df: pd.DataFrame, feature: str):
    temp = eda_df[[feature, TARGET]].copy()
    temp[feature] = pd.to_numeric(temp[feature], errors="coerce")
    temp = temp.dropna(subset=[feature])
    temp["Kelas"] = label_target(temp[TARGET])

    fig = px.box(
        temp,
        x="Kelas",
        y=feature,
        color="Kelas",
        points="outliers",
        title=f"Perbandingan {feature}: PCOS vs Tidak PCOS",
    )

    return apply_plot_layout(fig, 430)


def missing_value_chart(eda_df: pd.DataFrame, features: list):
    missing = eda_df[features].isna().sum().reset_index()
    missing.columns = ["Fitur", "Missing"]
    missing = missing[missing["Missing"] > 0].sort_values("Missing", ascending=True).tail(15)

    if missing.empty:
        return None

    fig = px.bar(
        missing,
        x="Missing",
        y="Fitur",
        orientation="h",
        title="Top Missing Value pada Fitur Model",
    )

    return apply_plot_layout(fig, 430)


def top_feature_difference(eda_df: pd.DataFrame, features: list, top_n=12):
    rows = []

    for feature in features:
        temp = pd.to_numeric(eda_df[feature], errors="coerce")
        group0 = temp[eda_df[TARGET] == 0].dropna()
        group1 = temp[eda_df[TARGET] == 1].dropna()

        if len(group0) < 2 or len(group1) < 2:
            continue

        mean0 = group0.mean()
        mean1 = group1.mean()
        pooled_std = temp.std()

        if pd.isna(pooled_std) or pooled_std == 0:
            continue

        diff = (mean1 - mean0) / pooled_std

        rows.append(
            {
                "Fitur": feature,
                "Mean Tidak PCOS": mean0,
                "Mean PCOS": mean1,
                "Selisih Terstandar": diff,
                "Arah": "Lebih tinggi pada PCOS" if diff > 0 else "Lebih tinggi pada Tidak PCOS",
                "Abs": abs(diff),
            }
        )

    diff_df = pd.DataFrame(rows)

    if diff_df.empty:
        return None, pd.DataFrame()

    diff_df = diff_df.sort_values("Abs", ascending=False).head(top_n)
    diff_df = diff_df.sort_values("Selisih Terstandar")

    fig = px.bar(
        diff_df,
        x="Selisih Terstandar",
        y="Fitur",
        color="Arah",
        orientation="h",
        title="Top Fitur dengan Perbedaan Rata-rata Paling Menonjol",
        hover_data={
            "Mean Tidak PCOS": ":.3f",
            "Mean PCOS": ":.3f",
            "Abs": False,
        },
    )

    fig.add_vline(
        x=0,
        line_width=1,
        line_dash="dash",
        line_color="#94a3b8",
    )

    return apply_plot_layout(fig, 560), diff_df.drop(columns=["Abs"])


def correlation_heatmap(eda_df: pd.DataFrame, features: list):
    corr_df = eda_df[features].apply(pd.to_numeric, errors="coerce")
    corr = corr_df.corr().round(2)

    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        zmin=-1,
        zmax=1,
        color_continuous_scale="RdBu_r",
        title="Heatmap Korelasi Fitur Terpilih",
    )

    fig.update_xaxes(tickangle=45)
    return apply_plot_layout(fig, 640)


def scatter_relation(eda_df: pd.DataFrame, x_feature: str, y_feature: str):
    temp = eda_df[[x_feature, y_feature, TARGET]].copy()
    temp[x_feature] = pd.to_numeric(temp[x_feature], errors="coerce")
    temp[y_feature] = pd.to_numeric(temp[y_feature], errors="coerce")
    temp = temp.dropna(subset=[x_feature, y_feature])
    temp["Kelas"] = label_target(temp[TARGET])

    fig = px.scatter(
        temp,
        x=x_feature,
        y=y_feature,
        color="Kelas",
        opacity=0.78,
        title=f"Hubungan {x_feature} dan {y_feature}",
    )

    return apply_plot_layout(fig, 460)


def set_page(page_name: str):
    st.session_state["page"] = page_name


if "page" not in st.session_state:
    st.session_state["page"] = "Beranda"


nav_items = [
    ("Beranda", "🏠 Beranda"),
    ("Prediksi PCOS", "🩺 Prediksi PCOS"),
    ("Informasi Model", "📊 Informasi Model"),
    ("Edukasi PCOS", "📚 Edukasi PCOS"),
]

page = st.session_state["page"]

st.markdown(
    """
    <div class="page-intro">
        <div class="page-intro-inner">
            <div class="page-icon">🩺</div>
            <h1 class="page-title">
                Sistem Prediksi <span class="page-title-gradient">PCOS</span>
            </h1>
            <div class="page-subtitle">
                Sistem berbasis web untuk membantu memperkirakan kemungkinan seseorang mengalami
                <em>Polycystic Ovary Syndrome</em> berdasarkan data gejala, riwayat kesehatan,
                dan parameter medis tertentu.
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="top-nav">', unsafe_allow_html=True)
with st.container(key="top_nav"):
    nav_space_left, *nav_cols, nav_space_right = st.columns([0.18, 1, 1, 1, 1, 0.18], gap="medium")

    for col, (page_key, label) in zip(nav_cols, nav_items):
        with col:
            if st.button(label, key=f"nav_{page_key}", use_container_width=True):
                set_page(page_key)
                st.rerun()
st.markdown('</div>', unsafe_allow_html=True)


if page == "Beranda":
    st.markdown(
        """
        <div class="hero">
            <h1>Prediksi <em>Polycystic Ovary Syndrome</em></h1>
            <p>
                Klasifikasi <em>Polycystic Ovary Syndrome</em> Menggunakan Algoritma
                <em>Support Vector Machine Kernel Linear</em> dengan Seleksi Fitur
                <em>Genetic Algorithm</em> dan Optimasi <em>Hyperparameter</em>
                Berbasis <em>Bayesian Optimization</em>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        metric_card("Accuracy", f"{metrics['accuracy']:.3f}", "Ketepatan prediksi keseluruhan")
    with m2:
        metric_card("Precision", f"{metrics['precision']:.3f}", "Ketepatan prediksi PCOS")
    with m3:
        metric_card("Recall", f"{metrics['recall']:.3f}", "Kemampuan mendeteksi PCOS")
    with m4:
        metric_card("F1-score", f"{metrics['f1_score']:.3f}", "Keseimbangan presisi recall")
    with m5:
        metric_card("ROC-AUC", f"{metrics['roc_auc']:.3f}", "Kemampuan membedakan kelas")

    st.markdown('<div class="section-title-center">Ringkasan Sistem</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        info_card(
            "Model Final",
            "Model final menggunakan SVM kernel linear dengan seleksi fitur GA dan optimasi Bayesian.",
            "🧠",
        )
    with c2:
        info_card(
            "Cek Prediksi",
            "Pengguna dapat memprediksi PCOS menggunakan fitur terpilih dari Genetic Algorithm.",
            "🔍",
        )
    with c3:
        info_card(
            "Output",
            "Sistem menampilkan hasil klasifikasi PCOS dan skor keputusan model terhadap prediksi.",
            "📊",
        )

    st.markdown('<div class="section-title-center">Alur Penggunaan</div>', unsafe_allow_html=True)

    s1, s2, s3, s4 = st.columns(4, gap="large")
    with s1:
        step_card(1, "Isi Data", "Masukkan data sesuai fitur yang digunakan model.")
    with s2:
        step_card(2, "Preprocess", "Sistem menyiapkan data sesuai proses training.")
    with s3:
        step_card(3, "Prediksi", "Model SVM linear memproses data input.")
    with s4:
        step_card(4, "Interpretasi", "Sistem menampilkan hasil klasifikasi dan skor keputusan.")

    st.markdown("<br>", unsafe_allow_html=True)


elif page == "Prediksi PCOS":
    st.markdown(
        """
        <div class="hero">
            <h1>Prediksi PCOS</h1>
            <p>Lakukan klasifikasi PCOS melalui input data tunggal atau unggah CSV untuk memproses banyak data sekaligus.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Mode Prediksi</div>', unsafe_allow_html=True)
    tab_manual, tab_batch = st.tabs(["🧾 Prediksi Manual", "📦 Prediksi Batch CSV"])

    with tab_manual:
        st.markdown('<div class="section-title">Form Input Data Manual</div>', unsafe_allow_html=True)
        st.caption("Gunakan mode ini untuk memprediksi satu data responden/pasien.")

        with st.form("prediction_form"):
            tab1, tab2, tab3 = st.tabs(["Antropometri", "Riwayat & Gejala", "Klinis/Lab"])
            values = {}

            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    values["Age (yrs)"] = make_number_input("Age (yrs)", "Usia (tahun)")
                    values["Weight (Kg)"] = make_number_input("Weight (Kg)", "Berat badan (Kg)")
                    values["Height(Cm)"] = make_number_input("Height(Cm)", "Tinggi badan (Cm)")
                    values["Hip(inch)"] = make_number_input("Hip(inch)", "Lingkar panggul (inch)")
                    values["Waist(inch)"] = make_number_input("Waist(inch)", "Lingkar pinggang (inch)")
                with col2:
                    values["Waist:Hip Ratio"] = make_number_input("Waist:Hip Ratio", "Rasio pinggang:panggul")
                    values["Pulse rate(bpm)"] = make_number_input("Pulse rate(bpm)", "Pulse rate (bpm)")
                    values["RR (breaths/min)"] = make_number_input("RR (breaths/min)", "Respiratory rate (breaths/min)")
                    values["BP _Systolic (mmHg)"] = make_number_input("BP _Systolic (mmHg)", "Tekanan sistolik")
                    values["BP _Diastolic (mmHg)"] = make_number_input("BP _Diastolic (mmHg)", "Tekanan diastolik")

            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    values["Cycle(R/I)"] = st.radio(
                        "Cycle(R/I)",
                        options=[2, 4, 5],
                        horizontal=True,
                        key="cycle_code",
                        help="Gunakan kode sesuai dataset penelitian. Umumnya 2 dan 4 muncul sebagai kode utama pada data.",
                    )
                    values["Cycle length(days)"] = make_number_input("Cycle length(days)", "Panjang siklus (hari)")
                    values["Marraige Status (Yrs)"] = make_number_input("Marraige Status (Yrs)", "Lama menikah (tahun)")
                    values["Pregnant(Y/N)"] = yes_no_input("Sedang/pernah hamil", "pregnant")
                    values["Weight gain(Y/N)"] = yes_no_input("Riwayat kenaikan berat badan", "weight_gain")
                with col2:
                    values["hair growth(Y/N)"] = yes_no_input("Pertumbuhan rambut berlebih", "hair_growth")
                    values["Skin darkening (Y/N)"] = yes_no_input("Penggelapan kulit", "skin_darkening")
                    values["Hair loss(Y/N)"] = yes_no_input("Kerontokan rambut", "hair_loss")
                    values["Pimples(Y/N)"] = yes_no_input("Jerawat", "pimples")
                    values["Fast food (Y/N)"] = yes_no_input("Konsumsi fast food", "fast_food")
                    values["Reg.Exercise(Y/N)"] = yes_no_input("Olahraga teratur", "regular_exercise")

            with tab3:
                col1, col2 = st.columns(2)
                with col1:
                    values["Hb(g/dl)"] = make_number_input("Hb(g/dl)", "Hb (g/dl)")
                    values["I   beta-HCG(mIU/mL)"] = make_number_input("I   beta-HCG(mIU/mL)", "I beta-HCG (mIU/mL)")
                    values["II    beta-HCG(mIU/mL)"] = make_number_input("II    beta-HCG(mIU/mL)", "II beta-HCG (mIU/mL)")
                    values["FSH(mIU/mL)"] = make_number_input("FSH(mIU/mL)", "FSH (mIU/mL)")
                    values["LH(mIU/mL)"] = make_number_input("LH(mIU/mL)", "LH (mIU/mL)")
                    values["FSH/LH"] = make_number_input("FSH/LH", "Rasio FSH/LH")
                    values["TSH (mIU/L)"] = make_number_input("TSH (mIU/L)", "TSH (mIU/L)")
                with col2:
                    values["AMH(ng/mL)"] = make_number_input("AMH(ng/mL)", "AMH (ng/mL)")
                    values["PRL(ng/mL)"] = make_number_input("PRL(ng/mL)", "PRL (ng/mL)")
                    values["Vit D3 (ng/mL)"] = make_number_input("Vit D3 (ng/mL)", "Vitamin D3 (ng/mL)")
                    values["PRG(ng/mL)"] = make_number_input("PRG(ng/mL)", "Progesterone/PRG (ng/mL)")
                    values["Follicle No. (L)"] = make_number_input("Follicle No. (L)", "Jumlah folikel kiri")
                    values["Follicle No. (R)"] = make_number_input("Follicle No. (R)", "Jumlah folikel kanan")
                    values["Avg. F size (L) (mm)"] = make_number_input("Avg. F size (L) (mm)", "Rata-rata ukuran folikel kiri")
                    values["Endometrium (mm)"] = make_number_input("Endometrium (mm)", "Endometrium (mm)")

            submit = st.form_submit_button("Prediksi", use_container_width=True)

        if submit:
            X_input = preprocess_input(values)
            pred = int(model.predict(X_input)[0])
            score = float(model.decision_function(X_input)[0])
            margin = classify_margin(score)

            st.markdown('<div class="section-title">Hasil Prediksi</div>', unsafe_allow_html=True)
            res_col, score_col = st.columns([1.25, 1])

            with res_col:
                if pred == 1:
                    st.markdown(
                        f"""
                        <div class="result-positive">
                            <h2 style="margin:0; color:#9f1239;">Berisiko PCOS</h2>
                            <p style="margin:.45rem 0 0 0;">Skor keputusan model: <b>{score:.4f}</b>. Status margin: <b>{margin}</b>.</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="result-negative">
                            <h2 style="margin:0; color:#166534;">Tidak Berisiko PCOS Berdasarkan Model</h2>
                            <p style="margin:.45rem 0 0 0;">Skor keputusan model: <b>{score:.4f}</b>. Status margin: <b>{margin}</b>.</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            with score_col:
                st.markdown(
                    """
                    <div class="glass-card">
                        <h3 style="margin-top:0; color:#3b0764;">Cara baca skor</h3>
                        <p class="small-muted">
                            Skor positif cenderung menuju kelas PCOS. Skor negatif cenderung menuju kelas tidak PCOS.
                            Semakin jauh dari nol, margin keputusan model semakin kuat.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown('<div class="section-title">Kontribusi Fitur pada Prediksi Ini</div>', unsafe_allow_html=True)
            coef = model.coef_[0]
            contributions = X_input.iloc[0].values * coef
            contrib_df = pd.DataFrame(
                {
                    "Fitur": SELECTED_FEATURES,
                    "Nilai setelah scaling": X_input.iloc[0].values,
                    "Koefisien": coef,
                    "Kontribusi": contributions,
                    "Abs Kontribusi": np.abs(contributions),
                }
            ).sort_values("Abs Kontribusi", ascending=False)

            ctbl, cplot = st.columns([1.15, 1])
            with ctbl:
                st.dataframe(contrib_df.head(12), use_container_width=True, hide_index=True)
            with cplot:
                st.pyplot(
                    plot_barh(
                        contrib_df.rename(columns={"Fitur": "Feature"}),
                        "Abs Kontribusi",
                        "Feature",
                        "Top Kontribusi Fitur",
                        top_n=12,
                    )
                )
            st.caption("Kontribusi fitur menjelaskan mekanisme keputusan model linear, bukan hubungan sebab-akibat medis.")

    with tab_batch:
        st.markdown('<div class="section-title">Prediksi Batch Menggunakan CSV</div>', unsafe_allow_html=True)
        st.caption("Gunakan mode ini untuk memprediksi banyak baris data sekaligus.")

        template_df = build_csv_template()
        st.download_button(
            label="⬇️ Download Template CSV",
            data=template_df.to_csv(index=False).encode("utf-8"),
            file_name="template_prediksi_pcos.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.markdown(
            """
            <div class="csv-rules-card">
                <b>Ketentuan CSV:</b><br>
                1. File CSV harus mengikuti format kolom pada template fitur.<br>
                2. Nama kolom harus sama dengan template agar dapat terbaca oleh sistem.<br>
                3. Nilai kategori Ya/Tidak ditulis dalam bentuk numerik, yaitu 1 untuk Ya dan 0 untuk Tidak.<br>
                4. Jika terdapat nilai kosong, sistem akan mengisinya menggunakan median sesuai proses training model.
            </div>
            """,
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader("Upload file CSV", type=["csv"], key="batch_csv_upload")
        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                batch_df.columns = batch_df.columns.astype(str).str.strip()
            except Exception as exc:
                st.error(f"File CSV gagal dibaca: {exc}")
                batch_df = None

            if batch_df is not None:
                st.markdown('<div class="section-title">Preview Data Upload</div>', unsafe_allow_html=True)
                st.dataframe(batch_df.head(10), use_container_width=True, hide_index=True)

                b1, b2, b3 = st.columns(3)
                with b1:
                    metric_card("Jumlah Baris", str(len(batch_df)), "Data upload")
                with b2:
                    metric_card("Jumlah Kolom", str(batch_df.shape[1]), "Kolom CSV")
                with b3:
                    matched_cols = len([c for c in batch_df.columns if c in FEATURE_COLUMNS])
                    metric_card("Kolom Cocok", str(matched_cols), "Sesuai fitur training")

                X_batch, missing_columns, missing_selected, extra_columns = preprocess_batch_dataframe(batch_df)
                if missing_selected:
                    st.warning(
                        "Ada fitur final yang tidak ditemukan pada CSV. Sistem tetap memprediksi dengan nilai imputasi median: "
                        + ", ".join(missing_selected)
                    )
                elif missing_columns:
                    st.info(
                        "Beberapa fitur training nonfinal tidak ada pada CSV dan akan diisi melalui median imputer: "
                        + ", ".join(missing_columns[:8])
                        + ("..." if len(missing_columns) > 8 else "")
                    )
                if extra_columns:
                    st.caption(
                        "Kolom tambahan di luar fitur training tetap dipertahankan pada hasil unduhan: "
                        + ", ".join(extra_columns[:8])
                        + ("..." if len(extra_columns) > 8 else "")
                    )

                if st.button("🚀 Jalankan Prediksi Batch", use_container_width=True):
                    predictions = model.predict(X_batch).astype(int)
                    scores = model.decision_function(X_batch).astype(float)

                    result_df = batch_df.copy()
                    result_df["Prediksi"] = predictions
                    result_df["Hasil_Prediksi"] = np.where(
                        predictions == 1,
                        "Berisiko PCOS",
                        "Tidak Berisiko PCOS Berdasarkan Model",
                    )
                    result_df["Skor_Keputusan"] = np.round(scores, 6)
                    result_df["Status_Margin"] = [classify_margin(float(score)) for score in scores]

                    st.markdown('<div class="section-title">Ringkasan Hasil Batch</div>', unsafe_allow_html=True)
                    total = len(result_df)
                    pcos_count = int((predictions == 1).sum())
                    non_pcos_count = int((predictions == 0).sum())
                    avg_score = float(np.mean(scores)) if total else 0.0

                    r1, r2, r3, r4 = st.columns(4)
                    with r1:
                        metric_card("Total Data", str(total), "Baris diproses")
                    with r2:
                        metric_card("Berisiko PCOS", str(pcos_count), "Prediksi model")
                    with r3:
                        metric_card("Tidak Berisiko", str(non_pcos_count), "Prediksi model")
                    with r4:
                        metric_card("Rerata Skor", f"{avg_score:.3f}", "Decision function")

                    st.markdown('<div class="section-title">Tabel Hasil Prediksi</div>', unsafe_allow_html=True)
                    st.dataframe(result_df, use_container_width=True, hide_index=True)

                    csv_result = result_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="⬇️ Download Hasil Prediksi CSV",
                        data=csv_result,
                        file_name="hasil_prediksi_batch_pcos.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )


elif page == "Informasi Model":
    st.markdown(
        """
        <div class="hero">
            <h1>Informasi Model</h1>
            <p>Menampilkan ringkasan dataset, EDA, preprocessing, evaluasi performa, dan interpretasi model klasifikasi PCOS.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_dataset, tab_eda, tab_prep, tab_eval, tab_interpret = st.tabs(
        ["Dataset", "EDA", "Preprocessing", "Evaluasi", "Interpretasi"]
    )

    with tab_dataset:
        st.markdown('<div class="section-title-center">Dataset Penelitian</div>', unsafe_allow_html=True)
        d1, d2, d3, d4 = st.columns(4, gap="large")
        with d1:
            metric_card("Total Data", str(metrics["total_rows"]), "Observasi")
        with d2:
            metric_card("Fitur Awal", str(metrics["original_feature_count"]), "Sebelum Genetic Algorithm")
        with d3:
            metric_card("Fitur Final", str(metrics["selected_feature_count"]), "Setelah Genetic Algorithm")
        with d4:
            metric_card("Data Uji", str(metrics["test_rows"]), "20 persen")

        st.markdown("<div style='height: 32px;'></div>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1], gap="large")
        with col1:
            st.dataframe(tables["summary"], use_container_width=True, hide_index=True)
        with col2:
            target_counts = metrics["target_distribution"]
            target_df = pd.DataFrame(
                {
                    "Kelas": ["Tidak PCOS", "PCOS"],
                    "Jumlah": [target_counts.get("0", 0), target_counts.get("1", 0)],
                }
            )
            fig, ax = plt.subplots(figsize=(5, 3.4))
            ax.bar(target_df["Kelas"], target_df["Jumlah"])
            ax.set_title("Distribusi Target")
            ax.set_ylabel("Jumlah")
            fig.tight_layout()
            st.pyplot(fig)

        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-title-center">Fitur Terpilih Genetic Algorithm</div>', unsafe_allow_html=True)
        st.dataframe(
            pd.DataFrame({"Fitur Terpilih": SELECTED_FEATURES}),
            use_container_width=True,
            hide_index=True,
        )

    with tab_eda:
        st.markdown('<div class="section-title-center">Exploratory Data Analysis Visual</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="glass-card">
                EDA digunakan untuk memahami pola awal dataset, distribusi target, missing value,
                perbandingan fitur antar kelas, korelasi antarfitur, dan hubungan antarvariabel penting.
            </div>
            """,
            unsafe_allow_html=True,
        )
        eda_df = prepare_eda_dataframe()
        if eda_df is None:
            st.error("Kolom target PCOS (Y/N) tidak ditemukan pada dataset.")
        else:
            available_features = [feat for feat in SELECTED_FEATURES if feat in eda_df.columns]
            y_eda = eda_df[TARGET]
            st.markdown('<div class="section-title-center">Overview Komposisi Data</div>', unsafe_allow_html=True)
            v1, v2 = st.columns([0.92, 1.08], gap="large")
            with v1:
                eda_card("Donut Chart Target", "Menampilkan proporsi kelas PCOS dan tidak PCOS pada dataset.")
                st.plotly_chart(target_donut_chart(y_eda), use_container_width=True)
            with v2:
                eda_card("Missing Value Fitur Model", "Menampilkan fitur model yang memiliki nilai kosong sebelum proses imputasi.")
                miss_fig = missing_value_chart(eda_df, available_features)
                if miss_fig is None:
                    st.success("Tidak ditemukan missing value pada fitur hasil seleksi Genetic Algorithm.")
                else:
                    st.plotly_chart(miss_fig, use_container_width=True)

            st.markdown('<div class="section-title-center">Distribusi dan Perbandingan Fitur</div>', unsafe_allow_html=True)
            if len(available_features) > 0:
                default_feature = "AMH(ng/mL)" if "AMH(ng/mL)" in available_features else available_features[0]
                feature_choice = st.selectbox(
                    "Pilih fitur untuk melihat histogram dan boxplot",
                    options=available_features,
                    index=available_features.index(default_feature),
                    key="eda_feature_choice",
                )
                h1, h2 = st.columns(2, gap="large")
                with h1:
                    eda_card("Histogram dan Boxplot", "Melihat sebaran nilai fitur berdasarkan kelas PCOS dan tidak PCOS.")
                    st.plotly_chart(numeric_histogram(eda_df, feature_choice), use_container_width=True)
                with h2:
                    eda_card("Boxplot Per Kelas", "Membandingkan median, rentang nilai, dan outlier antara kelas PCOS dan tidak PCOS.")
                    st.plotly_chart(grouped_boxplot(eda_df, feature_choice), use_container_width=True)

                st.markdown('<div class="section-title-center">Fitur Paling Membedakan Kelas</div>', unsafe_allow_html=True)
                eda_card(
                    "Ranking Selisih Rata-rata Terstandar",
                    "Grafik ini menunjukkan fitur dengan perbedaan rata-rata paling menonjol antara kelas PCOS dan tidak PCOS.",
                )
                diff_fig, diff_table = top_feature_difference(eda_df, available_features, top_n=12)
                if diff_fig is not None:
                    st.plotly_chart(diff_fig, use_container_width=True)
                    with st.expander("Lihat tabel detail selisih fitur"):
                        st.dataframe(diff_table, use_container_width=True, hide_index=True)
                else:
                    st.info("Data belum cukup untuk menghitung perbedaan rata-rata fitur antar kelas.")

                st.markdown('<div class="section-title-center">Korelasi Antar Fitur Pilihan</div>', unsafe_allow_html=True)
                default_corr = [
                    feat for feat in [
                        "Age (yrs)", "Weight (Kg)", "Cycle length(days)", "FSH(mIU/mL)",
                        "LH(mIU/mL)", "FSH/LH", "AMH(ng/mL)", "Weight gain(Y/N)",
                        "hair growth(Y/N)", "Follicle No. (L)", "Follicle No. (R)", "Endometrium (mm)",
                    ] if feat in available_features
                ]
                if len(default_corr) < 4:
                    default_corr = available_features[: min(12, len(available_features))]
                corr_features = st.multiselect(
                    "Pilih fitur untuk heatmap korelasi",
                    options=available_features,
                    default=default_corr,
                    key="eda_corr_features",
                )
                if len(corr_features) < 2:
                    st.warning("Pilih minimal 2 fitur untuk menampilkan heatmap korelasi.")
                else:
                    st.plotly_chart(correlation_heatmap(eda_df, corr_features), use_container_width=True)

                st.markdown('<div class="section-title-center">Scatter Plot Hubungan Antarvariabel</div>', unsafe_allow_html=True)
                s1, s2 = st.columns(2, gap="large")
                default_x = "Follicle No. (L)" if "Follicle No. (L)" in available_features else available_features[0]
                default_y = "Follicle No. (R)" if "Follicle No. (R)" in available_features else available_features[min(1, len(available_features) - 1)]
                with s1:
                    x_feature = st.selectbox("Sumbu X", options=available_features, index=available_features.index(default_x), key="eda_x_feature")
                with s2:
                    y_feature = st.selectbox("Sumbu Y", options=available_features, index=available_features.index(default_y), key="eda_y_feature")
                if x_feature == y_feature:
                    st.info("Pilih dua fitur yang berbeda agar hubungan antarvariabel lebih terlihat.")
                else:
                    st.plotly_chart(scatter_relation(eda_df, x_feature, y_feature), use_container_width=True)
            else:
                st.warning("Tidak ada fitur hasil seleksi Genetic Algorithm yang ditemukan pada dataset.")

    with tab_prep:
        st.markdown('<div class="section-title-center">Tahapan Preprocessing</div>', unsafe_allow_html=True)
        p1, p2, p3 = st.columns(3, gap="large")
        with p1:
            step_card(1, "Load Data", "Dataset PCOS dimuat dari file Excel, lalu nama kolom dirapikan.")
        with p2:
            step_card(2, "Drop Kolom", "Kolom identitas dan kolom tidak relevan dihapus dari dataset.")
        with p3:
            step_card(3, "Hapus Duplikat", "Data duplikat diperiksa dan dihapus supaya tidak memengaruhi proses pembentukan model.")
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        p4, p5, p6 = st.columns(3, gap="large")
        with p4:
            step_card(4, "Cleaning Nilai", "Nilai kosong, simbol tidak valid, dan format data yang tidak sesuai dibersihkan sebelum diproses lebih lanjut.")
        with p5:
            step_card(5, "Cek Missing Value", "Missing value pada data diperiksa sebelum dilakukan penanganan pada tahap imputasi.")
        with p6:
            step_card(6, "Target & Numerik", "Kolom PCOS (Y/N) ditetapkan sebagai target, lalu seluruh fitur diubah ke bentuk numerik.")
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        p7, p8, p9 = st.columns(3, gap="large")
        with p7:
            step_card(7, "Split Data", "Data dibagi menjadi data latih dan data uji dengan rasio 80:20 menggunakan stratify agar distribusi kelas tetap seimbang.")
        with p8:
            step_card(8, "Imputasi Median", "Missing value pada fitur ditangani menggunakan SimpleImputer dengan strategi median yang di-fit hanya pada data latih.")
        with p9:
            step_card(9, "Standard Scaling", "Fitur distandardisasi menggunakan StandardScaler agar setiap fitur memiliki skala yang sebanding sebelum masuk ke model.")

    with tab_eval:
        st.markdown('<div class="section-title-center">Metrik Model Final</div>', unsafe_allow_html=True)
        e1, e2, e3, e4, e5 = st.columns(5)
        with e1:
            metric_card("Accuracy", f"{metrics['accuracy']:.4f}", "Overall")
        with e2:
            metric_card("Precision", f"{metrics['precision']:.4f}", "PCOS")
        with e3:
            metric_card("Recall", f"{metrics['recall']:.4f}", "PCOS")
        with e4:
            metric_card("F1-score", f"{metrics['f1_score']:.4f}", "PCOS")
        with e5:
            metric_card("ROC-AUC", f"{metrics['roc_auc']:.4f}", "Model")

        st.markdown("<div style='height: 32px;'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-title-center">Confusion Matrix</div>', unsafe_allow_html=True)
        cm_left, cm_mid, cm_right = st.columns([0.18, 0.64, 0.18])
        with cm_mid:
            st.pyplot(plot_confusion_matrix(metrics["confusion_matrix"]), use_container_width=True)

        st.markdown("<div style='height: 36px;'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-title-center">Tabel Perbandingan Evaluasi Model</div>', unsafe_allow_html=True)
        st.markdown("Bayesian Optimization SVM + GA")
        st.dataframe(tables["bayes"], use_container_width=True, hide_index=True)
        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
        st.markdown("Baseline SVM")
        st.dataframe(tables["baseline"], use_container_width=True, hide_index=True)
        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
        st.markdown("SVM + GA sebelum Bayesian Optimization")
        st.dataframe(tables["ga"], use_container_width=True, hide_index=True)

    with tab_interpret:
        st.markdown('<div class="section-title-center">Interpretasi SVM Linear</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="glass-card">
                Pada SVM kernel linear, koefisien dapat digunakan untuk melihat kontribusi relatif fitur terhadap batas keputusan model.
                Nilai koefisien yang besar secara absolut menunjukkan fitur yang lebih kuat memengaruhi keputusan model.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        coef_df = tables["coef"]
        itbl, iplot = st.columns([1.15, 1], gap="large")
        with itbl:
            st.dataframe(coef_df, use_container_width=True, hide_index=True)
        with iplot:
            st.pyplot(plot_barh(coef_df, "Abs Coefficient", "Feature", "Top Koefisien Absolut SVM Linear", top_n=15))


elif page == "Edukasi PCOS":
    st.markdown(
        """
        <div class="hero">
            <h1>Edukasi Singkat PCOS</h1>
            <p>Informasi ringkas untuk membantu pengguna memahami konteks hasil prediksi.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title-center">Data Singkat PCOS</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="edu-source-note" style="text-align:center;">
            Data ringkas ini disajikan untuk tujuan edukasi. Angka dapat berbeda tergantung populasi,
            metode penelitian, dan kriteria diagnosis yang digunakan.
        </div>
        """,
        unsafe_allow_html=True,
    )

    d1, d2 = st.columns(2, gap="large")
    with d1:
        prevalence_df = pd.DataFrame({"Kategori": ["WHO", "Meta-analisis 2024"], "Persentase": [11.5, 9.2]})
        fig_bar = px.bar(prevalence_df, x="Kategori", y="Persentase", text="Persentase", title="Prevalensi PCOS")
        fig_bar.update_traces(texttemplate="%{text}%", textposition="outside")
        fig_bar.update_layout(yaxis_title="Persentase (%)", xaxis_title="", height=400, margin=dict(l=30, r=20, t=60, b=30))
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown(
            """
            <div class="edu-source-note">
                Sumber:
                <a href="https://www.who.int/news-room/fact-sheets/detail/polycystic-ovary-syndrome" target="_blank">
                    WHO Fact Sheet PCOS
                </a>
                menyebut PCOS diperkirakan terjadi pada 10–13% perempuan usia reproduktif.
                Nilai 11,5% pada grafik merupakan nilai tengah dari rentang tersebut.
                Data meta-analisis 2024 mengacu pada
                <a href="https://pubmed.ncbi.nlm.nih.gov/38922413/" target="_blank">Salari et al. (2024)</a>
                yang memperkirakan prevalensi global PCOS sebesar 9,2%.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with d2:
        undiagnosed_df = pd.DataFrame({"Status": ["Belum Terdiagnosis", "Sudah/terdeteksi"], "Jumlah": [70, 30]})
        fig_donut = px.pie(
            undiagnosed_df,
            names="Status",
            values="Jumlah",
            hole=0.6,
            title="Kasus PCOS yang Belum Terdiagnosis",
            color="Status",
            color_discrete_map={"Belum Terdiagnosis": "#0d6ec9", "Sudah/terdeteksi": "#7ec3f7"},
        )
        fig_donut.update_traces(textinfo="percent", textposition="inside", insidetextfont=dict(size=16, color="white"))
        fig_donut.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=60, b=20),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig_donut, use_container_width=True)
        st.markdown(
            """
            <div class="edu-source-note">
                Sumber:
                <a href="https://www.who.int/news-room/fact-sheets/detail/polycystic-ovary-syndrome" target="_blank">
                    WHO Fact Sheet PCOS
                </a>
                menyebut hingga 70% perempuan dengan PCOS di dunia belum mengetahui bahwa mereka memiliki kondisi ini.
                Bagian 30% pada diagram digunakan sebagai pelengkap visual dari estimasi tersebut.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 36px;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-title-center">Ringkasan Edukasi PCOS</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown(
            """
            <div class="edu-card">
                <h3>Apa itu PCOS?</h3>
                <p>
                    <em>Polycystic Ovary Syndrome</em> atau PCOS adalah gangguan hormonal yang dapat terjadi pada perempuan
                    usia reproduktif. Kondisi ini berkaitan dengan ketidakseimbangan hormon, gangguan ovulasi,
                    siklus menstruasi tidak teratur, serta perubahan metabolik tubuh.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="edu-card">
                <h3>Gejala yang sering dikaitkan</h3>
                <p>
                    Gejala PCOS dapat berupa siklus menstruasi tidak teratur, gangguan ovulasi, jerawat,
                    pertumbuhan rambut berlebih, rambut rontok, kesulitan hamil, kenaikan berat badan,
                    serta gangguan metabolik seperti resistensi insulin.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div class='edu-row-space'></div>", unsafe_allow_html=True)
    c3, c4 = st.columns(2, gap="large")
    with c3:
        st.markdown(
            """
            <div class="edu-card">
                <h3>Apa dampak dari PCOS?</h3>
                <p>
                    PCOS dapat berdampak pada aspek reproduksi, metabolik, dan psikologis. Beberapa dampaknya
                    meliputi infertilitas, resistensi insulin, obesitas, diabetes melitus tipe II, risiko
                    penyakit kardiovaskular, gangguan kecemasan, depresi, dan penurunan kualitas hidup.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            """
            <div class="edu-card">
                <h3>Apa faktor penyebab PCOS?</h3>
                <p>
                    PCOS tidak memiliki satu penyebab tunggal. Kondisi ini dapat dipengaruhi oleh kombinasi
                    faktor hormonal, metabolik, genetik, dan lingkungan. Riwayat keluarga PCOS atau diabetes tipe II
                    juga dapat meningkatkan risiko seseorang mengalami PCOS.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 34px;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-title-center">Mitos dan Fakta PCOS</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="edu-source-note" style="text-align:center;">
            Klik kartu mitos untuk melihat penjelasan faktanya.
        </div>
        """,
        unsafe_allow_html=True,
    )

    myth_fact_items = [
        {
            "id": "myth_1",
            "myth": "PCOS hanya terjadi pada perempuan obesitas.",
            "fact": "PCOS bisa terjadi pada berbagai kondisi berat badan. Obesitas dapat memperburuk risiko metabolik, tetapi bukan satu-satunya penyebab.",
            "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC9600591/",
            "source": "(Carmina et al., 2022)",
        },
        {
            "id": "myth_2",
            "myth": "Menstruasi tidak teratur pasti PCOS.",
            "fact": "Menstruasi tidak teratur bisa menjadi tanda, tetapi belum tentu PCOS. Diagnosis PCOS tetap perlu mempertimbangkan kriteria klinis lain.",
            "url": "https://www.monash.edu/__data/assets/pdf_file/0003/3371133/PCOS-Guideline-Summary-2023.pdf",
            "source": "(Teede et al., 2023)",
        },
        {
            "id": "myth_3",
            "myth": "PCOS pasti membuat perempuan tidak bisa hamil.",
            "fact": "PCOS dapat mengganggu ovulasi, tetapi bukan berarti pasti tidak bisa hamil. Peluang kehamilan tetap dapat ditangani melalui evaluasi dan terapi yang sesuai.",
            "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC4175743/",
            "source": "(Legro et al., 2014)",
        },
        {
            "id": "myth_4",
            "myth": "PCOS hanya masalah menstruasi.",
            "fact": "PCOS juga berkaitan dengan hormon, resistensi insulin, gangguan metabolisme, serta kesehatan reproduksi.",
            "url": "https://academic.oup.com/humrep/article/38/9/1655/7241786",
            "source": "(Teede et al., 2023)",
        },
        {
            "id": "myth_5",
            "myth": "PCOS pasti terlihat dari gejala fisik.",
            "fact": "Gejala PCOS bisa berbeda-beda pada setiap orang. Ada yang menunjukkan gejala jelas, tetapi ada juga yang gejalanya tidak terlalu tampak.",
            "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC6683693/",
            "source": "(Sachdeva et al., 2019)",
        },
        {
            "id": "myth_6",
            "myth": "PCOS disebabkan oleh satu faktor saja.",
            "fact": "PCOS bersifat kompleks dan dapat dipengaruhi faktor hormonal, metabolik, genetik, lingkungan, serta gaya hidup.",
            "url": "https://pubmed.ncbi.nlm.nih.gov/36429632/",
            "source": "(Mirza et al., 2022)",
        },
    ]

    row1 = st.columns(3, gap="large")
    for col, item in zip(row1, myth_fact_items[:3]):
        with col:
            myth_fact_card(item["id"], item["myth"], item["fact"], item["url"], item["source"])

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    row2 = st.columns(3, gap="large")
    for col, item in zip(row2, myth_fact_items[3:]):
        with col:
            myth_fact_card(item["id"], item["myth"], item["fact"], item["url"], item["source"])

    st.markdown("<div style='height: 32px;'></div>", unsafe_allow_html=True)
