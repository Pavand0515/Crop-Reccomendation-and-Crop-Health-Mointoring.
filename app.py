from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from disease_detection import load_disease_artifacts, predict_disease

warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = MODELS_DIR / "outputs"
FORECAST_DIR = MODELS_DIR / "forecasting_models"

CROP_NAME_MAP = {
    "apple": "Apple",
    "banana": "Banana",
    "blackgram": "Black Gram Dal(Urd Dal)",
    "chickpea": "Lentil(Masur)(Whole)",
    "coconut": "Coconut",
    "coffee": "Coffee",
    "cotton": "Cotton",
    "grapes": "Grapes",
    "jute": "Jute",
    "kidneybeans": "Pegeon Pea(Arhar Fali)",
    "lentil": "Lentil(Masur)(Whole)",
    "maize": "Maize",
    "mango": "Mango",
    "mothbeans": "Pegeon Pea(Arhar Fali)",
    "mungbean": "Black Gram Dal(Urd Dal)",
    "muskmelon": "Water Melon",
    "orange": "Orange",
    "papaya": "Papaya",
    "pigeonpeas": "Pegeon Pea(Arhar Fali)",
    "pomegranate": "Pomegranate",
    "rice": "Rice",
    "watermelon": "Water Melon",
}

MONTH_BY_SEASON = {"Kharif": 7, "Rabi": 1, "Zaid": 4}

st.set_page_config(
    page_title="AgriIntel Dashboard",
    page_icon="AI",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_CSS = """
<style>
    :root {
        --app-bg: #f7f8f5;
        --ink: #17231d;
        --muted: #66736c;
        --panel: #ffffff;
        --panel-2: #f1f5f2;
        --line: #dbe3dd;
        --green: #236b45;
        --green-dark: #153d2b;
        --blue: #315f8f;
        --gold: #b47b20;
        --danger: #9c3c2f;
        --shadow: 0 14px 32px rgba(27, 43, 34, 0.08);
    }

    .stApp {
        background: var(--app-bg);
        color: var(--ink);
        font-family: "Inter", "Segoe UI", Arial, sans-serif;
    }

    .block-container {
        max-width: 1360px;
        padding: 1.25rem 1.6rem 3rem;
    }

    h1, h2, h3 {
        color: var(--ink);
        letter-spacing: 0;
        font-family: "Inter", "Segoe UI", Arial, sans-serif;
    }

    [data-testid="stSidebar"] {
        background: #10251b;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    [data-testid="stSidebar"] * {
        color: #f5faf7;
    }

    [data-testid="stSidebar"] [role="radiogroup"] {
        gap: .35rem;
    }

    [data-testid="stSidebar"] [data-baseweb="radio"] {
        padding: .62rem .72rem;
        border: 1px solid rgba(255, 255, 255, .10);
        background: rgba(255, 255, 255, .055);
        border-radius: 8px;
    }

    .topbar {
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
        gap: 1rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid var(--line);
        margin-bottom: 1rem;
    }

    .brand-kicker {
        color: var(--green);
        font-size: .78rem;
        font-weight: 800;
        letter-spacing: .08em;
        text-transform: uppercase;
        margin-bottom: .2rem;
    }

    .topbar h1 {
        margin: 0;
        font-size: clamp(1.9rem, 3vw, 3rem);
        line-height: 1.05;
        font-weight: 850;
    }

    .topbar p {
        margin: .55rem 0 0;
        max-width: 760px;
        color: var(--muted);
        font-size: 1rem;
        line-height: 1.55;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        white-space: nowrap;
        padding: .55rem .75rem;
        border-radius: 8px;
        background: #e8f1ec;
        color: var(--green-dark);
        border: 1px solid #cddbd2;
        font-weight: 750;
        font-size: .9rem;
    }

    .metric-card, .panel, .step-card, .callout {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        box-shadow: var(--shadow);
    }

    .metric-card {
        padding: 1rem;
        min-height: 112px;
    }

    .metric-label {
        color: var(--muted);
        font-size: .76rem;
        font-weight: 800;
        letter-spacing: .06em;
        text-transform: uppercase;
    }

    .metric-value {
        color: var(--ink);
        font-size: 1.75rem;
        line-height: 1.1;
        font-weight: 850;
        margin-top: .45rem;
        overflow-wrap: anywhere;
    }

    .metric-note {
        color: var(--muted);
        font-size: .84rem;
        margin-top: .35rem;
    }

    .section-title {
        margin: 1.45rem 0 .8rem;
    }

    .section-title h2 {
        font-size: 1.55rem;
        margin: 0;
        font-weight: 820;
    }

    .section-title p {
        margin: .35rem 0 0;
        color: var(--muted);
        max-width: 850px;
        line-height: 1.5;
    }

    .panel {
        padding: 1rem;
        margin-bottom: 1rem;
    }

    .panel h3 {
        margin: 0 0 .35rem;
        font-size: 1.05rem;
        font-weight: 800;
    }

    .panel p {
        margin: 0;
        color: var(--muted);
        line-height: 1.5;
    }

    .step-card {
        padding: 1rem;
        min-height: 158px;
    }

    .step-number {
        display: inline-flex;
        width: 2rem;
        height: 2rem;
        align-items: center;
        justify-content: center;
        border-radius: 8px;
        background: var(--green-dark);
        color: #fff;
        font-weight: 850;
        margin-bottom: .7rem;
    }

    .step-card h3 {
        margin: 0 0 .35rem;
        font-size: 1.02rem;
        font-weight: 820;
    }

    .step-card p {
        margin: 0;
        color: var(--muted);
        line-height: 1.48;
    }

    .callout {
        padding: .95rem 1rem;
        border-left: 5px solid var(--green);
        background: #fbfcfb;
        color: #2a3d32;
        margin: .8rem 0 1rem;
        line-height: 1.5;
    }

    .result-strip {
        border-radius: 8px;
        border: 1px solid #cbded2;
        background: #edf6f0;
        padding: 1rem;
        margin: 1rem 0;
        color: #163d2b;
        line-height: 1.55;
    }

    .result-strip strong {
        color: #0f2f20;
    }

    .info-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: .7rem;
    }

    .info-cell {
        background: var(--panel-2);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: .85rem;
    }

    .info-label {
        color: var(--muted);
        font-size: .74rem;
        font-weight: 800;
        letter-spacing: .05em;
        text-transform: uppercase;
    }

    .info-value {
        margin-top: .25rem;
        font-size: 1.05rem;
        font-weight: 820;
        color: var(--ink);
        overflow-wrap: anywhere;
    }

    [data-testid="stForm"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 1rem;
        box-shadow: var(--shadow);
    }

    .stButton > button {
        min-height: 2.85rem;
        border-radius: 8px;
        border: 1px solid var(--green-dark);
        background: var(--green);
        color: #fff;
        font-weight: 800;
    }

    .stButton > button:hover {
        border-color: var(--green-dark);
        background: var(--green-dark);
        color: #fff;
    }

    [data-testid="stTabs"] [data-baseweb="tab"] {
        height: 2.85rem;
        border-radius: 8px;
        padding: 0 .9rem;
        border: 1px solid var(--line);
        background: #fff;
        color: var(--ink);
    }

    [data-testid="stTabs"] [aria-selected="true"] {
        background: #153d2b;
        color: #fff;
    }

    .stDataFrame, .stTable {
        border-radius: 8px;
        overflow: hidden;
    }

    @media (max-width: 900px) {
        .topbar {
            display: block;
        }

        .status-pill {
            margin-top: .9rem;
        }

        .info-grid {
            grid-template-columns: 1fr;
        }
    }
</style>
"""


def infer_season(month: int) -> str:
    if month in [6, 7, 8, 9, 10, 11]:
        return "Kharif"
    if month in [11, 12, 1, 2, 3, 4]:
        return "Rabi"
    return "Zaid"


@st.cache_data(show_spinner=False)
def load_market_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "market_prices_real.csv")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Month"] = df["Date"].dt.month
    df["Quarter"] = df["Date"].dt.quarter
    df["Year"] = df["Date"].dt.year
    df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
    df["Season"] = df["Month"].apply(infer_season)
    return df.sort_values("Date")


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_fertilizer_data() -> pd.DataFrame:
    path = DATA_DIR / "fertilizer_recommendation_dataset.csv"
    df = pd.read_csv(path)
    df["Crop"] = df["Crop"].astype(str).str.strip().str.title()
    df["Soil"] = df["Soil"].astype(str).str.strip().str.title()
    return df


@st.cache_resource(show_spinner=False)
def load_assets() -> dict:
    assets = {
        "crop_model": joblib.load(MODELS_DIR / "crop_recommendation" / "rf.pkl"),
        "crop_scaler": joblib.load(MODELS_DIR / "crop_recommendation" / "scaler.pkl"),
        "crop_encoder": joblib.load(MODELS_DIR / "crop_recommendation" / "label_encoder.pkl"),
        "price_model": joblib.load(MODELS_DIR / "random_forest_model.pkl"),
        "price_scaler": joblib.load(MODELS_DIR / "scaler.pkl"),
        "encoders": joblib.load(MODELS_DIR / "encoders.pkl"),
        "metadata": json.loads((MODELS_DIR / "model_metadata.json").read_text(encoding="utf-8")),
    }

    fert_dir = MODELS_DIR / "fertilizer_recommendation"
    disease_dir = MODELS_DIR / "disease_detection"
    fert_data_path = DATA_DIR / "fertilizer_recommendation_dataset.csv"
    assets["fertilizer_available"] = fert_data_path.exists() or (fert_dir / "model.pkl").exists() or fert_dir.exists()
    assets["fertilizer_df"] = load_fertilizer_data() if fert_data_path.exists() else pd.DataFrame()
    assets["disease_available"] = False
    assets["disease_error"] = None

    disease_model_exists = (
        (disease_dir / "model.pkl").exists()
        or (disease_dir / "model.keras").exists()
        or (disease_dir / "model.weights.h5").exists()
    )
    if disease_model_exists and (disease_dir / "metadata.json").exists():
        try:
            assets["disease"] = load_disease_artifacts(disease_dir)
            assets["disease_available"] = True
        except Exception as exc:
            assets["disease_error"] = str(exc)
    return assets


def encode_value(encoders: dict, key: str, value: str) -> int:
    encoder = encoders[key]
    if value not in set(encoder.classes_):
        value = encoder.classes_[0]
    return int(encoder.transform([value])[0])


def metric_card(label: str, value: str, note: str = "") -> None:
    note_html = f'<div class="metric-note">{note}</div>' if note else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {note_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="section-title">
            <h2>{title}</h2>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def panel(title: str, text: str) -> None:
    st.markdown(
        f"""
        <div class="panel">
            <h3>{title}</h3>
            <p>{text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def step_card(number: int, title: str, text: str) -> None:
    st.markdown(
        f"""
        <div class="step-card">
            <div class="step-number">{number}</div>
            <h3>{title}</h3>
            <p>{text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def callout(text: str) -> None:
    st.markdown(f'<div class="callout">{text}</div>', unsafe_allow_html=True)


def info_grid(items: list[tuple[str, str]]) -> None:
    cells = "".join(
        f"""
        <div class="info-cell">
            <div class="info-label">{label}</div>
            <div class="info-value">{value}</div>
        </div>
        """
        for label, value in items
    )
    st.markdown(f'<div class="info-grid">{cells}</div>', unsafe_allow_html=True)


def result_strip(html: str) -> None:
    st.markdown(f'<div class="result-strip">{html}</div>', unsafe_allow_html=True)


def build_feature_row(
    market_df: pd.DataFrame,
    encoders: dict,
    crop: str,
    state: str,
    season: str,
    month: int,
    year: int,
) -> tuple[pd.DataFrame, pd.Series]:
    scoped = market_df[
        (market_df["Crop"] == crop)
        & (market_df["State"] == state)
        & (market_df["Season"] == season)
    ].copy()
    if scoped.empty:
        scoped = market_df[(market_df["Crop"] == crop) & (market_df["State"] == state)].copy()
    if scoped.empty:
        scoped = market_df[market_df["Crop"] == crop].copy()
    if scoped.empty:
        scoped = market_df.copy()

    scoped = scoped.sort_values("Date")
    row = scoped.iloc[-1]
    rolling7 = float(scoped["Modal_Price"].tail(7).mean())
    rolling14 = float(scoped["Modal_Price"].tail(14).mean())
    lag1 = float(scoped["Modal_Price"].iloc[-1])

    feature_values = {
        "Crop_enc": encode_value(encoders, "Crop", crop),
        "State_enc": encode_value(encoders, "State", state),
        "Variety_enc": encode_value(encoders, "Variety", str(row["Variety"])),
        "Grade_enc": encode_value(encoders, "Grade", str(row["Grade"])),
        "District_enc": encode_value(encoders, "District", str(row["District"])),
        "Commodity_Code": int(row["Commodity_Code"]),
        "Month": int(month),
        "Quarter": int(((month - 1) // 3) + 1),
        "Season_enc": encode_value(encoders, "Season", season),
        "Year": int(year),
        "Week": int(pd.Timestamp(year=year, month=month, day=1).isocalendar().week),
        "Price_Trend_30d": float(row["Price_Trend_30d"]),
        "Trend_Label_enc": encode_value(encoders, "Trend_Label", str(row["Trend_Label"])),
        "Cultivation_Cost": float(row["Cultivation_Cost"]),
        "Avg_Yield_Qtl": max(
            float((lag1 * (1 + float(row["Profit_Margin"]) / 100.0) * 10) / max(float(row["Cultivation_Cost"]), 1)),
            1.0,
        ),
        "Rolling_7d_Avg": rolling7,
        "Rolling_14d_Avg": rolling14,
        "Lag_1d_Price": lag1,
    }
    return pd.DataFrame([feature_values]), row


def predict_crop_recommendations(assets: dict, sample: list[float]) -> pd.DataFrame:
    scaled = assets["crop_scaler"].transform([sample])
    probs = assets["crop_model"].predict_proba(scaled)[0]
    labels = assets["crop_encoder"].inverse_transform(np.arange(len(probs)))
    result = pd.DataFrame({"Crop": labels, "Confidence": probs})
    return result.sort_values("Confidence", ascending=False).head(6).reset_index(drop=True)


def recommend_fertilizer(df: pd.DataFrame, sample: dict, top_n: int = 3) -> pd.DataFrame:
    search = df.copy()
    numeric_cols = ["Temperature", "Moisture", "Rainfall", "PH", "Nitrogen", "Phosphorous", "Potassium", "Carbon"]
    sample_crop = str(sample.get("Crop", "")).strip().title()
    sample_soil = str(sample.get("Soil", "")).strip().title()

    filtered = search[(search["Crop"] == sample_crop) & (search["Soil"] == sample_soil)]
    if filtered.empty:
        filtered = search[search["Crop"] == sample_crop]
    if filtered.empty:
        filtered = search[search["Soil"] == sample_soil]
    if filtered.empty:
        filtered = search

    metrics = filtered[numeric_cols].astype(float)
    min_vals = metrics.min()
    range_vals = (metrics.max() - min_vals).replace(0, 1)
    normalized_rows = (metrics - min_vals) / range_vals
    sample_values = pd.Series(
        [
            float(sample.get("Temperature", 0.0)),
            float(sample.get("Moisture", 0.0)),
            float(sample.get("Rainfall", 0.0)),
            float(sample.get("PH", 0.0)),
            float(sample.get("Nitrogen", 0.0)),
            float(sample.get("Phosphorous", 0.0)),
            float(sample.get("Potassium", 0.0)),
            float(sample.get("Carbon", 0.0)),
        ],
        index=numeric_cols,
    )
    normalized_sample = (sample_values - min_vals) / range_vals
    distances = ((normalized_rows - normalized_sample) ** 2).sum(axis=1).pow(0.5)
    filtered = filtered.copy()
    filtered["Distance"] = distances
    recommended = (
        filtered.sort_values("Distance", ascending=True)
        .drop_duplicates(subset=["Fertilizer"])
        .head(top_n)
    )
    return recommended[["Crop", "Soil", "Fertilizer", "Remark", "Distance"]]


def predict_price(
    assets: dict,
    market_df: pd.DataFrame,
    crop: str,
    state: str,
    season: str,
    month: int,
    year: int,
) -> tuple[float, pd.Series]:
    features, context_row = build_feature_row(market_df, assets["encoders"], crop, state, season, month, year)
    features = features[assets["metadata"]["features"]]
    scaled = assets["price_scaler"].transform(features)
    prediction = float(assets["price_model"].predict(scaled)[0])
    return prediction, context_row


def to_market_crop_name(crop_label: str) -> str:
    return CROP_NAME_MAP.get(crop_label.strip().lower(), crop_label.title())


def normalize_series(values: pd.Series) -> pd.Series:
    if values.max() == values.min():
        return pd.Series([1.0] * len(values), index=values.index)
    return (values - values.min()) / (values.max() - values.min())


def build_ranked_recommendations(
    assets: dict,
    market_df: pd.DataFrame,
    crop_candidates: pd.DataFrame,
    state: str,
    season: str,
    month: int,
    year: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    for _, rec in crop_candidates.iterrows():
        market_crop = to_market_crop_name(str(rec["Crop"]))
        predicted_price, ctx = predict_price(assets, market_df, market_crop, state, season, month, year)
        indicative_yield = max(
            float((float(ctx["Cultivation_Cost"]) * (1 + float(ctx["Profit_Margin"]) / 100.0)) / max(float(ctx["Modal_Price"]), 1)),
            1.0,
        )
        estimated_revenue = predicted_price * indicative_yield
        estimated_profit = estimated_revenue - float(ctx["Cultivation_Cost"])
        rows.append(
            {
                "Crop": market_crop,
                "Base Crop Label": str(rec["Crop"]),
                "Probability": float(rec["Confidence"]) * 100,
                "Predicted Price": float(predicted_price),
                "Latest Price": float(ctx["Modal_Price"]),
                "Trend": str(ctx["Trend_Label"]),
                "Cultivation Cost": float(ctx["Cultivation_Cost"]),
                "Indicative Margin %": float(ctx["Profit_Margin"]),
                "Estimated Yield": indicative_yield,
                "Estimated Revenue": estimated_revenue,
                "Estimated Profit": estimated_profit,
            }
        )

    ranked = pd.DataFrame(rows).sort_values("Probability", ascending=False).reset_index(drop=True)
    ranked["Probability Gap To Leader"] = ranked.iloc[0]["Probability"] - ranked["Probability"]
    ranked["Price Score"] = normalize_series(ranked["Predicted Price"])
    ranked["Profit Score"] = normalize_series(ranked["Estimated Profit"])
    ranked["Probability Score"] = ranked["Probability"] / 100.0
    ranked["TieBandBoost"] = ranked["Probability Gap To Leader"].between(5, 9, inclusive="both").astype(float)
    ranked["Decision Score"] = (
        0.58 * ranked["Probability Score"]
        + 0.27 * ranked["Profit Score"]
        + 0.15 * ranked["Price Score"]
        + 0.05 * ranked["TieBandBoost"] * ranked["Profit Score"]
    )

    leader = ranked.iloc[0].copy()
    for idx in range(1, len(ranked)):
        challenger = ranked.iloc[idx]
        gap = float(leader["Probability"] - challenger["Probability"])
        more_profitable = float(challenger["Estimated Profit"]) > float(leader["Estimated Profit"])
        if 5 <= gap <= 9 and more_profitable:
            profit_ratio = challenger["Estimated Profit"] / max(abs(leader["Estimated Profit"]), 1.0)
            if profit_ratio >= 1.10:
                ranked.at[idx, "Decision Score"] += 0.08
                ranked.at[idx, "TieBreak Reason"] = "Close suitability; stronger profit outlook."
        elif gap < 5 and more_profitable:
            ranked.at[idx, "Decision Score"] += 0.04
            ranked.at[idx, "TieBreak Reason"] = "Very close suitability; better profit outlook."
        else:
            ranked.at[idx, "TieBreak Reason"] = "Suitability remains the main driver."

    ranked.loc[0, "TieBreak Reason"] = "Highest field suitability."
    ranked = ranked.sort_values(["Decision Score", "Probability", "Estimated Profit"], ascending=False).reset_index(drop=True)
    ranked["Rank"] = np.arange(1, len(ranked) + 1)
    return ranked.head(3)


def disease_save_snippet() -> str:
    return r"""python disease_detection.py --data-dir Data/mixed --model-dir models/disease_detection
streamlit run app.py"""


def load_top3_file(state: str, season: str) -> pd.DataFrame | None:
    file_name = f"top3_{state.replace(' ', '_')}_{season}.csv"
    path = OUTPUTS_DIR / file_name
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df


def sidebar(market_df: pd.DataFrame) -> str:
    st.sidebar.title("AgriIntel")
    return st.sidebar.radio(
        "Dashboard section",
        ["Overview", "Crop Advisor", "Fertilizer Recommendation", "Price Forecast", "Disease Scan"],
    )


def render_header(market_df: pd.DataFrame, assets: dict) -> None:
    st.title("AgriIntel")
    st.write("A compact crop planner for crop choice, price outlook, and disease scan.")
    col1, col2, col3 = st.columns(3)
    col1.metric("Market records", f"{len(market_df):,}")
    col2.metric("Crops", f"{market_df['Crop'].nunique()}")
    col3.metric("States", f"{market_df['State'].nunique()}")


def render_overview(market_df: pd.DataFrame, assets: dict) -> None:
    st.header("Overview")
    st.write("Review the dataset coverage and recent market context.")
    state_counts = market_df.groupby("State").size().sort_values(ascending=False).head(10).reset_index(name="Records")
    st.bar_chart(state_counts.set_index("State"))
    st.markdown("#### Summary")
    st.write(
        {
            "Latest year": int(market_df["Year"].max()),
            "States": int(market_df["State"].nunique()),
            "Unique crops": int(market_df["Crop"].nunique()),
        }
    )


def render_crop_advisor(market_df: pd.DataFrame, assets: dict) -> None:
    st.header("Crop Advisor")
    st.write("Enter farm conditions to see the top crop options.")

    with st.form("crop_advice_form"):
        left, right = st.columns(2)
        with left:
            nitrogen = st.number_input("Nitrogen (N)", min_value=0.0, value=90.0)
            phosphorus = st.number_input("Phosphorus (P)", min_value=0.0, value=42.0)
            potassium = st.number_input("Potassium (K)", min_value=0.0, value=43.0)
            temperature = st.number_input("Temperature (C)", value=20.88)
            humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=82.0)
        with right:
            ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5)
            rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=202.94)
            state = st.selectbox("State", sorted(market_df["State"].dropna().unique()))
            month = st.selectbox("Planning month", list(range(1, 13)), index=5)
        submitted = st.form_submit_button("Show recommendations")

    if not submitted:
        return

    candidates = predict_crop_recommendations(
        assets,
        [nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall],
    ).head(3)
    season = infer_season(month)
    results = []
    for _, rec in candidates.iterrows():
        market_crop = to_market_crop_name(str(rec["Crop"]))
        predicted_price, context = predict_price(
            assets,
            market_df,
            market_crop,
            state,
            season,
            month,
            int(market_df["Year"].max()),
        )
        results.append(
            {
                "Crop": market_crop,
                "Suitability (%)": round(float(rec["Confidence"]) * 100, 2),
                "Expected price": f"Rs. {predicted_price:,.0f}",
                "Latest price": f"Rs. {float(context['Modal_Price']):,.0f}",
                "Trend": str(context["Trend_Label"]),
            }
        )
    st.table(pd.DataFrame(results))


def render_fertilizer_recommendation(assets: dict) -> None:
    st.header("Fertilizer Recommendation")
    st.write("Choose soil and crop details to find the best fertilizer recommendation from the dataset.")

    if not assets.get("fertilizer_available", False):
        st.info("Fertilizer recommendation data is not available. Please add Data/fertilizer_recommendation_dataset.csv.")
        return

    fertilizer_df = assets.get("fertilizer_df")
    if fertilizer_df is None or fertilizer_df.empty:
        st.error("Could not load fertilizer recommendation data.")
        return

    crops = sorted(fertilizer_df["Crop"].dropna().unique())
    soils = sorted(fertilizer_df["Soil"].dropna().unique())

    with st.form("fertilizer_recommendation_form"):
        left, right = st.columns(2)
        with left:
            temperature = st.number_input("Temperature (°C)", value=25.0)
            moisture = st.number_input("Moisture", min_value=0.0, max_value=1.0, value=0.70, format="%.3f")
            rainfall = st.number_input("Rainfall (mm)", value=150.0)
            ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5)
        with right:
            nitrogen = st.number_input("Nitrogen", min_value=0.0, value=60.0)
            phosphorous = st.number_input("Phosphorous", min_value=0.0, value=60.0)
            potassium = st.number_input("Potassium", min_value=0.0, value=60.0)
            carbon = st.number_input("Carbon", min_value=0.0, value=0.5)
        crop = st.selectbox("Crop", crops)
        soil = st.selectbox("Soil type", soils)
        submitted = st.form_submit_button("Get fertilizer recommendations")

    if not submitted:
        return

    sample = {
        "Temperature": temperature,
        "Moisture": moisture,
        "Rainfall": rainfall,
        "PH": ph,
        "Nitrogen": nitrogen,
        "Phosphorous": phosphorous,
        "Potassium": potassium,
        "Carbon": carbon,
        "Crop": crop,
        "Soil": soil,
    }
    recommendations = recommend_fertilizer(fertilizer_df, sample, top_n=3)
    if recommendations.empty:
        st.warning("No fertilizer recommendations were found for this input.")
        return

    st.subheader("Top fertilizer matches")
    st.dataframe(recommendations.reset_index(drop=True), use_container_width=True)


def render_price_forecast(market_df: pd.DataFrame, assets: dict) -> None:
    st.header("Price Forecast")
    st.write("Choose a crop, state, and season to estimate the selling price.")

    with st.form("price_check_form"):
        state = st.selectbox("State", sorted(market_df["State"].dropna().unique()), key="forecast_state")
        season = st.selectbox("Season", ["Kharif", "Rabi", "Zaid"], key="forecast_season")
        crop = st.selectbox("Crop", sorted(market_df["Crop"].dropna().unique()), key="forecast_crop")
        checked = st.form_submit_button("Forecast price")

    if not checked:
        return

    pred_price, context = predict_price(
        assets,
        market_df,
        crop,
        state,
        season,
        MONTH_BY_SEASON[season],
        int(market_df["Year"].max()),
    )

    st.metric("Expected price", f"Rs. {pred_price:,.2f}")
    st.metric("Trend", str(context["Trend_Label"]))
    st.metric("Profit margin", f"{float(context['Profit_Margin']):.2f}%")

    history = market_df[(market_df["Crop"] == crop) & (market_df["State"] == state)].copy()
    if not history.empty:
        history = history.sort_values("Date")[['Date', 'Modal_Price']].set_index('Date')
        st.line_chart(history.rename(columns={'Modal_Price': 'Modal Price'}))
    else:
        st.info("No historical price data is available for this crop and state.")


def render_disease_scan(assets: dict) -> None:
    st.header("Disease Scan")
    st.write("Upload a leaf image for a quick diagnosis.")
    if not assets.get("disease_available", False):
        if assets.get("disease_error"):
            st.error("Disease model artifacts were found, but the model could not be loaded.")
            st.write(assets["disease_error"])
            with st.expander("How to fix this"):
                st.markdown(
                    "1. Install TensorFlow in the app environment with `pip install tensorflow`\n"
                    "2. Restart the app after installation\n"
                    "3. Ensure `models/disease_detection/model.keras` and `models/disease_detection/metadata.json` exist."
                )
        else:
            st.info("Disease model is not available. Please place the disease artifacts in models/disease_detection.")
            with st.expander("How to generate the disease model"):
                st.markdown(
                    "Run the training script from the project root and then restart the app."
                )
                st.code(disease_save_snippet(), language="bash")
        return

    uploaded_leaf = st.file_uploader(
        "Upload a crop leaf image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
    )
    if uploaded_leaf is None:
        return

    st.image(uploaded_leaf, caption="Uploaded image", use_container_width=True)
    try:
        disease_result = predict_disease(uploaded_leaf.getvalue(), assets["disease"], top_k=3)
        best = disease_result["top_predictions"][0]
        st.success(f"Likely diagnosis: {disease_result['label']} ({best['confidence'] * 100:.1f}%)")
        prediction_df = pd.DataFrame(disease_result["top_predictions"]).rename(
            columns={"label": "Class", "plant": "Plant", "disease": "Disease", "confidence": "Confidence"}
        )
        prediction_df["Confidence"] = prediction_df["Confidence"].map(lambda value: round(value * 100, 2))
        st.table(prediction_df)
    except Exception as exc:
        st.error(f"Image analysis failed: {exc}")


def render_insights() -> None:
    section_title(
        "Farm Insights",
        "Compare profitability, risk, market stability, and seasonal behavior from saved analysis files.",
    )
    tab1, tab2, tab3 = st.tabs(["Profitability", "Risk and Stability", "Saved Charts"])

    with tab1:
        profit_df = load_csv(str(OUTPUTS_DIR / "crop_profitability.csv"))
        state_profit = load_csv(str(OUTPUTS_DIR / "state_profitability.csv"))
        st.subheader("Crop profitability")
        st.dataframe(profit_df, use_container_width=True, hide_index=True)
        st.bar_chart(profit_df.set_index("Crop")["Avg_Margin%"])
        st.subheader("State profitability")
        st.dataframe(state_profit, use_container_width=True, hide_index=True)

    with tab2:
        risk_df = load_csv(str(OUTPUTS_DIR / "risk_reward_matrix.csv"))
        stability_df = load_csv(str(OUTPUTS_DIR / "market_stability.csv"))
        seasonal_df = load_csv(str(OUTPUTS_DIR / "seasonal_analysis.csv"))
        st.subheader("Risk reward matrix")
        st.dataframe(risk_df, use_container_width=True, hide_index=True)
        st.bar_chart(risk_df.set_index("Crop")["Risk_Score"])
        st.subheader("Market stability")
        st.dataframe(stability_df, use_container_width=True, hide_index=True)
        selected_crop = st.selectbox("Seasonality crop", sorted(seasonal_df["Crop"].unique()))
        seasonal_view = seasonal_df[seasonal_df["Crop"] == selected_crop][["Season", "Avg_Price"]].set_index("Season")
        st.bar_chart(seasonal_view)

    with tab3:
        chart_paths = [
            OUTPUTS_DIR / "crop_profitability.png",
            OUTPUTS_DIR / "profit_heatmap.png",
            OUTPUTS_DIR / "risk_reward_scatter.png",
            OUTPUTS_DIR / "seasonal_analysis.png",
            OUTPUTS_DIR / "state_profitability.png",
            OUTPUTS_DIR / "market_comparison_heatmap.png",
        ]
        available = [path for path in chart_paths if path.exists()]
        if not available:
            st.warning("No saved charts were found in models/outputs.")
        for path in available:
            caption = path.name.replace("_", " ").replace(".png", "").title()
            st.image(str(path), caption=caption, use_container_width=True)


def main() -> None:
    market_df = load_market_data()
    assets = load_assets()
    page = sidebar(market_df)
    render_header(market_df, assets)

    if page == "Overview":
        render_overview(market_df, assets)
    elif page == "Crop Advisor":
        render_crop_advisor(market_df, assets)
    elif page == "Fertilizer Recommendation":
        render_fertilizer_recommendation(assets)
    elif page == "Price Forecast":
        render_price_forecast(market_df, assets)
    elif page == "Disease Scan":
        render_disease_scan(assets)


if __name__ == "__main__":
    main()

