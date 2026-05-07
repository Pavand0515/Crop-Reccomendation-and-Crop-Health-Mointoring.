from __future__ import annotations

import json
import logging
import traceback
import warnings
import importlib.util
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAXResults

from huggingface_hub import hf_hub_download

from disease_detection import load_disease_artifacts, predict_disease

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler()])

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = MODELS_DIR / "outputs"
FRONTEND_DIST = BASE_DIR / "frontend" / "dist"

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
SEASONS = ["Kharif", "Rabi", "Zaid"]
CHART_FILES = [
    "crop_profitability.png",
    "profit_heatmap.png",
    "risk_reward_scatter.png",
    "seasonal_analysis.png",
    "state_profitability.png",
    "market_comparison_heatmap.png",
]


class CropAdvisorRequest(BaseModel):
    nitrogen: float = Field(90.0, ge=0)
    phosphorus: float = Field(42.0, ge=0)
    potassium: float = Field(43.0, ge=0)
    temperature: float = 20.88
    humidity: float = Field(82.0, ge=0, le=100)
    ph: float = Field(6.5, ge=0, le=14)
    rainfall: float = Field(202.94, ge=0)
    state: str
    month: int = Field(6, ge=1, le=12)


class FertilizerRequest(BaseModel):
    temperature: float = 25.0
    moisture: float = Field(0.7, ge=0)
    rainfall: float = 150.0
    ph: float = Field(6.5, ge=0, le=14)
    nitrogen: float = Field(60.0, ge=0)
    phosphorous: float = Field(60.0, ge=0)
    potassium: float = Field(60.0, ge=0)
    carbon: float = Field(0.5, ge=0)
    crop: str
    soil: str


class PriceForecastRequest(BaseModel):
    crop: str
    state: str
    season: str
    horizon: int = 4


app = FastAPI(title="AgriIntel API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")


def infer_season(month: int) -> str:
    if 6 <= month <= 10:
        return "Kharif"
    if month >= 11 or month <= 3:
        return "Rabi"
    return "Zaid"


@lru_cache(maxsize=1)
def load_market_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "market_prices_real.csv")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Month"] = df["Date"].dt.month
    df["Quarter"] = df["Date"].dt.quarter
    df["Year"] = df["Date"].dt.year
    df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
    df["Season"] = df["Month"].apply(infer_season)
    return df.sort_values("Date")


@lru_cache(maxsize=1)
def load_fertilizer_data() -> pd.DataFrame:
    path = DATA_DIR / "fertilizer_recommendation_dataset.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["Crop"] = df["Crop"].astype(str).str.strip().str.title()
    df["Soil"] = df["Soil"].astype(str).str.strip().str.title()
    return df


HF_REPO_ID = "Relentless707/crop-models"


def hf_load(repo_path: str) -> Any | None:
    """Download a file from HuggingFace Hub and load it with joblib. Returns None on failure."""
    try:
        local_path = hf_hub_download(repo_id=HF_REPO_ID, filename=repo_path)
        return joblib.load(local_path)
    except Exception as exc:
        logging.warning(f"Could not load {repo_path} from HuggingFace Hub: {exc}")
        return None


def hf_read_text(repo_path: str) -> str | None:
    """Download a text file from HuggingFace Hub and return its contents. Returns None on failure."""
    try:
        local_path = hf_hub_download(repo_id=HF_REPO_ID, filename=repo_path)
        return Path(local_path).read_text(encoding="utf-8")
    except Exception as exc:
        logging.warning(f"Could not read {repo_path} from HuggingFace Hub: {exc}")
        return None


@lru_cache(maxsize=1)
def load_assets() -> dict[str, Any]:
    assets = {
        "crop_model": hf_load("Models/crop_recommendation/rf.pkl"),
        "crop_scaler": hf_load("Models/crop_recommendation/scaler.pkl"),
        "crop_encoder": hf_load("Models/crop_recommendation/label_encoder.pkl"),
        "price_model": hf_load("Models/random_forest_model.pkl"),
        "price_scaler": hf_load("Models/scaler.pkl"),
        "encoders": hf_load("Models/encoders.pkl"),
        "metadata": {},
        "fertilizer_df": load_fertilizer_data(),
    }

    metadata_text = hf_read_text("Models/model_metadata.json")
    if metadata_text:
        assets["metadata"] = json.loads(metadata_text)
    return assets


SUPPORTED_CROPS_TS = {"Banana", "Maize", "Mango", "Papaya", "Water_Melon"}


@lru_cache(maxsize=32)
def load_ts_model(crop: str) -> SARIMAXResults | None:
    """Load best TS model (SARIMA preferred) for crop from HuggingFace Hub. Returns None if not available."""
    # Prefer SARIMA
    model = hf_load(f"Models/forecasting_models/{crop}_sarima_fit.pkl")
    if model is not None:
        return model
    # Fallback to ARIMA
    return hf_load(f"Models/forecasting_models/{crop}_arima_fit.pkl")


def disease_artifacts_exist() -> bool:
    """Check if disease model artifacts are available on HuggingFace Hub."""
    metadata_text = hf_read_text("Models/disease_detection/metadata.json")
    if not metadata_text:
        return False
    for model_file in ["model.pkl", "model.keras", "model.weights.h5"]:
        try:
            hf_hub_download(repo_id=HF_REPO_ID, filename=f"Models/disease_detection/{model_file}")
            return True
        except Exception:
            continue
    return False


def disease_status() -> tuple[bool, str | None]:
    if not disease_artifacts_exist():
        return False, "Disease model artifacts are unavailable."
    if importlib.util.find_spec("tensorflow") is None:
        return False, "TensorFlow is not installed in this Python environment."
    return True, None


@lru_cache(maxsize=1)
def load_disease_assets() -> dict[str, Any]:
    """Download disease detection artifacts from HuggingFace Hub into a temp dir and load them."""
    import tempfile, shutil
    tmp_dir = Path(tempfile.mkdtemp()) / "disease_detection"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for fname in ["model.pkl", "model.keras", "model.weights.h5", "metadata.json"]:
        try:
            src = hf_hub_download(repo_id=HF_REPO_ID, filename=f"Models/disease_detection/{fname}")
            shutil.copy(src, tmp_dir / fname)
        except Exception:
            pass
    return load_disease_artifacts(tmp_dir)


def require_assets(assets: dict[str, Any], keys: list[str], feature: str) -> None:
    missing = [
        key
        for key in keys
        if assets.get(key) is None or (key == "metadata" and not assets["metadata"])
    ]
    if missing:
        raise HTTPException(
            status_code=503,
            detail=f"{feature} is unavailable. Missing model artifact(s): {', '.join(missing)}.",
        )


def encode_value(encoders: dict, key: str, value: str) -> int:
    encoder = encoders[key]
    if value not in set(encoder.classes_):
        value = encoder.classes_[0]
    return int(encoder.transform([value])[0])


def records(df: pd.DataFrame) -> list[dict[str, Any]]:
    clean = df.replace({np.nan: None})
    return clean.to_dict(orient="records")


def build_feature_row(
    market_df: pd.DataFrame,
    encoders: dict,
    crop: str,
    state: str,
    season: str,
    month: int,
    year: int,
) -> tuple[pd.DataFrame, pd.Series]:
    # Multi-level fallback to ensure we find some context for the crop/state
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

    if scoped.empty:
        raise HTTPException(status_code=404, detail=f"No data found for crop: {crop}")

    scoped = scoped.sort_values("Date")
    row = scoped.iloc[-1]
    
    # Calculate rolling averages safely
    prices_tail = scoped["Modal_Price"].astype(float)
    rolling7 = float(prices_tail.tail(7).mean()) if len(prices_tail) >= 1 else 0.0
    rolling14 = float(prices_tail.tail(14).mean()) if len(prices_tail) >= 1 else 0.0
    lag1 = float(row["Modal_Price"])
    
    quarter = int(((month - 1) // 3) + 1)
    
    # Estimate yield based on profit margin if not available
    margin = float(row.get("Profit_Margin", 0))
    cost = max(float(row.get("Cultivation_Cost", 1)), 1.0)
    estimated_yield = max((lag1 * (1 + margin / 100.0) * 10) / cost, 1.0)

    feature_values = {
        "Crop_enc": encode_value(encoders, "Crop", crop),
        "State_enc": encode_value(encoders, "State", state),
        "Variety_enc": encode_value(encoders, "Variety", str(row["Variety"])),
        "Grade_enc": encode_value(encoders, "Grade", str(row["Grade"])),
        "District_enc": encode_value(encoders, "District", str(row["District"])),
        "Commodity_Code": int(row["Commodity_Code"]),
        "Month": int(month),
        "Quarter": quarter,
        "Season_enc": encode_value(encoders, "Season", season),
        "Year": int(year),
        "Week": int(pd.Timestamp(year=year, month=month, day=1).isocalendar().week),
        "Price_Trend_30d": float(row["Price_Trend_30d"]),
        "Trend_Label_enc": encode_value(encoders, "Trend_Label", str(row["Trend_Label"])),
        "Cultivation_Cost": float(row["Cultivation_Cost"]),
        "Avg_Yield_Qtl": estimated_yield,
        "Rolling_7d_Avg": rolling7,
        "Rolling_14d_Avg": rolling14,
        "Lag_1d_Price": lag1,
    }
    return pd.DataFrame([feature_values]), row


def predict_crop_recommendations(assets: dict[str, Any], sample: list[float]) -> pd.DataFrame:
    require_assets(assets, ["crop_model", "crop_scaler", "crop_encoder"], "Crop advisor")
    scaled = assets["crop_scaler"].transform([sample])
    probs = assets["crop_model"].predict_proba(scaled)[0]
    labels = assets["crop_encoder"].inverse_transform(np.arange(len(probs)))
    result = pd.DataFrame({"Crop": labels, "Confidence": probs})
    return result.sort_values("Confidence", ascending=False).head(6).reset_index(drop=True)


def recommend_fertilizer(df: pd.DataFrame, sample: dict[str, Any], top_n: int = 3) -> pd.DataFrame:
    numeric_cols = ["Temperature", "Moisture", "Rainfall", "PH", "Nitrogen", "Phosphorous", "Potassium", "Carbon"]
    sample_crop = str(sample.get("Crop", "")).strip().title()
    sample_soil = str(sample.get("Soil", "")).strip().title()

    filtered = df[(df["Crop"] == sample_crop) & (df["Soil"] == sample_soil)]
    if filtered.empty:
        filtered = df[df["Crop"] == sample_crop]
    if filtered.empty:
        filtered = df[df["Soil"] == sample_soil]
    if filtered.empty:
        filtered = df

    metrics = filtered[numeric_cols].astype(float)
    min_vals = metrics.min()
    range_vals = (metrics.max() - min_vals).replace(0, 1)
    normalized_rows = (metrics - min_vals) / range_vals
    sample_values = pd.Series([float(sample.get(col, 0.0)) for col in numeric_cols], index=numeric_cols)
    normalized_sample = (sample_values - min_vals) / range_vals
    filtered = filtered.copy()
    filtered["Distance"] = ((normalized_rows - normalized_sample) ** 2).sum(axis=1).pow(0.5)
    return (
        filtered.sort_values("Distance")
        .drop_duplicates(subset=["Fertilizer"])
        .head(top_n)[["Crop", "Soil", "Fertilizer", "Remark", "Distance"]]
    )


def forecast_price_ts(market_df: pd.DataFrame, crop: str, state: str, horizon: int = 4) -> tuple[list[dict], pd.Series | None]:
    scoped = market_df[(market_df["Crop"] == crop) & (market_df["State"] == state)].copy()
    if scoped.empty:
        scoped = market_df[market_df["Crop"] == crop].copy()
    if scoped.empty or len(scoped) < 10:
        return [], None
    
    # Aggregate to weekly prices (mean Modal_Price)
    scoped["Date"] = pd.to_datetime(scoped["Date"])
    scoped = scoped.set_index("Date").resample("W").agg({"Modal_Price": "mean"}).dropna()
    if len(scoped) < 10:
        return [], None
    
    prices = scoped["Modal_Price"]
    
    model = load_ts_model(crop)
    if model is None:
        return [], None
    
    try:
        forecast = model.forecast(steps=horizon)
        conf_int = model.get_forecast(steps=horizon).conf_int()
        
        forecasts = []
        last_date = prices.index[-1]
        for i in range(horizon):
            date = last_date + pd.Timedelta(weeks=i+1)
            forecasts.append({
                "date": date.date().isoformat(),
                "price": round(float(forecast.iloc[i]), 2),
                "lower_ci": round(float(conf_int.iloc[i, 0]), 2),
                "upper_ci": round(float(conf_int.iloc[i, 1]), 2)
            })
        logging.info(f"TS forecast succeeded for {crop}/{state}, horizon={horizon}")
        return forecasts, scoped.iloc[-1]
    except Exception as e:
        logging.error(f"TS forecast failed for {crop}/{state}: {str(e)}\n{traceback.format_exc()}")
        return [], None


def predict_price(
    assets: dict[str, Any],
    market_df: pd.DataFrame,
    crop: str,
    state: str,
    season: str,
    month: int,
    year: int,
    use_ts: bool = True,
    horizon: int = 1,
) -> tuple[list[dict] | float, pd.Series]:
    logging.info(f"Price predict called for {crop}/{state}/{season}, horizon={horizon}, use_ts={use_ts}")
    # Try TS forecast first
    if use_ts and crop in SUPPORTED_CROPS_TS and horizon > 1:
        logging.info(f"Attempting TS forecast for {crop}")
        ts_forecast, context = forecast_price_ts(market_df, crop, state, horizon)
        if ts_forecast:
            return ts_forecast, context
        logging.info(f"TS forecast returned empty for {crop}, falling back to RF")
    
    logging.info(f"Using RF fallback for {crop}")
    require_assets(assets, ["price_model", "price_scaler", "encoders", "metadata"], "Price forecast")
    features, context_row = build_feature_row(market_df, assets["encoders"], crop, state, season, month, year)
    features = features[assets["metadata"]["features"]]
    scaled = assets["price_scaler"].transform(features)
    pred = float(assets["price_model"].predict(scaled)[0])
    return pred, context_row


def to_market_crop_name(crop_label: str) -> str:
    return CROP_NAME_MAP.get(crop_label.strip().lower(), crop_label.title())


@app.get("/api/health")
def health() -> dict[str, Any]:
    assets = load_assets()
    ts_available = any(load_ts_model(crop) is not None for crop in SUPPORTED_CROPS_TS)
    disease_available, disease_error = disease_status()
    return {
        "status": "ok",
        "cropAdvisorAvailable": all(assets.get(key) is not None for key in ["crop_model", "crop_scaler", "crop_encoder"]),
        "priceForecastAvailable": all(assets.get(key) is not None for key in ["price_model", "price_scaler", "encoders"]) and bool(assets["metadata"]),
        "tsForecastAvailable": ts_available,
        "fertilizerAvailable": not assets["fertilizer_df"].empty,
        "diseaseAvailable": disease_available,
        "diseaseError": disease_error,
    }


@app.get("/api/options")
def options() -> dict[str, Any]:
    market_df = load_market_data()
    fertilizer_df = load_fertilizer_data()
    return {
        "states": sorted(market_df["State"].dropna().unique().tolist()),
        "crops": sorted(market_df["Crop"].dropna().unique().tolist()),
        "seasons": SEASONS,
        "months": list(range(1, 13)),
        "fertilizerCrops": sorted(fertilizer_df["Crop"].dropna().unique().tolist()) if not fertilizer_df.empty else [],
        "soils": sorted(fertilizer_df["Soil"].dropna().unique().tolist()) if not fertilizer_df.empty else [],
    }


@app.get("/api/overview")
def overview() -> dict[str, Any]:
    market_df = load_market_data()
    state_counts = (
        market_df.groupby("State").size().sort_values(ascending=False).head(10).reset_index(name="Records")
    )
    return {
        "metrics": {
            "marketRecords": int(len(market_df)),
            "crops": int(market_df["Crop"].nunique()),
            "states": int(market_df["State"].nunique()),
            "latestYear": int(market_df["Year"].max()),
        },
        "stateCounts": records(state_counts),
    }


@app.post("/api/crop-advisor")
def crop_advisor(payload: CropAdvisorRequest) -> dict[str, Any]:
    market_df = load_market_data()
    assets = load_assets()
    candidates = predict_crop_recommendations(
        assets,
        [
            payload.nitrogen,
            payload.phosphorus,
            payload.potassium,
            payload.temperature,
            payload.humidity,
            payload.ph,
            payload.rainfall,
        ],
    ).head(3)
    season = infer_season(payload.month)
    rows = []
    for _, rec in candidates.iterrows():
        market_crop = to_market_crop_name(str(rec["Crop"]))
        row = {
            "crop": market_crop,
            "suitability": round(float(rec["Confidence"]) * 100, 2),
            "trend": "Unavailable",
        }
        try:
            predicted_price, context = predict_price(
                assets,
                market_df,
                market_crop,
                payload.state,
                season,
                payload.month,
                int(market_df["Year"].max()),
            )
            if isinstance(predicted_price, list):
                row["forecasts"] = predicted_price
            else:
                row["expectedPrice"] = round(predicted_price, 2)
            row["latestPrice"] = round(float(context["Modal_Price"]), 2)
            row["trend"] = str(context["Trend_Label"])
        except HTTPException:
            row.update({"expectedPrice": None, "latestPrice": None})
        rows.append(row)
    return {"season": season, "recommendations": rows}


@app.post("/api/fertilizer")
def fertilizer(payload: FertilizerRequest) -> dict[str, Any]:
    fertilizer_df = load_assets()["fertilizer_df"]
    if fertilizer_df.empty:
        raise HTTPException(status_code=503, detail="Fertilizer recommendation data is unavailable.")
    sample = {
        "Temperature": payload.temperature,
        "Moisture": payload.moisture,
        "Rainfall": payload.rainfall,
        "PH": payload.ph,
        "Nitrogen": payload.nitrogen,
        "Phosphorous": payload.phosphorous,
        "Potassium": payload.potassium,
        "Carbon": payload.carbon,
        "Crop": payload.crop,
        "Soil": payload.soil,
    }
    return {"recommendations": records(recommend_fertilizer(fertilizer_df, sample))}


@app.post("/api/price-forecast")
def price_forecast(payload: PriceForecastRequest) -> dict[str, Any]:
    logging.info(f"Price forecast request received: {payload.dict()}")
    market_df = load_market_data()
    logging.info(f"Market data loaded, shape: {market_df.shape}")
    if payload.season not in SEASONS:
        raise HTTPException(status_code=422, detail=f"Season must be one of: {', '.join(SEASONS)}")
    assets = load_assets()
    pred, context = predict_price(
        assets,
        market_df,
        payload.crop,
        payload.state,
        payload.season,
        MONTH_BY_SEASON[payload.season],
        int(market_df["Year"].max()),
        horizon=getattr(payload, 'horizon', 4),
    )
    history = market_df[(market_df["Crop"] == payload.crop) & (market_df["State"] == payload.state)].copy()
    history = history.sort_values("Date").tail(40)
    history_rows = [
        {"date": row["Date"].date().isoformat(), "price": round(float(row["Modal_Price"]), 2)}
        for _, row in history.iterrows()
    ]
    result = {
        "trend": str(context["Trend_Label"]),
        "profitMargin": round(float(context["Profit_Margin"]), 2),
        "history": history_rows,
    }
    if isinstance(pred, list):
        result["forecasts"] = pred
        result["expectedPrice"] = pred[0]["price"] if pred else None
    else:
        result["expectedPrice"] = round(pred, 2)
    return result


@app.post("/api/disease-scan")
async def disease_scan(file: UploadFile = File(...)) -> dict[str, Any]:
    disease_available, disease_error = disease_status()
    if not disease_available:
        raise HTTPException(status_code=503, detail=disease_error)
    image_bytes = await file.read()
    try:
        disease_assets = load_disease_assets()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    result = predict_disease(image_bytes, disease_assets, top_k=3)
    result["top_predictions"] = [
        {**item, "confidencePercent": round(float(item["confidence"]) * 100, 2)}
        for item in result["top_predictions"]
    ]
    return result


@app.get("/api/insights")
def insights() -> dict[str, Any]:
    tables = {}
    for name in ["crop_profitability", "state_profitability", "risk_reward_matrix", "market_stability", "seasonal_analysis"]:
        path = OUTPUTS_DIR / f"{name}.csv"
        tables[name] = records(pd.read_csv(path).head(25)) if path.exists() else []
    charts = [name for name in CHART_FILES if (OUTPUTS_DIR / name).exists()]
    return {"tables": tables, "charts": charts}


@app.get("/api/charts/{file_name}")
def chart(file_name: str) -> FileResponse:
    if file_name not in CHART_FILES:
        raise HTTPException(status_code=404, detail="Chart not found.")
    path = OUTPUTS_DIR / file_name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Chart not found.")
    return FileResponse(path)


@app.get("/{full_path:path}", response_model=None)
def serve_react(full_path: str):
    index_path = FRONTEND_DIST / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "Run the React app from frontend/ with npm run dev, or build it with npm run build."}