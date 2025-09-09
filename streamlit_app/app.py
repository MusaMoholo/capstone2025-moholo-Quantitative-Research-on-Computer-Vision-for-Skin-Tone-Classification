#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps
import streamlit as st

# --- NEW: for UV Index fetch & caching ---
import requests
from datetime import datetime
import pandas as pd
import math

# ---------- Defaults & artifact paths ----------
DEFAULT_CLASS_MAP = {str(i): lab for i, lab in enumerate(["I", "II", "III", "IV", "V", "VI"])}

# Literature-anchored MED tables in SED (1 SED = 100 J/m² erythemally weighted)
# Midpoint: typical central values across phototypes
MED_SED_MIDPOINT = {"I": 2.5, "II": 3.0, "III": 5.0, "IV": 7.0, "V": 9.0, "VI": 12.0}
# Conservative: lower-edge estimates (burns sooner → safer)
MED_SED_CONSERVATIVE = {"I": 2.0, "II": 2.0, "III": 3.0, "IV": 4.0, "V": 5.0, "VI": 8.0}

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "exports" / "champion"

# ---------- Math helpers ----------
def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / (np.sum(ex) + 1e-12)

def apply_temperature(probs_or_logits: np.ndarray, T: Optional[float]) -> np.ndarray:
    x = np.asarray(probs_or_logits, dtype=np.float64)
    if T is None or T <= 0:
        s = x.sum()
        if 0.9 <= s <= 1.1:
            return np.clip(x / (s + 1e-12), 0, 1)
        return softmax(x)
    if 0.98 <= x.sum() <= 1.02:  # looks like probabilities
        x = np.log(np.clip(x, 1e-8, 1.0))
    x = x / T
    return softmax(x)

def apply_gamma(rgb_float01: np.ndarray, gamma: float) -> np.ndarray:
    if gamma is None or abs(gamma - 1.0) < 1e-6:
        return rgb_float01
    rgb = np.clip(rgb_float01, 0.0, 1.0)
    return np.power(rgb, gamma)

def load_json(path: Path, default):
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return default

def load_temperature(path: Path) -> Optional[float]:
    try:
        if path.exists():
            return float(path.read_text().strip())
    except Exception:
        pass
    return None

# ---------- Inference backends ----------
class Backend:
    def name(self) -> str: ...
    def input_size(self) -> Tuple[int, int]: ...
    def predict(self, x: np.ndarray) -> np.ndarray: ...

class TFLiteBackend(Backend):
    def __init__(self, model_path: Path):
        try:
            from tflite_runtime.interpreter import Interpreter  # type: ignore
        except Exception as e:
            raise RuntimeError("tflite_runtime not available.") from e
        self.interpreter = Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        self.inp = self.interpreter.get_input_details()[0]
        self.out = self.interpreter.get_output_details()[0]
        h, w = int(self.inp["shape"][1]), int(self.inp["shape"][2])
        self._size = (w, h)

    def name(self) -> str: return "TFLite"
    def input_size(self) -> Tuple[int, int]: return self._size
    def predict(self, x: np.ndarray) -> np.ndarray:
        arr = x.astype(self.inp.get("dtype", np.float32), copy=False)
        self.interpreter.set_tensor(self.inp["index"], arr)
        self.interpreter.invoke()
        y = self.interpreter.get_tensor(self.out["index"])
        return np.squeeze(y)

class MockBackend(Backend):
    def __init__(self, size=(224, 224)): self._size = size
    def name(self) -> str: return "Mock"
    def input_size(self) -> Tuple[int, int]: return self._size
    def predict(self, x: np.ndarray) -> np.ndarray:
        rgb = np.squeeze(x, axis=0)
        y = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        mu = float(np.mean(y))
        bin_idx = int(np.clip(np.floor((1.0 - mu) * 6), 0, 5))
        scores = np.zeros(6, dtype=np.float64)
        for i in range(6):
            dist = abs(i - bin_idx)
            scores[i] = max(0.0, 1.0 - 0.5 * dist)
        return scores

def build_backend(art_dir: Path) -> Backend:
    tflite_path = art_dir / "model.tflite"
    if tflite_path.exists():
        try:
            return TFLiteBackend(tflite_path)
        except Exception as e:
            st.warning(f"TFLite unavailable: {e}. Falling back to Mock backend.")
    return MockBackend()

# ---------- Image I/O & Preprocessing ----------
def load_image(file):
    img = Image.open(file).convert("RGB")
    return ImageOps.exif_transpose(img)

def preprocess(img: Image.Image, target_wh: Tuple[int, int], preproc_cfg: Optional[Dict] = None) -> np.ndarray:
    w, h = target_wh
    resized = img.resize((w, h), Image.BILINEAR)
    x = np.asarray(resized, dtype=np.float32) / 255.0
    if preproc_cfg and preproc_cfg.get("name") == "gamma":
        gamma = float(preproc_cfg.get("gamma", 1.0))
        x = apply_gamma(x, gamma)
    return np.expand_dims(x, axis=0)

# ---------- UVI & TTE helpers ----------
@st.cache_data(show_spinner=False, ttl=900)
def fetch_open_meteo_uvi(lat: float, lon: float):
    """Return (current_uvi, hourly_df_today). Uses nearest hour from hourly UV index."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "uv_index",
        "forecast_days": 1,
        "timezone": "auto",
    }
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    times = [datetime.fromisoformat(t) for t in data["hourly"]["time"]]
    uvis = [float(u) if u is not None else 0.0 for u in data["hourly"]["uv_index"]]
    # nearest hour to local now
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    idx = int(np.argmin([abs((t - now).total_seconds()) for t in times]))
    current = float(uvis[idx])
    df = pd.DataFrame({"time": times, "uv_index": uvis})
    return current, df

def tte_minutes_from_uvi(SED: float, UVI: float) -> Optional[float]:
    """TTE (min) ≈ 66.67 × SED ÷ UVI. Returns None if invalid."""
    try:
        if SED is None or UVI is None or UVI <= 0:
            return None
        return 66.6667 * float(SED) / float(UVI)
    except Exception:
        return None

ROMAN = ["I", "II", "III", "IV", "V", "VI"]

def normalize_fitz_label(label: str, idx: int) -> str:
    """Ensure we have a Fitzpatrick Roman numeral for MED lookup."""
    lab = (label or "").strip().upper()
    return lab if lab in ROMAN else ROMAN[int(np.clip(idx, 0, 5))]

def get_med_for_label(
    fitz_label: str,
    pred_idx: int,
    med_policy: str,
    artifact_med_table: Optional[dict],
) -> float:
    """
    Returns MED in SED according to policy:
      - Auto: use artifact table if it contains a value; else fallback to midpoint
      - Literature midpoint
      - Conservative
    """
    ftz = normalize_fitz_label(fitz_label, pred_idx)
    # Try artifact first if policy is Auto
    if med_policy.startswith("Auto") and isinstance(artifact_med_table, dict):
        # Keys may be "I"/"1"/1 etc.
        for k in (ftz, str(ftz), str(pred_idx), pred_idx):
            if k in artifact_med_table:
                try:
                    return float(artifact_med_table[k])
                except Exception:
                    pass
    # Fallback by selected policy
    if "Conservative" in med_policy:
        return float(MED_SED_CONSERVATIVE[ftz])
    # Default to midpoint
    return float(MED_SED_MIDPOINT[ftz])

# ---------- UI ----------
st.set_page_config(page_title="Skin Tone Pipeline Demo", page_icon="🎛️", layout="centered")
st.title("Skin Tone Pipeline Demo")
st.caption("Predict phototype → get local UV Index → estimate time-to-erythema (TTE).")

# Sidebar: Artifacts + policies
with st.sidebar:
    st.header("Artifacts")
    art_str = st.text_input("Artifacts folder", str(ARTIFACT_DIR))
    art_dir = Path(art_str)
    st.write("Optional files:")
    st.code("model.tflite\nT.txt\nclass_map.json\npreproc.json\nmed_table.json", language="text")

    st.header("MED policy")
    med_policy = st.radio(
        "Phototype MED table (SED)",
        ["Auto (prefer med_table.json)", "Literature midpoint", "Conservative"],
        index=1,
        help="Auto uses med_table.json if present; otherwise falls back to a literature-anchored table.",
    )

# Load artifacts
class_map = load_json(art_dir / "class_map.json", DEFAULT_CLASS_MAP)
classes = [class_map.get(str(i), f"Class {i}") for i in range(6)]
temperature = load_temperature(art_dir / "T.txt")
preproc_cfg = load_json(art_dir / "preproc.json", None)
artifact_med_table = load_json(art_dir / "med_table.json", None)

# Backend
backend = build_backend(art_dir)
st.info(f"Backend: **{backend.name()}**  |  Input: **{backend.input_size()[0]}×{backend.input_size()[1]}**")

# Sidebar: UVI source
with st.sidebar:
    st.header("UV Index")
    src = st.radio("Source", ["Manual", "Open-Meteo (current)"], index=0)
    uvi_value: Optional[float] = None
    uvi_curve = None

    if src == "Manual":
        uvi_value = st.number_input("UVI (0–14)", min_value=0.0, max_value=14.0, value=6.0, step=0.1)
    else:
        col1, col2 = st.columns(2)
        lat = col1.number_input("Lat", min_value=-90.0, max_value=90.0, value=51.5, step=0.1)
        lon = col2.number_input("Lon", min_value=-180.0, max_value=180.0, value=-0.1, step=0.1)
        try:
            uvi_value, uvi_curve = fetch_open_meteo_uvi(lat, lon)
            st.metric("Current UVI", f"{uvi_value:.1f}")
        except Exception as e:
            st.warning(f"Could not fetch UVI: {e}. Enter manually instead.")
            uvi_value = st.number_input("UVI (0–14)", min_value=0.0, max_value=14.0, value=6.0, step=0.1)

uploaded = st.file_uploader("Upload a single face/skin image (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    img = load_image(uploaded)
    st.image(img, caption="Input image", use_container_width=True)

    x = preprocess(img, backend.input_size(), preproc_cfg)
    raw = backend.predict(x)
    probs = apply_temperature(raw, temperature)

    pred_idx = int(np.argmax(probs))
    pred_label = classes[pred_idx]
    conf = float(probs[pred_idx])
    fitz_for_med = normalize_fitz_label(pred_label, pred_idx)

    # MED lookup according to policy (artifact preferred if Auto)
    med_sed = get_med_for_label(fitz_for_med, pred_idx, med_policy, artifact_med_table)

    st.subheader("Result")
    st.markdown(f"**Predicted class:** {pred_label}  |  **Confidence:** {conf:.2%}")
    st.markdown(f"**Phototype (for MED):** {fitz_for_med}")
    st.markdown(f"**Estimated MED:** {med_sed:.2f} SED")

    # TTE from UVI
    tte_min = tte_minutes_from_uvi(med_sed, uvi_value) if (uvi_value is not None) else None
    if tte_min and math.isfinite(tte_min):
        hh = int(tte_min // 60)
        mm = int(round(tte_min % 60))
        st.markdown(f"**Local UVI:** {uvi_value:.1f}  →  **Time-to-Erythema (TTE):** ~{tte_min:.0f} min (~{hh}h {mm}m)")
    else:
        st.info("Provide a valid UVI (manual or fetched) to compute TTE.")

    # Probabilities table & chart
    df = pd.DataFrame({"class": classes, "probability": probs})
    st.dataframe(df.style.format({"probability": "{:.2%}"}), use_container_width=True)
    st.bar_chart(df.set_index("class"))

    # Optional: show today's UVI curve if fetched
    if uvi_curve is not None:
        st.caption("Today's hourly UVI (Open-Meteo)")
        st.line_chart(uvi_curve.set_index("time"))

    # Downloadable JSON
    result = {
        "backend": backend.name(),
        "input_size": backend.input_size(),
        "classes": classes,
        "probs": [float(p) for p in probs],
        "pred_idx": pred_idx,
        "pred_label": pred_label,
        "confidence": conf,
        "temperature": temperature,
        "preproc": preproc_cfg,
        "fitz_for_med": fitz_for_med,
        "estimated_MED_SED": float(med_sed),
        "UVI": float(uvi_value) if uvi_value is not None else None,
        "TTE_min": float(tte_min) if (tte_min is not None and math.isfinite(tte_min)) else None,
        "med_policy": med_policy,
        "artifact_dir": str(art_dir),
    }
    st.download_button(
        "Download result JSON",
        data=json.dumps(result, indent=2),
        file_name="prediction.json",
        mime="application/json",
    )

with st.expander("Notes & disclaimers"):
    st.markdown(
        """
- **TTE is an estimate** based on UVI→irradiance and MED in SED; real-world erythema depends on altitude, reflection, cloud type,
  sunscreen, body site, acclimatization, and inter-individual variability.
- Open-Meteo fetch uses the **nearest hour** of today's hourly UVI for your coordinates; you can also enter UVI manually.
- This demo is **not a medical device**.
        """
    )
