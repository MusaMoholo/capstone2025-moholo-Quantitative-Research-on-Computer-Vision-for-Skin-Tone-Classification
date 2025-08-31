
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["KERAS_BACKEND"] = "tensorflow"

import json
import io
from pathlib import Path
import numpy as np
import streamlit as st
from PIL import Image
import keras

CHAMPION_DIRS = [
    Path("exports/champion"),
]

IMG_SIZE = (224, 224)
CLASS_DEFAULT_MAP = {0: "I", 1: "II", 2: "III", 3: "IV", 4: "V", 5: "VI"}

def _find_n96_gamma_run():
    cands = []
    for cfg in Path("outputs/outputs_stage2").rglob("config.json"):
        try:
            d = json.loads(cfg.read_text())
        except Exception:
            continue
        if d.get("last_n") != 96:
            continue
        pre = d.get("preproc")
        run_dir = cfg.parent
        model_path = run_dir / "model.keras"
        preds = run_dir / "preds_test.csv"
        metrics = run_dir / "metrics.json"
        if not (model_path.exists() and metrics.exists() and preds.exists()):
            continue
        try:
            m = json.loads(metrics.read_text())
            mf1 = float(m.get("macro-F1") or m.get("macro_f1") or -1.0)
        except Exception:
            mf1 = -1.0
        cands.append((pre == "gamma", mf1, run_dir))
    if not cands:
        return None
    cands.sort(key=lambda t: (not t[0], -t[1]))
    return cands[0][2]

def _resolve_champion_dir():
    for p in CHAMPION_DIRS:
        if (p / "model.keras").exists():
            return p
    auto = _find_n96_gamma_run()
    return auto

CHAMP = _resolve_champion_dir()
if CHAMP is None:
    st.error("No champion model found. Place an export in exports/champion/ or keep outputs/outputs_stage2 available.")
    st.stop()

@st.cache_resource(show_spinner=False)
def load_model_and_assets(champ_dir: Path):
    model_path = champ_dir / "model.keras"
    model = keras.models.load_model(model_path, compile=False)

    logits_model = None
    try:
        last = model.layers[-1]
        logits_model = keras.Model(model.input, last.input)
    except Exception:
        logits_model = None

    T = 1.0
    t_path = champ_dir / "T.txt"
    if t_path.exists():
        try:
            T = float(t_path.read_text().strip())
        except Exception:
            pass

    cmap = CLASS_DEFAULT_MAP.copy()
    cmap_path = champ_dir / "class_map.json"
    if cmap_path.exists():
        try:
            raw = json.loads(cmap_path.read_text())
            if isinstance(raw, dict):
                cmap = {int(k): str(v) for k, v in raw.items()}
            elif isinstance(raw, list) and len(raw) == 6:
                cmap = {i: str(v) for i, v in enumerate(raw)}
        except Exception:
            pass

    gamma_value = 1.2
    preproc_path = champ_dir / "preproc.json"
    if preproc_path.exists():
        try:
            pconf = json.loads(preproc_path.read_text())
            if pconf.get("name") == "gamma" and "gamma" in pconf:
                gamma_value = float(pconf["gamma"])
        except Exception:
            pass

    med_table = None
    med_path = champ_dir / "med_table.json"
    if med_path.exists():
        try:
            med_table = json.loads(med_path.read_text())
            norm = {}
            for k, v in med_table.items():
                if isinstance(k, str) and k.upper() in {"I","II","III","IV","V","VI"}:
                    norm[k.upper()] = float(v)
                else:
                    kk = int(k)
                    norm[cmap.get(kk, str(kk))] = float(v)
            med_table = norm
        except Exception:
            med_table = None

    return model, logits_model, T, cmap, gamma_value, med_table

MODEL, LOGITS_MODEL, TEMP, CLASS_MAP, GAMMA, MED_TABLE = load_model_and_assets(CHAMP)
IDX2NAME = [CLASS_MAP[i] for i in range(6)]

def pil_to_array(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize(IMG_SIZE, Image.BILINEAR)
    arr = np.asarray(img).astype("float32") / 255.0
    return arr

def gamma_preprocess(arr01: np.ndarray, gamma: float) -> np.ndarray:
    if gamma <= 0:
        return arr01
    return np.clip(arr01 ** (1.0 / gamma), 0.0, 1.0)

def to_mobilenet_range(arr01: np.ndarray) -> np.ndarray:
    return arr01 * 2.0 - 1.0

def softmax(x: np.ndarray, axis=-1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def predict_with_temperature(x_bchw: np.ndarray) -> np.ndarray:
    """Returns calibrated probabilities shape (B,6).
    If we can get logits, use softmax(logits/T).
    Otherwise, approximate by log-prob temperature (fallback)."""
    if LOGITS_MODEL is not None:
        logits = LOGITS_MODEL.predict(x_bchw, verbose=0)
        return softmax(logits / float(TEMP), axis=-1)
    probs = MODEL.predict(x_bchw, verbose=0)
    logp = np.log(np.clip(probs, 1e-9, 1.0))
    return softmax(logp / float(TEMP), axis=-1)

def abstain_logic(probs: np.ndarray, conf_thresh: float = 0.5, top2_margin: float = 0.05):
    """Return a dict with decision, top-1, top-2, and flags."""
    p = probs[0]
    order = np.argsort(-p)
    k1, k2 = int(order[0]), int(order[1])
    p1, p2 = float(p[k1]), float(p[k2])
    abstain = (p1 < conf_thresh)
    tieish = ((p1 - p2) < top2_margin)
    return {"k1": k1, "k2": k2, "p1": p1, "p2": p2, "abstain": abstain, "tieish": tieish}

def time_to_erythema_minutes(uvi: float, klass_name: str) -> float | None:
    """t_MIN = MED_SED * 66.7 / UVI (see thesis background).
    Requires MED_TABLE with SED per class name."""
    if uvi <= 0 or MED_TABLE is None:
        return None
    if klass_name not in MED_TABLE:
        return None
    med_sed = float(MED_TABLE[klass_name])
    return (med_sed * 66.7) / float(uvi)

st.set_page_config(page_title="Fitzpatrick Classifier (gamma + N=96)", page_icon="🌤️", layout="centered")
st.title("Fitzpatrick Classifier (gamma + N=96)")
st.caption(f"Model: MobileNetV3-Large head, partial unfreeze last 96; Preproc: gamma; Temperature T={TEMP:.3g}; Running on CPU.")

with st.sidebar:
    st.header("Settings")
    conf_thresh = st.slider("Abstain below calibrated max-prob", 0.0, 1.0, 0.5, 0.01)
    top2_margin = st.slider("Flag when top-1 and top-2 are close", 0.0, 0.5, 0.05, 0.01)
    show_preproc = st.checkbox("Show preprocessed preview", value=False)
    st.write(f"Gamma (fixed): **{GAMMA:.2f}**")
    st.divider()
    uvi = st.number_input("UV Index (UVI)", min_value=0.0, value=5.0, step=0.5, format="%.1f")
    if MED_TABLE is None:
        st.info("Provide `med_table.json` in the model folder to enable time-to-erythema.")

upl = st.file_uploader("Upload a skin image (RGB)", type=["jpg","jpeg","png"])

if upl is None:
    st.info("Upload an image to get a prediction.")
    st.stop()

image = Image.open(io.BytesIO(upl.read()))
arr01 = pil_to_array(image)
arr01_gamma = gamma_preprocess(arr01, GAMMA)
x = to_mobilenet_range(arr01_gamma)[None, ...]

probs = predict_with_temperature(x)
decision = abstain_logic(probs, conf_thresh=conf_thresh, top2_margin=top2_margin)

c1, c2 = st.columns([1, 1])
with c1:
    st.subheader("Input")
    st.image(image, caption="Original (resized to 224×224)", use_column_width=True)
    if show_preproc:
        st.image((arr01_gamma * 255).astype(np.uint8), caption=f"After gamma (γ={GAMMA:.2f})", use_column_width=True)

with c2:
    st.subheader("Prediction")
    k1, k2 = decision["k1"], decision["k2"]
    p1, p2 = decision["p1"], decision["p2"]
    name1, name2 = IDX2NAME[k1], IDX2NAME[k2]
    badge = "🟡 Abstain" if decision["abstain"] else "🟢 Predict"
    st.markdown(
        f"**{badge}** — Top-1: **{name1}**  \n"
        f"Confidence (calibrated): **{p1:.2%}**  \n"
        f"Top-2: {name2} (**{p2:.2%}**)"
    )
    if decision["tieish"]:
        st.warning("Top-1 and top-2 are close. Consider additional capture or abstain.", icon="⚠️")

    st.write("Per-class calibrated probabilities:")
    for i in range(6):
        st.progress(float(probs[0, i]), text=f"Class {IDX2NAME[i]} — {probs[0, i]:.2%}")

st.divider()

if MED_TABLE is not None and not decision["abstain"]:
    t_min = time_to_erythema_minutes(uvi, IDX2NAME[decision["k1"]])
    if t_min is not None and np.isfinite(t_min):
        band = "short" if t_min < 20 else ("moderate" if t_min < 60 else "long")
        st.subheader("Estimated time-to-first-erythema")
        st.markdown(
            f"**{t_min:.0f} minutes** at UVI **{uvi:.1f}** (band: **{band}**), "
            f"given MED table and predicted type **{IDX2NAME[decision['k1']]}**."
        )
        st.caption("Estimates are conservative and context-dependent (cloud, altitude, shade, sunscreen, site).")
    else:
        st.info("MED table does not cover the predicted class; no time estimate.")
elif MED_TABLE is None:
    st.caption('Add `med_table.json` with SED per class (e.g., {"I":2.0,...,"VI":6.0}) to enable time-to-erythema.')
else:
    st.caption("Abstained: no time estimate shown.")

st.divider()
st.caption(f"Model folder: `{CHAMP}` • Classes: {', '.join(IDX2NAME)} • Temperature scaling leaves argmax unchanged.")
