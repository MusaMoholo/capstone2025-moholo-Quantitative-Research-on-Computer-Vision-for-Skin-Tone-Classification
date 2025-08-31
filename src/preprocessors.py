# src/preprocessors.py
import cv2
import numpy as np


def _as_numpy(img):
    """Accept TF tensors, NumPy arrays, or lists; return NumPy array."""
    try:
        import tensorflow as tf
        if isinstance(img, tf.Tensor):
            img = img.numpy()
    except Exception:
        pass
    if not isinstance(img, np.ndarray):
        img = np.asarray(img)
    return img

def _to_uint8(img: np.ndarray) -> np.ndarray:
    """Accept float [0,1] or uint8 [0,255]; return contiguous uint8 [0,255]."""
    img = _as_numpy(img)
    if img.dtype == np.uint8:
        out = img
    else:
        out = np.clip(img, 0.0, 1.0) * 255.0
        out = out.astype(np.uint8, copy=False)
    return np.ascontiguousarray(out)

def _to_float01(img: np.ndarray) -> np.ndarray:
    """Accept uint8 or float; return float32 in [0,1]."""
    img = _as_numpy(img)
    if img.dtype == np.uint8:
        out = img.astype(np.float32) / 255.0
    else:
        out = img.astype(np.float32, copy=False)
        if out.max() > 1.5:
            out = out / 255.0
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)

def _rgb_to_lab_u8(rgb_u8: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB)

def _lab_to_rgb_u8(lab_u8: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(lab_u8, cv2.COLOR_LAB2RGB)

def _rgb_to_hsv_u8(rgb_u8: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)

def _hsv_to_rgb_u8(hsv_u8: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(hsv_u8, cv2.COLOR_HSV2RGB)

def identity(img: np.ndarray) -> np.ndarray:
    """No-op; ensure float32 [0,1] output."""
    return _to_float01(img)

def bnorm_lab_l(img: np.ndarray) -> np.ndarray:
    """Normalize LAB L-channel to [0,255] via min-max (contrast/brightness leveling)."""
    rgb = _to_uint8(img)
    lab = _rgb_to_lab_u8(rgb)
    L, A, B = cv2.split(lab)
    Ln = cv2.normalize(L, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    out = _lab_to_rgb_u8(cv2.merge([Ln, A, B]))
    return _to_float01(out)

def clahe_lab(img: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    """Apply CLAHE on LAB L-channel (local contrast enhancement)."""
    rgb = _to_uint8(img)
    lab = _rgb_to_lab_u8(rgb)
    L, A, B = cv2.split(lab)
    clip = float(max(0.01, clip))
    grid = int(max(1, grid))
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    L2 = clahe.apply(L)
    out = _lab_to_rgb_u8(cv2.merge([L2, A, B]))
    return _to_float01(out)

def gray_world(img: np.ndarray) -> np.ndarray:
    """Gray-world white balance (simple color constancy)."""
    rgb = _to_uint8(img).astype(np.float32) + 1e-6
    means = rgb.reshape(-1, 3).mean(axis=0)
    scale = rgb.mean() / means
    wb = np.clip(rgb * scale, 0, 255).astype(np.uint8)
    return _to_float01(wb)

def hsv(img: np.ndarray) -> np.ndarray:
    """HSV round-trip; useful when paired with other ops (kept as identity-ish)."""
    rgb = _to_uint8(img)
    out = _hsv_to_rgb_u8(_rgb_to_hsv_u8(rgb))
    return _to_float01(out)

def lab(img: np.ndarray) -> np.ndarray:
    """LAB round-trip; often neutral but can stabilize color space mapping."""
    rgb = _to_uint8(img)
    out = _lab_to_rgb_u8(_rgb_to_lab_u8(rgb))
    return _to_float01(out)

def gamma(img: np.ndarray, gamma: float = 1.2) -> np.ndarray:
    """Gamma correction in RGB (power-law). gamma>1 brightens dark regions moderately."""
    x = _to_float01(img)
    g = float(max(0.1, gamma))
    y = np.power(np.clip(x, 0.0, 1.0), 1.0 / g)
    return np.clip(y, 0.0, 1.0).astype(np.float32)

def gaussian_denoise(img: np.ndarray, k: int = 3, sigma: float = 0.0) -> np.ndarray:
    """Light Gaussian blur (denoise + slight smoothing)."""
    rgb = _to_uint8(img)
    k = int(max(1, k))
    if k % 2 == 0:
        k += 1
    out = cv2.GaussianBlur(rgb, (k, k), sigmaX=sigma)
    return _to_float01(out)

REGISTRY = {
    "identity": identity,
    "bnorm_lab_l": bnorm_lab_l,
    "clahe_lab": clahe_lab,
    "gray_world": gray_world,
    "hsv": hsv,
    "lab": lab,
    "gamma": gamma,
    "gaussian_denoise": gaussian_denoise,
}

def get_registry_keys():
    return sorted(list(REGISTRY.keys()))

if __name__ == "__main__":
    for k in get_registry_keys():
        print(k)
