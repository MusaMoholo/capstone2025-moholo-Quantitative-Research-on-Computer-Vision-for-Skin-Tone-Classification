# src/evaluate_stage.py
import csv
import json
import time
import numpy as np
from src.metrics import macro_f1, per_class_f1, confmat, dp_delta, eo_gap

def get_probs_and_labels(model, ds):
    """Run the model over a tf.data.Dataset and collect probs + labels.
    Returns:
      probs: (N, 6) float array of class probabilities/logits post-softmax
      ys:    (N,) int array of true labels in 0..5
      latency_ms_per_image: average inference latency per image (ms)
    """
    probs, ys = [], []
    t0 = time.perf_counter()
    n_imgs = 0
    for x, y in ds:
        p = model(x, training=False).numpy()
        probs.append(p)
        ys.append(y.numpy())
        n_imgs += len(y)
    latency_ms = (time.perf_counter() - t0) * 1000.0 / max(1, n_imgs)
    return np.vstack(probs), np.concatenate(ys), latency_ms

def evaluate(model, ds_test, out_dir, calibrated: bool = False, T=None):
    """Evaluate a trained model on ds_test and write preds + metrics."""
    probs, y, latency_ms = get_probs_and_labels(model, ds_test)

    assert probs.ndim == 2 and probs.shape[1] == 6, f"probs shape {probs.shape}, expected (*,6)"

    if calibrated and T is not None:
        from src.calibrate_stage import apply_temp
        probs = apply_temp(np.asarray(probs, dtype=np.float32), T)
        assert probs.ndim == 2 and probs.shape[1] == 6, f"[calibrated] probs shape {probs.shape}, expected (*,6)"

    yhat = probs.argmax(axis=1)

    uy = np.unique(y)
    uyh = np.unique(yhat)
    assert set(uy).issubset(set(range(6))), f"y contains out-of-range labels: {uy}"
    assert set(uyh).issubset(set(range(6))), f"yhat contains out-of-range labels: {uyh}"

    pred_hist = np.bincount(yhat, minlength=6).tolist()
    true_hist = np.bincount(y, minlength=6).tolist()

    if len(uyh) == 1:
        print("[WARN] degenerate predictions: model predicted a single class on test set.")

    m = {
        "accuracy": float((yhat == y).mean()),
        "macro_f1": macro_f1(y, yhat),
        "per_class_f1": per_class_f1(y, yhat),
        "dp_delta": dp_delta(y, yhat),
        "eo_gap": eo_gap(y, yhat),
        "latency_ms_per_image_cpu": float(latency_ms),
        "pred_hist": pred_hist,
        "true_hist": true_hist,
    }

    with open(f"{out_dir}/preds_test.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "y_true", "y_pred"])
        for i, (yt, yp) in enumerate(zip(y, yhat)):
            w.writerow([i, int(yt), int(yp)])

    with open(f"{out_dir}/metrics.json", "w") as f:
        json.dump(m, f, indent=2)

    return m, probs, y

