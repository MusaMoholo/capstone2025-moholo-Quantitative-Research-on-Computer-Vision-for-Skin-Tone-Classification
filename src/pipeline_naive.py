# src/pipeline_naive.py
import os
import json, argparse, csv, random



def _should_force_cpu():
    return os.getenv("FORCE_CPU", "0") == "1"

def _apply_cpu_env_defaults():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", os.getenv("TF_INTRA_OP", "8"))
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", os.getenv("TF_INTER_OP", "1"))

def _maybe_force_cpu_env(force: bool):
    if force:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

_early_ap = argparse.ArgumentParser(add_help=False)
_early_ap.add_argument("--device", choices=["auto","cpu","gpu"], default="auto")
_early_args, _ = _early_ap.parse_known_args()

_force_cpu = _should_force_cpu() or (_early_args.device == "cpu")
_maybe_force_cpu_env(_force_cpu)
if _force_cpu:
    _apply_cpu_env_defaults()

import numpy as np
import tensorflow as tf

_gpus = tf.config.list_physical_devices('GPU')
_on_gpu = (not _force_cpu) and len(_gpus) > 0

if _on_gpu:
    try:
        for gpu in _gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass
    try:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy("mixed_float16")
        _mixed_precision = True
    except Exception:
        _mixed_precision = False
else:
    try:
        tf.config.threading.set_inter_op_parallelism_threads(int(os.getenv("TF_INTER_OP", "1")))
        tf.config.threading.set_intra_op_parallelism_threads(int(os.getenv("TF_INTRA_OP", "8")))
    except Exception:
        pass
    _mixed_precision = False

SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

from src.dataio import make_dataset
from src.train_stage import fit_model
from src.evaluate_stage import evaluate

RESULTS_CSV = "outputs/results.csv"
RESULTS_HEADER = ["exp_id","preproc","postproc","macro_f1","eo_gap","dp_delta","latency_ms"]

def upsert_results_row(results_csv, row, header):
    """Replace the row with same exp_id, or append if not present."""
    rows = []
    existing_header = None
    if os.path.exists(results_csv):
        with open(results_csv, newline="") as f:
            reader = csv.reader(f)
            try:
                existing_header = next(reader)
            except StopIteration:
                existing_header = None
            if existing_header == header:
                for r in reader:
                    if r and r[0] != row["exp_id"]:
                        rows.append(r)
            else:
                rows = []
    rows.append([
        row["exp_id"], row["preproc"], row["postproc"],
        row["macro_f1"], row["eo_gap"], row["dp_delta"], row["latency_ms"]
    ])
    os.makedirs(os.path.dirname(results_csv), exist_ok=True)
    with open(results_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)

def main(exp_id="naive_identity", epochs=20, img_size=224, batch=32, lr=3e-4):
    run_dir = f"outputs/runs/{exp_id}"
    os.makedirs(run_dir, exist_ok=True)

    ds_train = make_dataset("identity", "train", img_size, batch, augment_train=True)
    ds_val   = make_dataset("identity", "val",   img_size, batch, augment_train=False)
    ds_test  = make_dataset("identity", "test",  img_size, batch, augment_train=False)

    model, history = fit_model(ds_train, ds_val, epochs=epochs, lr=lr, num_classes=6, img_size=img_size)

    json.dump(
        {"exp_id": exp_id, "preproc": "identity", "postproc": "none",
         "class_weights": None, "epochs": epochs, "lr": lr,
         "img_size": img_size, "batch": batch, "seed": SEED,
         "device": ("gpu" if _on_gpu else "cpu"),
         "mixed_precision": _mixed_precision},
        open(f"{run_dir}/config.json", "w"), indent=2
    )
    json.dump(history, open(f"{run_dir}/history.json", "w"), indent=2)

    metrics, probs, y = evaluate(model, ds_test, run_dir, calibrated=False, T=None)
    print(f"[{exp_id}] macro_F1={metrics['macro_f1']:.3f}  EO_gap={metrics['eo_gap']:.3f}  DPΔ={metrics['dp_delta']:.3f}")

    model.save(f"{run_dir}/model.keras")

    row = {
        "exp_id": exp_id,
        "preproc": "identity",
        "postproc": "none",
        "macro_f1": metrics["macro_f1"],
        "eo_gap": metrics["eo_gap"],
        "dp_delta": metrics["dp_delta"],
        "latency_ms": metrics.get("latency_ms_per_image_cpu", None),
    }
    upsert_results_row(RESULTS_CSV, row, RESULTS_HEADER)

    tf.keras.backend.clear_session()
    import gc; gc.collect()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_id", default="naive_identity")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", choices=["auto","cpu","gpu"], default="auto",
                    help="auto: use GPU if available; cpu: force CPU; gpu: require GPU")
    args = ap.parse_args()

    if args.device == "gpu" and len(tf.config.list_physical_devices('GPU')) == 0:
        raise RuntimeError("`--device gpu` requested but no GPU is visible. Use --device auto or cpu.")

    main(exp_id=args.exp_id, epochs=args.epochs, img_size=args.img_size, batch=args.batch, lr=args.lr)
