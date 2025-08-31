import os, json, argparse, csv, random, time
import numpy as np

# ---- device control (cpu/gpu/auto) ----
def _maybe_set_device(device: str):
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

_early = argparse.ArgumentParser(add_help=False)
_early.add_argument("--device", choices=["auto","cpu","gpu"], default="auto")
_early_args, _ = _early.parse_known_args()
_must_cpu = (_early_args.device == "cpu")
_maybe_set_device(_early_args.device)

import tensorflow as tf

if _early_args.device != "gpu":
    try:
        tf.config.threading.set_inter_op_parallelism_threads(int(os.getenv("TF_INTER_OP","1")))
        tf.config.threading.set_intra_op_parallelism_threads(int(os.getenv("TF_INTRA_OP","8")))
    except Exception:
        pass

SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# ---- project imports (reuse Stage-1 utilities) ----
from src.dataio import make_dataset
from src.model_zoo import build_mobilenetv3, compile_model
from src.evaluate_stage import evaluate, get_probs_and_labels
from src.calibrate_stage import fit_temperature

RESULTS_CSV = "outputs/results.csv"
RESULTS_HEADER = ["exp_id","preproc","postproc","macro_f1","eo_gap","dp_delta","latency_ms"]

def _upsert_results_row(results_csv, row, header):
    rows = []
    existing_header = None
    if os.path.exists(results_csv):
        with open(results_csv, newline="") as f:
            reader = csv.reader(f)
            try: existing_header = next(reader)
            except StopIteration: existing_header = None
            if existing_header == header:
                for r in reader:
                    if r and r[0] != row["exp_id"]:
                        rows.append(r)
    rows.append([row["exp_id"], row["preproc"], row["postproc"],
                 row["macro_f1"], row["eo_gap"], row["dp_delta"], row["latency_ms"]])
    os.makedirs(os.path.dirname(results_csv), exist_ok=True)
    with open(results_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)

# ---- class weights (same recipe as Stage-1) ----
def _train_class_counts(images_csv="manifests/images.csv", splits_json="manifests/splits.json"):
    import json, csv
    name_to_cls = {}
    with open(images_csv) as f:
        r = csv.DictReader(f)
        for row in r:
            name_to_cls[row["filename"]] = int(row["class_idx"])
    splits = json.load(open(splits_json))
    counts = {i:0 for i in range(6)}
    for fname in splits["train"]:
        counts[name_to_cls[fname]] += 1
    return counts

def _class_weights_from_counts(counts):
    N = sum(counts.values()); K = len(counts)
    w = {c: N/(K*counts[c]) for c in counts}
    mean_w = sum(w.values())/K
    w = {c: min(3.0, w[c]/mean_w) for c in w}
    return w

# ---- backbone detection + unfreeze helper ----
def _find_backbone(model: tf.keras.Model):
    try:
        return model.get_layer("backbone")
    except Exception:
        pass
    submodels = [l for l in model.layers if isinstance(l, tf.keras.Model)]
    if submodels:
        return max(submodels, key=lambda m: len(m.layers))
    for l in model.layers:
        if "mobilenet" in l.name.lower():
            return l
    raise RuntimeError("Could not locate backbone submodel. Consider naming it 'backbone' in model_zoo.")

def _set_backbone_trainable_depth(model, last_n, train_bn=False):
    bb = _find_backbone(model)
    for l in bb.layers:
        l.trainable = False
    trainable = 0
    for l in reversed(bb.layers):
        if trainable >= last_n:
            break
        if isinstance(l, tf.keras.layers.BatchNormalization) and not train_bn:
            continue
        if l.weights:
            l.trainable = True
            trainable += 1
    bb.trainable = (last_n > 0)
    return trainable

def main(preproc, exp_id, img_size=224, batch=32,
         head_epochs=8, ft_epochs=6, last_n=60, train_bn=False,
         lr_head=3e-4, lr_ft=1e-4,
         images_csv="manifests/images.csv", splits_json="manifests/splits.json", label_smoothing=0.0):
    run_dir = f"outputs/runs/{exp_id}"
    os.makedirs(run_dir, exist_ok=True)

    ds_train = make_dataset(preproc, "train", img_size, batch, augment_train=True,
                            images_csv=images_csv, splits_json=splits_json)
    ds_val   = make_dataset(preproc, "val",   img_size, batch, augment_train=False,
                            images_csv=images_csv, splits_json=splits_json)
    ds_test  = make_dataset(preproc, "test",  img_size, batch, augment_train=False,
                            images_csv=images_csv, splits_json=splits_json)

    counts = _train_class_counts(images_csv, splits_json)
    cw = _class_weights_from_counts(counts)

    model = build_mobilenetv3(num_classes=6, img_size=img_size, freeze_backbone=True)
    compile_model(model, lr_head, label_smoothing=label_smoothing)
    es1 = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)
    hist1 = model.fit(ds_train, validation_data=ds_val, epochs=head_epochs,
                      class_weight=cw, callbacks=[es1], verbose=2)

    trained_layers = _set_backbone_trainable_depth(model, last_n=last_n, train_bn=train_bn)
    compile_model(model, lr_ft, label_smoothing=label_smoothing)
    es2 = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)
    hist2 = model.fit(ds_train, validation_data=ds_val, epochs=ft_epochs,
                      class_weight=cw, callbacks=[es2], verbose=2)

    val_probs, y_val, _ = get_probs_and_labels(model, ds_val)
    T = fit_temperature(np.asarray(val_probs, np.float32), y_val)
    open(f"{run_dir}/T.txt","w").write(f"{T:.6f}")

    metrics, probs, y = evaluate(model, ds_test, run_dir, calibrated=True, T=T)
    print(f"[{exp_id}] macro_F1={metrics['macro_f1']:.3f}  EO_gap={metrics['eo_gap']:.3f}  DPΔ={metrics['dp_delta']:.3f}")

    model.save(f"{run_dir}/model.keras")
    json.dump({
        "exp_id": exp_id,
        "stage": "stage2_finetune",
        "preproc": preproc,
        "img_size": img_size,
        "batch": batch,
        "head_epochs": head_epochs,
        "ft_epochs": ft_epochs,
        "last_n": last_n,
        "train_bn": train_bn,
        "lr_head": lr_head,
        "lr_ft": lr_ft,
	"label_smoothing": label_smoothing,
        "class_weights": cw,
        "seed": SEED,
        "device": ("gpu" if tf.config.list_physical_devices('GPU') else "cpu"),
        "trained_backbone_layers": trained_layers
    }, open(f"{run_dir}/config.json","w"), indent=2)

    json.dump({
        "head": hist1.history,
        "finetune": hist2.history
    }, open(f"{run_dir}/history.json","w"), indent=2)

    row = {
        "exp_id": exp_id,
        "preproc": preproc,
        "postproc": "temp_scaling",
        "macro_f1": metrics["macro_f1"],
        "eo_gap": metrics["eo_gap"],
        "dp_delta": metrics["dp_delta"],
        "latency_ms": metrics.get("latency_ms_per_image_cpu", None),
    }
    _upsert_results_row(RESULTS_CSV, row, RESULTS_HEADER)

    tf.keras.backend.clear_session()
    import gc; gc.collect()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preproc", default="gamma", help="fixed preproc used in Stage-2")
    ap.add_argument("--exp_id", required=True, help="outputs/runs/<exp_id>")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--head_epochs", type=int, default=8)
    ap.add_argument("--ft_epochs", type=int, default=6)
    ap.add_argument("--last_n", type=int, default=60, help="unfreeze last N backbone layers")
    ap.add_argument("--train_bn", action="store_true", help="also train BN layers (default False)")
    ap.add_argument("--lr_head", type=float, default=3e-4)
    ap.add_argument("--lr_ft", type=float, default=1e-4)
    ap.add_argument("--device", choices=["auto","cpu","gpu"], default="auto")
    ap.add_argument("--images_csv", default="manifests/images.csv")
    ap.add_argument("--splits_json", default="manifests/splits.json")

    ap.add_argument("--label_smoothing", type=float, default=0.0,
                    help="Label smoothing ε for SparseCategoricalCrossentropy.")

    args = ap.parse_args()

    if args.device == "gpu" and len(tf.config.list_physical_devices('GPU')) == 0:
        raise RuntimeError("`--device gpu` requested but no GPU visible. Use --device auto or cpu.")

    main(args.preproc, args.exp_id, args.img_size, args.batch,
         args.head_epochs, args.ft_epochs, args.last_n, args.train_bn,
         args.lr_head, args.lr_ft, args.images_csv, args.splits_json)
