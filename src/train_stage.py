import json, csv
import tensorflow as tf
from src.model_zoo import build_mobilenetv3, compile_model

def _train_class_counts(images_csv, splits_json):
    name_to_cls = {}
    with open(images_csv) as f:
        r = csv.DictReader(f)
        for row in r:
            name_to_cls[row["filename"]] = int(row["class_idx"])
    splits = json.load(open(splits_json))
    counts = {i: 0 for i in range(6)}
    for fname in splits["train"]:
        counts[name_to_cls[fname]] += 1
    return counts

def class_weights_from_counts(counts):
    N = sum(counts.values()); K = len(counts)
    w = {c: N/(K*counts[c]) for c in counts}
    mean_w = sum(w.values())/K
    w = {c: min(3.0, w[c]/mean_w) for c in w}
    return w

def _assert_head_is_six(model):
    out = model.output_shape[-1]
    assert out == 6, f"Model outputs {out} classes, expected 6."

def fit_model(ds_train, ds_val, epochs=20, lr=3e-4, num_classes=6, img_size=224):
    model = build_mobilenetv3(num_classes=num_classes, img_size=img_size)
    _assert_head_is_six(model)
    compile_model(model, lr)
    es = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)
    history = model.fit(ds_train, validation_data=ds_val, epochs=epochs, callbacks=[es], verbose=2)
    return model, history.history

def fit_model_with_class_weights(ds_train, ds_val,
                                 images_csv="manifests/images.csv", splits_json="manifests/splits.json",
                                 epochs=20, lr=3e-4, num_classes=6, img_size=224):
    model = build_mobilenetv3(num_classes=num_classes, img_size=img_size)
    _assert_head_is_six(model)
    compile_model(model, lr)
    counts = _train_class_counts(images_csv, splits_json)
    cw = class_weights_from_counts(counts)
    es = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)
    history = model.fit(ds_train, validation_data=ds_val, epochs=epochs, callbacks=[es], verbose=2, class_weight=cw)
    return model, history.history, cw

