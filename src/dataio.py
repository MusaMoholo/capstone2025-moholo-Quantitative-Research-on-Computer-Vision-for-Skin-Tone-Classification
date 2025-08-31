# src/dataio.py
import os, json, csv, tensorflow as tf
from src.preprocessors import REGISTRY as PRE
from src.augment import static_augment

DATA_ROOT = os.environ.get("DATA_ROOT", os.path.join("data", "raw"))

def _load_manifests(images_csv="manifests/images.csv", splits_json="manifests/splits.json"):
    """
    Returns:
      name2row: filename -> {fitzpatrick_scale, class_idx, rel_path}
      splits:   {"train": [...], "val": [...], "test": [...]}
    """
    name2row = {}
    with open(images_csv) as f:
        r = csv.DictReader(f)
        for row in r:
            name2row[row["filename"]] = {
                "fitzpatrick_scale": int(row["fitzpatrick_scale"]),
                "class_idx": int(row["class_idx"]),
                "rel_path": row["rel_path"],
            }
    splits = json.load(open(splits_json))
    return name2row, splits

def make_dataset(preproc_name, split_name, img_size=224, batch=32, augment_train=True,
                 images_csv="manifests/images.csv", splits_json="manifests/splits.json"):
    name2row, splits = _load_manifests(images_csv, splits_json)
    filenames = splits[split_name]

    paths, labels = [], []
    missing = []
    for fname in filenames:
        row = name2row.get(fname)
        if row is None:
            continue
        abs_path = os.path.join(DATA_ROOT, row["rel_path"])
        if not os.path.exists(abs_path):
            missing.append(abs_path)
            continue
        paths.append(abs_path)
        labels.append(row["class_idx"])

    if missing:
        print(f"[{split_name}] WARNING: {len(missing)} files not found under DATA_ROOT='{DATA_ROOT}'. "
              f"Using {len(paths)} samples.")
        for p in missing[:3]:
            print(f"  - missing: {p}")
    else:
        print(f"[{split_name}] Using {len(paths)} images (preproc={preproc_name}).")

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _load(path, y):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, (img_size, img_size), method=tf.image.ResizeMethod.BILINEAR, antialias=True)

        def _pp(a):
            import cv2
            fn = PRE[preproc_name]
            return fn(a)

        img = tf.py_function(_pp, [img], Tout=tf.float32)
        img.set_shape([img_size, img_size, 3])

        img = tf.clip_by_value(img, 0.0, 1.0)

        if split_name == "train" and augment_train:
            img = static_augment(img)
        return img, y

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)

    if split_name == "train":
        ds = ds.shuffle(len(paths), reshuffle_each_iteration=True)

    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    return ds
