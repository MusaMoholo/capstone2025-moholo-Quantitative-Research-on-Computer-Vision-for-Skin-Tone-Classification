# scripts/make_splits_from_csv.py
import csv, json, os, pathlib, random
from collections import defaultdict, Counter

RAW_ROOT   = "data/raw"
CSV_PATH   = "data/fitzpatrick17k.csv"
IMAGES_OUT = "manifests/images.csv"
SPLITS_OUT = "manifests/splits.json"
COUNTS_OUT = "manifests/split_counts.json"
SEED       = 30
SPLITS     = (0.8, 0.1, 0.1)
VALID_SCALES = {1,2,3,4,5,6}

os.makedirs("manifests", exist_ok=True)

fs_index = {}
for scale in sorted(VALID_SCALES):
    folder = pathlib.Path(RAW_ROOT, str(scale))
    if not folder.exists():
        continue
    for p in folder.iterdir():
        if p.is_file() and len(p.stem) == 32:
            md5 = p.stem.lower()
            rel = f"{scale}/{p.name}"
            fs_index[md5] = (scale, rel, p.name)

rows = []
n_total_csv = 0
n_bad_scale = 0
n_missing   = 0

with open(CSV_PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        n_total_csv += 1
        md5 = (r.get("md5hash") or "").strip().lower()
        try:
            fp = int(float(r.get("fitzpatrick_scale") or -1))
        except ValueError:
            fp = -1
        if fp not in VALID_SCALES:
            n_bad_scale += 1
            continue
        if md5 not in fs_index:
            n_missing += 1
            continue
        scale_folder, rel_path, filename = fs_index[md5]
        class_idx = fp - 1
        rows.append({
            "image_id": md5,
            "fitzpatrick_scale": fp,
            "class_idx": class_idx,
            "filename": filename,
            "rel_path": rel_path
        })

filenames = [r["filename"] for r in rows]
dupes = [k for k,v in Counter(filenames).items() if v > 1]
assert not dupes, f"Duplicate filenames found: {dupes[:5]} (and more)"

with open(IMAGES_OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["image_id","fitzpatrick_scale","class_idx","filename","rel_path"])
    for r in rows:
        w.writerow([r["image_id"], r["fitzpatrick_scale"], r["class_idx"], r["filename"], r["rel_path"]])

by_class = defaultdict(list)
for r in rows:
    by_class[r["class_idx"]].append(r["filename"])

rng = random.Random(SEED)
for L in by_class.values():
    rng.shuffle(L)

train, val, test = [], [], []
for c, L in sorted(by_class.items()):
    n = len(L)
    ntr = int(n * SPLITS[0])
    nv  = int(n * SPLITS[1])
    train += L[:ntr]
    val   += L[ntr:ntr+nv]
    test  += L[ntr+nv:]

with open(SPLITS_OUT, "w") as f:
    json.dump({"seed": SEED, "train": train, "val": val, "test": test}, f, indent=2)

def counts_by_class(file_list):
    name_to_cls = {r["filename"]: r["class_idx"] for r in rows}
    c = Counter(name_to_cls[n] for n in file_list)
    return {int(k): int(c.get(k,0)) for k in range(len(by_class))}

counts = {
    "total_csv_rows": n_total_csv,
    "excluded_bad_scale": n_bad_scale,
    "excluded_missing_on_disk": n_missing,
    "kept_total": len(rows),
    "train": counts_by_class(train),
    "val": counts_by_class(val),
    "test": counts_by_class(test),
}

with open(COUNTS_OUT, "w") as f:
    json.dump(counts, f, indent=2)

print(f"Wrote {IMAGES_OUT}, {SPLITS_OUT}, {COUNTS_OUT}")
print(f"Kept: {len(rows)} | Excluded bad_scale: {n_bad_scale} | Missing on disk: {n_missing}")
print(f"Counts per split (by class): {counts['train']} / {counts['val']} / {counts['test']}")