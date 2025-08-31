#!/usr/bin/env python3
import argparse, datetime as dt, json, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

def get_preproc_keys():
    from src.preprocessors import REGISTRY
    return sorted(list(REGISTRY.keys()))

def log_and_run(cmd, fh):
    print("RUN:", " ".join(cmd), flush=True)
    fh.write(json.dumps({"ts": dt.datetime.utcnow().isoformat(), "cmd": cmd}) + "\n")
    fh.flush()
    return subprocess.call(cmd)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--device", choices=["auto","cpu","gpu"], default="auto")
    ap.add_argument("--runs_root", type=str, default="outputs/runs")
    ap.add_argument("--only", nargs="*", help="subset of preprocessors")
    ap.add_argument("--exclude", nargs="*", help="preprocessors to skip")
    ap.add_argument("--tag", type=str, default=None, help="optional tag for exp_id")
    ap.add_argument("--skip_naive", action="store_true", help="skip naive identity baseline")
    ap.add_argument("--skip_identity_weighted", action="store_true", help="skip identity weighted+calib")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    preprocs = get_preproc_keys()
    if args.only:
        only = set(args.only); preprocs = [k for k in preprocs if k in only]
    if args.exclude:
        ex = set(args.exclude); preprocs = [k for k in preprocs if k not in ex]

    ts = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    tag = f"_{args.tag}" if args.tag else ""
    meta_dir = ROOT / "outputs" / "sweep_meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_log = meta_dir / f"sweep_{ts}.jsonl"

    with open(meta_log, "a", encoding="utf-8") as fh:
        if not args.skip_naive:
            cmd = [sys.executable, "-m", "src.pipeline_naive",
                   "--exp_id", f"identity_naive_{ts}{tag}",
                   "--epochs", str(args.epochs),
                   "--img_size", str(args.img_size),
                   "--batch", str(args.batch),
                   "--device", args.device]
            if args.dry_run: print("[DRY]", " ".join(cmd))
            else: log_and_run(cmd, fh)

        if not args.skip_identity_weighted:
            cmd = [sys.executable, "-m", "src.pipeline",
                   "--preproc", "identity",
                   "--exp_id", f"identity_weighted_calib_{ts}{tag}",
                   "--epochs", str(args.epochs),
                   "--img_size", str(args.img_size),
                   "--batch", str(args.batch),
                   "--device", args.device]
            if args.dry_run: print("[DRY]", " ".join(cmd))
            else: log_and_run(cmd, fh)

        for pre in [p for p in preprocs if p != "identity"]:
            exp_id = f"{pre}_weighted_calib_{ts}{tag}"
            cmd = [sys.executable, "-m", "src.pipeline",
                   "--preproc", pre,
                   "--exp_id", exp_id,
                   "--epochs", str(args.epochs),
                   "--img_size", str(args.img_size),
                   "--batch", str(args.batch),
                   "--device", args.device]
            if args.dry_run: print("[DRY]", " ".join(cmd))
            else:
                rc = log_and_run(cmd, fh)
                if rc != 0:
                    print(f"[WARN] {pre} exited with {rc}; continuing")
                    time.sleep(2)

    print(f"[INFO] Sweep finished. Meta-log: {meta_log}")

if __name__ == "__main__":
    main()
