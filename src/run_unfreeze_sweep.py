import argparse, datetime as dt, json, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
def log_and_run(cmd, fh):
    print("RUN:", " ".join(cmd), flush=True)
    fh.write(json.dumps({"ts": dt.datetime.utcnow().isoformat(), "cmd": cmd}) + "\n")
    fh.flush()
    return subprocess.call(cmd)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preproc", default="gamma", help="fixed preproc for Stage-2")
    ap.add_argument("--last_n", type=int, nargs="+", default=[0,32,60,96],
                    help="list of unfreeze depths to try")
    ap.add_argument("--head_epochs", type=int, default=8)
    ap.add_argument("--ft_epochs", type=int, default=6)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", choices=["auto","cpu","gpu"], default="auto")
    ap.add_argument("--tag", type=str, default=None)
    ap.add_argument("--train_bn", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    ts = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    tag = f"_{args.tag}" if args.tag else ""
    meta_dir = ROOT / "outputs" / "sweep_meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_log = meta_dir / f"stage2_unfreeze_{ts}.jsonl"

    with open(meta_log, "a", encoding="utf-8") as fh:
        for n in args.last_n:
            exp_id = f"{args.preproc}_ftN{n}_{ts}{tag}"
            cmd = [sys.executable, "-m", "src.pipeline_finetune",
                   "--preproc", args.preproc,
                   "--exp_id", exp_id,
                   "--img_size", str(args.img_size),
                   "--batch", str(args.batch),
                   "--head_epochs", str(args.head_epochs),
                   "--ft_epochs", str(args.ft_epochs),
                   "--last_n", str(n),
                   "--device", args.device]
            if args.train_bn: cmd.append("--train_bn")
            if args.dry_run:
                print("[DRY]", " ".join(cmd))
            else:
                rc = log_and_run(cmd, fh)
                if rc != 0:
                    print(f"[WARN] ftN={n} exited with {rc}; continuing")
                    time.sleep(2)

    print(f"[INFO] Stage-2 sweep finished. Meta-log: {meta_log}")

if __name__ == "__main__":
    main()
