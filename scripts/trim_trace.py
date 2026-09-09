# scripts/trim_trace.py — keep the first N training episodes plus all eval/coverage episodes
import csv, shutil, sys
from pathlib import Path
src, dst, n_train = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
(dst / "trace" / "episodes").mkdir(parents=True, exist_ok=True)
with open(src / "trace" / "index.csv", newline="") as fh:
    rows = list(csv.DictReader(fh)); cols = list(rows[0].keys())
kept, n = [], 0
for r in rows:
    train = r["phase"].startswith("train_")
    if train and n >= n_train:
        continue
    n += train
    kept.append(r); shutil.copy(src / "trace" / r["file"], dst / "trace" / r["file"])
with open(dst / "trace" / "index.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(kept)
print(f"kept {len(kept)} of {len(rows)} episodes")
