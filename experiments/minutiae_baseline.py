"""Conventional minutiae baseline (SourceAFIS) on the same multi-session pairs as our encoder.

    python minutiae_baseline.py --root /content/fvc --out results_minutiae

Uses exactly the genuine/impostor pairs of fvc_eval.py and fvc_v2.py:
  * "all fingers": every real database on its own (the zero-shot protocol), pooled;
  * "CV test folds": the test half of each 2-fold split for seeds 0-2 (the fine-tuning protocol),
    pooled per seed, so the numbers compare directly with the fine-tuned encoder (16.2%).
SourceAFIS needs no training. Images are converted to PNG and given their sensor resolution (DPI),
templates are extracted once, and the listed pairs are scored by tools/sourceafis/Match.java.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.append(str(_Path(__file__).resolve().parents[1]))  # repo root, for the securefinger and server packages
import argparse
import itertools
import json
import random
import subprocess
from pathlib import Path

import cv2
import numpy as np

import fvc_eval as F

# sensor resolution of each database (FVC reports and Neurotechnology sample notes); default 500
DPI = {"FVC2002_DB2_B": 569, "FVC2004_DB3_B": 512}
TOOL = Path(__file__).resolve().parents[1] / "tools" / "sourceafis"


def pairs_like_db_scores(fingers, max_imp=20000, seed=0):
    """Reproduce fvc_eval.db_scores pair construction exactly (same ordering, same impostor sample)."""
    keys = sorted(fingers)
    paths = [p for k in keys for p in fingers[k]]
    owner = [k for k in keys for _ in fingers[k]]
    idx = {}
    for n, k in enumerate(owner):
        idx.setdefault(k, []).append(n)
    gen = [(paths[a], paths[b]) for k in keys for a, b in itertools.combinations(idx[k], 2)]
    pairs = [(a, b) for a, b in itertools.combinations(range(len(paths)), 2) if owner[a] != owner[b]]
    if len(pairs) > max_imp:
        pairs = random.Random(seed).sample(pairs, max_imp)
    imp = [(paths[a], paths[b]) for a, b in pairs]
    return gen, imp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="fvc_data")
    ap.add_argument("--out", default="results_minutiae")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--no-download", action="store_true")
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if not a.no_download:
        F.download(root)
    dbs = F.scan(root)
    real = [d for d in dbs if d not in F.SYNTHETIC]

    # ---- protocols -> list of (label, db, genuine pairs, impostor pairs)
    protocols = {"all_fingers": []}
    for d in real:
        g, i = pairs_like_db_scores(dbs[d])
        protocols["all_fingers"].append((d, g, i))
    allf = [(d, k) for d in real for k in dbs[d]]
    for seed in a.seeds:
        rng = random.Random(seed); order = allf[:]; rng.shuffle(order)
        folds = [order[0::2], order[1::2]]
        lst = []
        for f in (0, 1):
            te = {}
            for d, k in folds[1 - f]:
                te.setdefault(d, {})[k] = dbs[d][k]
            for d, fingers in te.items():
                if len(fingers) < 2:
                    continue
                g, i = pairs_like_db_scores(fingers)
                lst.append((d, g, i))
        protocols[f"cv_seed{seed}"] = lst

    # ---- images (PNG) and unique pairs
    png_dir = out / "png"; png_dir.mkdir(exist_ok=True)
    images, index = [], {}
    for d in real:
        for k, ps in dbs[d].items():
            for p in ps:
                q = png_dir / d / (Path(p).stem + ".png")
                q.parent.mkdir(exist_ok=True)
                if not q.exists():
                    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                    cv2.imwrite(str(q), img)
                index[str(p)] = len(images)
                images.append((str(q), DPI.get(d, 500)))
    uniq = {}
    for lst in protocols.values():
        for _, g, i in lst:
            for pa, pb in g + i:
                uniq.setdefault((index[str(pa)], index[str(pb)]), None)
    pair_list = list(uniq)
    (out / "images.tsv").write_text("".join(f"{n}\t{q}\t{dpi}\n" for n, (q, dpi) in enumerate(images)))
    (out / "pairs.tsv").write_text("".join(f"{x}\t{y}\n" for x, y in pair_list))
    print(f"{len(images)} images, {len(pair_list)} unique pairs -> SourceAFIS", flush=True)
    cp = f"{TOOL}/lib/*:{TOOL}"
    subprocess.run(["java", "-Xmx6g", "-cp", cp, "Match", str(out / "images.tsv"), str(out / "pairs.tsv"),
                    str(out / "scores.txt")], check=True)
    sc = np.loadtxt(out / "scores.txt")
    score = {p: s for p, s in zip(pair_list, sc)}

    def S(pairs):
        return np.array([score[(index[str(x)], index[str(y)])] for x, y in pairs])

    res = {"dpi": {d: DPI.get(d, 500) for d in real}}
    per_db, G, I = {}, [], []
    for d, g, i in protocols["all_fingers"]:
        gs, is_ = S(g), S(i)
        per_db[d] = F.metrics(gs, is_); G.append(gs); I.append(is_)
    res["all_fingers"] = {"per_db": per_db, "pooled_real": F.metrics(np.concatenate(G), np.concatenate(I))}
    runs = []
    for seed in a.seeds:
        G, I = [], []
        for _, g, i in protocols[f"cv_seed{seed}"]:
            G.append(S(g)); I.append(S(i))
        runs.append({"seed": seed, "pooled_real": F.metrics(np.concatenate(G), np.concatenate(I))})
    summ = {}
    for k in ("eer", "fnmr_at_fmr_1e-2", "fnmr_at_fmr_1e-3"):
        v = [r["pooled_real"][k] for r in runs]
        summ[k] = {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0}
    res["cv_test_folds"] = {"runs": runs, "summary": summ}
    json.dump(res, open(out / "minutiae.json", "w"), indent=1)

    print("\n=== SourceAFIS (minutiae) baseline ===")
    p = res["all_fingers"]["pooled_real"]
    print(f"  all fingers, pooled : EER {p['eer'] * 100:6.2f}%  FMR100 {p['fnmr_at_fmr_1e-2'] * 100:6.2f}%")
    for k, v in summ.items():
        print(f"  CV test folds {k:18s} {v['mean'] * 100:6.2f} +- {v['std'] * 100:.2f} %")
    for d, m in per_db.items():
        print(f"  {d:16s} EER {m['eer'] * 100:6.2f}%")
    print(f"wrote {out}/minutiae.json")


if __name__ == "__main__":
    main()
