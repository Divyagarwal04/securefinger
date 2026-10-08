"""Multi-session evaluation with the training recipe chosen on held-out data (nested selection).

    python fvc_nested.py --root /content/fvc --weights encoder_seed0.pt --out results_nested

Same outer protocol as fvc_eval.py / fvc_v2.py (2-fold finger-disjoint cross-validation over the
pooled real databases, seeds 0-2, impostors only within each database). Inside every outer training
fold the fingers are split again, per database, into inner-train (about 75%) and inner-validation
(about 25%, at least 2 fingers per database). Three candidate recipes are fine-tuned on inner-train and
scored on inner-validation:
    R1  first recipe   : 96 px,  40 epochs, rotation +-20, ArcFace m=0.5 s=64     (fvc_eval.finetune)
    R2  new recipe     : 96 px,  60 epochs, stronger aug + erasing, m=0.3 s=32   (fvc_v2.finetune)
    R3  new recipe     : 192 px, 60 epochs, stronger aug + erasing, m=0.3 s=32   (fvc_v2.finetune)
The recipe with the lowest inner-validation EER is retrained on the whole outer training fold and
tested once on the outer test fold, which it has never seen. Writes <out>/nested.json.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.append(str(_Path(__file__).resolve().parents[1]))  # repo root, for the securefinger and server packages
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

import fvc_eval as F
import fvc_v2 as V

PREP96 = F.prep                     # fvc_eval's original 96 px preprocessing
PREP = {96: PREP96}


def prep_for(size):
    if size not in PREP:
        PREP[size] = V.make_prep(size)
    return PREP[size]


RECIPES = {
    "R1_first_96": {"kind": "v1", "size": 96, "epochs": 40},
    "R2_new_96": {"kind": "v2", "size": 96, "epochs": 60},
    "R3_new_192": {"kind": "v2", "size": 192, "epochs": 60},
}


def train(recipe, weights, fingers, seed, workers):
    r = RECIPES[recipe]
    F.prep = prep_for(r["size"])            # training and evaluation of this recipe use its own input size
    if r["kind"] == "v1":
        return F.finetune(weights, fingers, r["epochs"], seed)
    return V.finetune(weights, fingers, F.prep, r["size"], r["epochs"], seed, 1e-3, 0.3, 32.0, workers)


def score(model, recipe, by_db):
    F.prep = prep_for(RECIPES[recipe]["size"])
    G, I = [], []
    for d, fingers in by_db.items():
        if len(fingers) < 2:
            continue
        g, i = F.db_scores(model, fingers)
        G.append(g); I.append(i)
    return np.concatenate(G), np.concatenate(I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="fvc_data")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", default="results_nested")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--quick", action="store_true", help="1 epoch per recipe (smoke test only)")
    a = ap.parse_args()
    if a.quick:
        for r in RECIPES.values():
            r["epochs"] = 1
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    root = Path(a.root)
    if not a.no_download:
        F.download(root)
    dbs = F.scan(root)
    real = [d for d in dbs if d not in F.SYNTHETIC]
    allf = [(d, k) for d in real for k in dbs[d]]
    runs, t0 = [], time.time()
    for seed in a.seeds:
        rng = random.Random(seed); order = allf[:]; rng.shuffle(order)
        folds = [order[0::2], order[1::2]]               # identical to fvc_eval / fvc_v2
        G, I, choices = [], [], []
        for f in (0, 1):
            train_pairs, test_pairs = folds[f], folds[1 - f]
            # inner split, per database, finger-disjoint
            irng = random.Random(1000 + seed * 10 + f)
            per_db = {}
            for d, k in train_pairs:
                per_db.setdefault(d, []).append(k)
            inner_tr, inner_va = {}, {}
            for d, ks in per_db.items():
                ks = sorted(ks); irng.shuffle(ks)
                n_val = max(2, round(0.25 * len(ks))) if len(ks) >= 4 else 0
                for k in ks[:n_val]:
                    inner_va.setdefault(d, {})[k] = dbs[d][k]
                for k in ks[n_val:]:
                    inner_tr[f"{d}:{k}"] = dbs[d][k]
            val_eer = {}
            for rec in RECIPES:
                m = train(rec, a.weights, inner_tr, seed * 10 + f, a.workers)
                g, i = score(m, rec, inner_va)
                val_eer[rec] = float(F.eer(g, i)[0])
                print(f"  seed {seed} fold {f} {rec:12s} inner-val EER {val_eer[rec] * 100:6.2f}%  ({time.time() - t0:.0f}s)", flush=True)
            best = min(val_eer, key=val_eer.get)
            full_tr = {f"{d}:{k}": dbs[d][k] for d, k in train_pairs}
            te = {}
            for d, k in test_pairs:
                te.setdefault(d, {})[k] = dbs[d][k]
            m = train(best, a.weights, full_tr, seed * 10 + f, a.workers)
            g, i = score(m, best, te)
            G.append(g); I.append(i)
            choices.append({"fold": f, "inner_val_eer": val_eer, "chosen": best,
                            "test_fold_eer": float(F.eer(g, i)[0])})
            print(f"  seed {seed} fold {f}: chose {best}, outer test EER {choices[-1]['test_fold_eer'] * 100:.2f}%", flush=True)
        run = {"seed": seed, "folds": choices, "pooled_real": F.metrics(np.concatenate(G), np.concatenate(I))}
        runs.append(run)
        print(f"seed {seed}: pooled EER {run['pooled_real']['eer'] * 100:.2f}%  "
              f"FMR100 {run['pooled_real']['fnmr_at_fmr_1e-2'] * 100:.2f}%", flush=True)
        json.dump({"runs": runs}, open(out / "nested.json", "w"), indent=1)
    summ = {}
    for k in ("eer", "fnmr_at_fmr_1e-2", "fnmr_at_fmr_1e-3"):
        v = [r["pooled_real"][k] for r in runs]
        summ[k] = {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0}
    chosen = [c["chosen"] for r in runs for c in r["folds"]]
    res = {"recipes": RECIPES, "runs": runs, "summary": summ,
           "chosen_counts": {r: chosen.count(r) for r in RECIPES}, "minutes": (time.time() - t0) / 60}
    json.dump(res, open(out / "nested.json", "w"), indent=1)
    print("\n=== nested selection, pooled real, mean +- std over seeds ===")
    for k, v in summ.items():
        print(f"  {k:18s} {v['mean'] * 100:6.2f} +- {v['std'] * 100:.2f} %")
    print("  recipe chosen per fold:", res["chosen_counts"])
    print(f"wrote {out}/nested.json  ({res['minutes']:.1f} min)")


if __name__ == "__main__":
    main()
