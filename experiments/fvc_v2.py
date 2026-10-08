"""Improved multi-session fine-tuning on FVC + Neurotechnology data (about 30 minutes on a T4).

    python fvc_v2.py --root /content/fvc --weights encoder_seed0.pt --size 192 --out results_fvc_v2

Same protocol as fvc_eval.py (2-fold finger-disjoint cross-validation over the pooled real
databases, 3 seeds, all 28 genuine pairs per finger, impostors within each database), with four
changes aimed at real multi-session variation:
  * larger input (--size, default 192 instead of 96), so less ridge detail is lost;
  * stronger augmentation: rotation +-30 deg, scale 0.85-1.15, shift +-size/8, and random erasing
    of a rectangle (simulates partial contact / small overlap);
  * a gentler ArcFace for the small number of fingers (m = 0.3, s = 32) and a 10x lower learning
    rate for the pre-trained backbone than for the new projection and head;
  * more epochs (default 60) with a short warm-up.
Writes <out>/fvc_v2.json with per-database and pooled EER / FMR100 / FMR1000 (mean +- std over seeds).
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.append(str(_Path(__file__).resolve().parents[1]))  # repo root, for the securefinger and server packages
import argparse
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import fvc_eval as F
from securefinger.encoder import ArcFaceHead, FingerprintEncoder
from securefinger.preprocess import _roi_crop

DEV = F.DEV


def make_prep(size):
    cache = {}

    def prep(p):
        if p not in cache:
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            img = _roi_crop(img)
            img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
            cache[p] = ((img - img.mean()) / (img.std() + 1e-6))[None]
        return cache[p]
    return prep


class StrongAug(Dataset):
    def __init__(self, items, prep, size):
        self.items, self.prep, self.size = items, prep, size

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        p, y = self.items[k]
        n = self.size
        x = self.prep(p)[0]
        M = cv2.getRotationMatrix2D((n / 2, n / 2), np.random.uniform(-30, 30), np.random.uniform(0.85, 1.15))
        M[:, 2] += np.random.uniform(-n / 8, n / 8, 2)
        x = cv2.warpAffine(x, M, (n, n), borderValue=0.0)
        if np.random.rand() < 0.5:  # random erasing: partial contact
            h, w = np.random.randint(n // 6, n // 2, 2)
            y0, x0 = np.random.randint(0, n - h), np.random.randint(0, n - w)
            x[y0:y0 + h, x0:x0 + w] = 0.0
        return torch.from_numpy(x[None].astype(np.float32)), y


def finetune(weights, train_fingers, prep, size, epochs, seed, lr, m, s, workers):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = FingerprintEncoder().to(DEV)
    model.load_state_dict(torch.load(weights, map_location=DEV)["encoder"])
    keys = sorted(train_fingers)
    items = [(p, n) for n, k in enumerate(keys) for p in train_fingers[k]]
    for p, _ in items:  # fill the cache before workers fork
        prep(p)
    head = ArcFaceHead(256, len(keys), s=s, m=m).to(DEV)
    opt = torch.optim.AdamW([{"params": model.backbone.parameters(), "lr": lr / 10},
                             {"params": list(model.proj.parameters()) + list(head.parameters()), "lr": lr}],
                            weight_decay=5e-4)
    warm = 3
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda e: (e + 1) / warm if e < warm else 0.5 * (1 + np.cos(np.pi * (e - warm) / max(1, epochs - warm))))
    dl = DataLoader(StrongAug(items, prep, size), batch_size=64, shuffle=True, drop_last=True,
                    num_workers=workers, persistent_workers=workers > 0)
    for _ in range(epochs):
        model.train(); head.train()
        for x, y in dl:
            loss = head(model(x.to(DEV)), y.to(DEV))
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="fvc_data")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", default="results_fvc_v2")
    ap.add_argument("--size", type=int, default=192)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--m", type=float, default=0.3)
    ap.add_argument("--s", type=float, default=32.0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-download", action="store_true")
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if not a.no_download:
        F.download(root)
    dbs = F.scan(root)
    real = [d for d in dbs if d not in F.SYNTHETIC]
    prep = make_prep(a.size)
    F.prep = prep  # db_scores/embed in fvc_eval use this preprocessing from now on
    cfg = {k: getattr(a, k) for k in ("size", "epochs", "lr", "m", "s", "seeds")}
    print("config:", cfg, "device:", DEV, flush=True)

    allf = [(d, k) for d in real for k in dbs[d]]
    runs, t0 = [], time.time()
    for seed in a.seeds:
        rng = random.Random(seed); order = allf[:]; rng.shuffle(order)
        folds = [order[0::2], order[1::2]]
        G, I, per_db = [], [], {}
        for f in (0, 1):
            tr = {f"{d}:{k}": dbs[d][k] for d, k in folds[f]}
            te = {}
            for d, k in folds[1 - f]:
                te.setdefault(d, {})[k] = dbs[d][k]
            model = finetune(a.weights, tr, prep, a.size, a.epochs, seed * 10 + f, a.lr, a.m, a.s, a.workers)
            for d, fingers in te.items():
                if len(fingers) < 2:
                    continue
                g, i = F.db_scores(model, fingers)
                G.append(g); I.append(i)
                per_db.setdefault(d, ([], []))
                per_db[d][0].append(g); per_db[d][1].append(i)
            print(f"  seed {seed} fold {f} done ({time.time() - t0:.0f}s)", flush=True)
        run = {"seed": seed, "pooled_real": F.metrics(np.concatenate(G), np.concatenate(I)),
               "per_db": {d: F.metrics(np.concatenate(v[0]), np.concatenate(v[1])) for d, v in per_db.items()}}
        runs.append(run)
        print(f"seed {seed}: pooled EER {run['pooled_real']['eer'] * 100:.2f}%  "
              f"FMR100 {run['pooled_real']['fnmr_at_fmr_1e-2'] * 100:.2f}%", flush=True)
        res = {"config": cfg, "runs": runs}
        json.dump(res, open(out / "fvc_v2.json", "w"), indent=1)

    summ = {}
    for k in ("eer", "fnmr_at_fmr_1e-2", "fnmr_at_fmr_1e-3"):
        v = [r["pooled_real"][k] for r in runs]
        summ[k] = {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0}
    per_db = {}
    for d in runs[0]["per_db"]:
        v = [r["per_db"][d]["eer"] for r in runs if d in r["per_db"]]
        per_db[d] = {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0}
    res = {"config": cfg, "runs": runs, "summary": summ, "per_db_eer": per_db, "minutes": (time.time() - t0) / 60}
    json.dump(res, open(out / "fvc_v2.json", "w"), indent=1)
    print("\n=== pooled real, mean +- std over seeds ===")
    for k, v in summ.items():
        print(f"  {k:18s} {v['mean'] * 100:6.2f} +- {v['std'] * 100:.2f} %")
    for d, v in per_db.items():
        print(f"  {d:16s} EER {v['mean'] * 100:6.2f} +- {v['std'] * 100:.2f} %")
    print(f"wrote {out}/fvc_v2.json  ({res['minutes']:.1f} min)")


if __name__ == "__main__":
    main()
