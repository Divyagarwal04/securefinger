"""Multi-session evaluation on public FVC and Neurotechnology sample databases.

    python fvc_eval.py --root /content/fvc --weights encoder_seed0.pt --out results_fvc

Every finger in these databases was captured 8 times in separate placements (FVC: two or three
sessions), so genuine pairs differ in pressure, rotation, partial contact and skin condition, unlike
SOCOFing's synthetic edits. Freely downloadable sets used (10 fingers x 8 impressions each, plus the
two Neurotechnology sample databases):
    FVC2000 / FVC2002 / FVC2004  DB1_B..DB4_B   (DB4 is synthetic, reported separately)
    Neurotechnology CrossMatch (51 fingers) and UareU (65 fingers) sample databases
Protocols
  * Zero-shot: the SOCOFing-trained encoder is applied unchanged.
  * Fine-tuned: 2-fold finger-disjoint cross-validation over the pooled real databases; the SOCOFing
    encoder is fine-tuned with ArcFace on one fold and tested on the other, for several seeds.
Metrics per database and pooled: EER, FNMR at FMR = 1% (FMR100) and 0.1% (FMR1000), over all genuine
pairs (28 per finger) and all impostor pairs between different fingers of the same database.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.append(str(_Path(__file__).resolve().parents[1]))  # repo root, for the securefinger and server packages
import argparse
import itertools
import json
import random
import re
import time
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from securefinger.encoder import ArcFaceHead, FingerprintEncoder
from securefinger.metrics import eer
from securefinger.preprocess import load_and_preprocess

DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
URLS = {f"FVC{y}_DB{d}_B": f"http://bias.csr.unibo.it/fvc{y}/Downloads/DB{d}_B.zip"
        for y in (2000, 2002, 2004) for d in (1, 2, 3, 4)}
URLS["NT_CrossMatch"] = "https://www.neurotechnology.com/download/CrossMatch_Sample_DB.zip"
URLS["NT_UareU"] = "https://www.neurotechnology.com/download/UareU_sample_DB.zip"
SYNTHETIC = {f"FVC{y}_DB4_B" for y in (2000, 2002, 2004)}
EXTS = {".tif", ".tiff", ".bmp", ".png", ".jpg"}


# ----------------------------------------------------------------------------- data
def download(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    for name, url in URLS.items():
        d = root / name
        if d.exists() and any(d.rglob("*")):
            continue
        z = root / f"{name}.zip"
        try:
            print("downloading", name, flush=True)
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as r, open(z, "wb") as fh:
                fh.write(r.read())
            with zipfile.ZipFile(z) as zf:
                zf.extractall(d)
        except Exception as e:  # keep going with whatever is available
            print(f"  could not get {name}: {e}", flush=True)


def scan(root: Path):
    """{db: {finger_id: [paths sorted by impression]}} from names like 101_3.tif or 012_3_2.tif."""
    dbs = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        fingers = {}
        for p in d.rglob("*"):
            if p.suffix.lower() not in EXTS or p.name.startswith("."):
                continue
            parts = re.split(r"[_\-]", p.stem)
            if len(parts) < 2 or not parts[-1].isdigit():
                continue
            fingers.setdefault("_".join(parts[:-1]), []).append(p)
        fingers = {k: sorted(v, key=lambda q: int(re.split(r"[_\-]", q.stem)[-1])) for k, v in fingers.items() if len(v) >= 2}
        if fingers:
            dbs[d.name] = fingers
            n = [len(v) for v in fingers.values()]
            print(f"  {d.name:16s} fingers={len(fingers):3d} images={sum(n):4d} impressions/finger={min(n)}-{max(n)}", flush=True)
    return dbs


_CACHE = {}


def prep(p):
    if p not in _CACHE:
        _CACHE[p] = load_and_preprocess(str(p)).astype(np.float32)
    return _CACHE[p]


@torch.no_grad()
def embed(model, paths, bs=128):
    model.eval()
    out = []
    for i in range(0, len(paths), bs):
        x = torch.from_numpy(np.stack([prep(p) for p in paths[i:i + bs]])).to(DEV)
        out.append(model(x).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


# ----------------------------------------------------------------------------- metrics
def fnmr_at_fmr(g, i, fmr):
    thr = np.quantile(i, 1 - fmr)
    return float((g <= thr).mean())


def db_scores(model, fingers, max_imp=20000, seed=0):
    keys = sorted(fingers)
    paths = [p for k in keys for p in fingers[k]]
    owner = [k for k in keys for _ in fingers[k]]
    E = embed(model, paths)
    idx = {}
    for n, k in enumerate(owner):
        idx.setdefault(k, []).append(n)
    gen = [E[a] @ E[b] for k in keys for a, b in itertools.combinations(idx[k], 2)]
    pairs = [(a, b) for a, b in itertools.combinations(range(len(paths)), 2) if owner[a] != owner[b]]
    if len(pairs) > max_imp:
        pairs = random.Random(seed).sample(pairs, max_imp)
    imp = [E[a] @ E[b] for a, b in pairs]
    return np.array(gen), np.array(imp)


def metrics(g, i):
    e, _ = eer(g, i)
    return {"genuine_pairs": int(len(g)), "impostor_pairs": int(len(i)), "eer": e,
            "fnmr_at_fmr_1e-2": fnmr_at_fmr(g, i, 1e-2), "fnmr_at_fmr_1e-3": fnmr_at_fmr(g, i, 1e-3)}


def evaluate(model, dbs, real):
    res, G, I = {}, [], []
    for name, fingers in dbs.items():
        g, i = db_scores(model, fingers)
        res[name] = metrics(g, i)
        if name in real:
            G.append(g); I.append(i)
    if G:
        res["pooled_real"] = metrics(np.concatenate(G), np.concatenate(I))
    return res


# ----------------------------------------------------------------------------- fine-tuning
class AugDS(Dataset):
    """Preprocessed 96x96 images with rotation (+-20 deg), scale (0.9-1.1) and shift (+-8 px)."""

    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        p, y = self.items[k]
        x = prep(p)[0]
        a, s = np.random.uniform(-20, 20), np.random.uniform(0.9, 1.1)
        M = cv2.getRotationMatrix2D((48, 48), a, s)
        M[:, 2] += np.random.uniform(-8, 8, 2)
        x = cv2.warpAffine(x, M, (96, 96), borderValue=0.0)
        return torch.from_numpy(x[None].astype(np.float32)), y


def finetune(weights, train_fingers, epochs, seed, lr=3e-4):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = FingerprintEncoder().to(DEV)
    model.load_state_dict(torch.load(weights, map_location=DEV)["encoder"])
    keys = sorted(train_fingers)
    items = [(p, n) for n, k in enumerate(keys) for p in train_fingers[k]]
    head = ArcFaceHead(256, len(keys)).to(DEV)
    opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    dl = DataLoader(AugDS(items), batch_size=64, shuffle=True, drop_last=True)
    for ep in range(epochs):
        model.train(); head.train()
        for x, y in dl:
            loss = head(model(x.to(DEV)), y.to(DEV))
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="fvc_data")
    ap.add_argument("--weights", required=True, help="SOCOFing-trained encoder (encoder.pt / encoder_seed0.pt)")
    ap.add_argument("--out", default="results_fvc")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not args.no_download:
        download(root)
    print("databases found:", flush=True)
    dbs = scan(root)
    real = [d for d in dbs if d not in SYNTHETIC]
    res = {"device": DEV, "databases": {d: {"fingers": len(f), "images": sum(map(len, f.values()))} for d, f in dbs.items()},
           "real_databases": real}

    t0 = time.time()
    base = FingerprintEncoder().to(DEV)
    base.load_state_dict(torch.load(args.weights, map_location=DEV)["encoder"])
    res["zero_shot"] = evaluate(base, dbs, real)
    print("zero-shot pooled real:", json.dumps(res["zero_shot"].get("pooled_real")), f"({time.time() - t0:.0f}s)", flush=True)
    json.dump(res, open(out / "fvc.json", "w"), indent=1)

    # 2-fold finger-disjoint cross-validation on the pooled real databases
    allf = [(d, k) for d in real for k in dbs[d]]
    runs = []
    for seed in args.seeds:
        rng = random.Random(seed); order = allf[:]; rng.shuffle(order)
        folds = [order[0::2], order[1::2]]
        G, I, per_db = [], [], {}
        for f in (0, 1):
            tr = {f"{d}:{k}": dbs[d][k] for d, k in folds[f]}
            te = {}
            for d, k in folds[1 - f]:
                te.setdefault(d, {})[k] = dbs[d][k]
            model = finetune(args.weights, tr, args.epochs, seed * 10 + f)
            for d, fingers in te.items():
                if len(fingers) < 2:
                    continue
                g, i = db_scores(model, fingers)
                G.append(g); I.append(i)
                per_db.setdefault(d, ([], []))
                per_db[d][0].append(g); per_db[d][1].append(i)
        run = {"seed": seed, "pooled_real": metrics(np.concatenate(G), np.concatenate(I)),
               "per_db": {d: metrics(np.concatenate(v[0]), np.concatenate(v[1])) for d, v in per_db.items()}}
        runs.append(run)
        print(f"fine-tuned seed {seed}: pooled EER {run['pooled_real']['eer'] * 100:.2f}%", flush=True)
    res["fine_tuned"] = {"protocol": "2-fold finger-disjoint CV over pooled real databases", "epochs": args.epochs,
                         "runs": runs, "summary": {}}
    for k in ("eer", "fnmr_at_fmr_1e-2", "fnmr_at_fmr_1e-3"):
        v = [r["pooled_real"][k] for r in runs]
        res["fine_tuned"]["summary"][k] = {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0}
    json.dump(res, open(out / "fvc.json", "w"), indent=1)
    print(f"wrote {out}/fvc.json")


if __name__ == "__main__":
    main()
