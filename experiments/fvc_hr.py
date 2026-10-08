"""Can resolution and geometric augmentation close the SOCOFing -> FVC gap?

    python fvc_hr.py --data /content/socofing_raw/SOCOFing --fvc /content/fvc --out results_fvc_hr \
                     --sizes 96 160 --epochs 15

For each input size R, an encoder is trained on SOCOFing with strong augmentation that imitates real
multi-session variation (rotation +-25 deg, scale 0.85-1.15, shift +-10%, random occlusion of up to
30% of the image for partial contact). It is then evaluated
  * on the SOCOFing test split (standard protocol, to check nothing is lost),
  * zero-shot on the pooled real FVC / Neurotechnology databases,
  * after fine-tuning with 2-fold finger-disjoint cross-validation (3 seeds),
with exactly the metrics of fvc_eval.py, so the numbers are directly comparable.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.append(str(_Path(__file__).resolve().parents[1]))  # repo root, for the securefinger and server packages
import argparse
import itertools
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from fvc_eval import SYNTHETIC, metrics, scan
from securefinger.encoder import ArcFaceHead, FingerprintEncoder
from securefinger.metrics import eer
from securefinger.preprocess import _roi_crop
from train import make_pairs, scan_socofing, split_subjects

DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
_CACHE = {}


def prep(path, size):
    key = (str(path), size)
    if key not in _CACHE:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        img = _CLAHE.apply(_roi_crop(img))
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA if img.shape[0] > size else cv2.INTER_CUBIC)
        x = img.astype(np.float32)
        _CACHE[key] = ((x - x.mean()) / (x.std() + 1e-6)).astype(np.float16)
    return _CACHE[key]


def augment(x, size, strong=True):
    a = np.random.uniform(-25, 25) if strong else 0.0
    s = np.random.uniform(0.85, 1.15) if strong else 1.0
    M = cv2.getRotationMatrix2D((size / 2, size / 2), a, s)
    M[:, 2] += np.random.uniform(-0.1, 0.1, 2) * size
    x = cv2.warpAffine(x, M, (size, size), borderValue=0.0)
    if strong and np.random.rand() < 0.5:                      # partial contact
        h, w = (np.random.uniform(0.1, 0.3, 2) * size).astype(int)
        y0, x0 = np.random.randint(0, size - h), np.random.randint(0, size - w)
        x[y0:y0 + h, x0:x0 + w] = 0.0
    return x


class DS(Dataset):
    def __init__(self, items, size):
        self.items, self.size = items, size

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        p, y = self.items[k]
        x = augment(prep(p, self.size).astype(np.float32), self.size)
        return torch.from_numpy(x[None]), y


@torch.no_grad()
def embed(model, paths, size, bs=128):
    model.eval(); out = []
    for i in range(0, len(paths), bs):
        x = torch.from_numpy(np.stack([prep(p, size).astype(np.float32)[None] for p in paths[i:i + bs]])).to(DEV)
        out.append(model(x).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def train(items, n_classes, size, epochs, seed, init=None, lr=1e-3, bs=128, log=""):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = FingerprintEncoder().to(DEV)
    if init is not None:
        model.load_state_dict(init)
    head = ArcFaceHead(256, n_classes).to(DEV)
    opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    dl = DataLoader(DS(items, size), batch_size=bs, shuffle=True, drop_last=True,
                    num_workers=2 if DEV == "cuda" else 0)
    for ep in range(epochs):
        model.train(); head.train(); t0, tot, n = time.time(), 0.0, 0
        for x, y in dl:
            loss = head(model(x.to(DEV)), y.to(DEV))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(y); n += len(y)
        sched.step()
        if log:
            print(f"  {log} ep {ep + 1:2d} loss {tot / n:.3f} ({time.time() - t0:.0f}s)", flush=True)
    return model


def socofing_test(model, test_items, size):
    e = embed(model, [p for p, *_ in test_items], size)
    gen, imp = make_pairs(test_items)
    g = np.array([e[a] @ e[b] for a, b, _ in gen]); i = np.array([e[a] @ e[b] for a, b, _ in imp])
    return eer(g, i)[0]


def db_scores(model, fingers, size, max_imp=20000, seed=0):
    keys = sorted(fingers)
    paths = [p for k in keys for p in fingers[k]]
    owner = [k for k in keys for _ in fingers[k]]
    E = embed(model, paths, size)
    idx = {}
    for n, k in enumerate(owner):
        idx.setdefault(k, []).append(n)
    gen = [E[a] @ E[b] for k in keys for a, b in itertools.combinations(idx[k], 2)]
    pairs = [(a, b) for a, b in itertools.combinations(range(len(paths)), 2) if owner[a] != owner[b]]
    if len(pairs) > max_imp:
        pairs = random.Random(seed).sample(pairs, max_imp)
    return np.array(gen), np.array([E[a] @ E[b] for a, b in pairs])


def fvc_pooled(model, dbs, real, size):
    G, I, per = [], [], {}
    for d in real:
        g, i = db_scores(model, dbs[d], size)
        per[d] = metrics(g, i); G.append(g); I.append(i)
    return metrics(np.concatenate(G), np.concatenate(I)), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--fvc", required=True)
    ap.add_argument("--out", default="results_fvc_hr")
    ap.add_argument("--sizes", type=int, nargs="+", default=[96, 160])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--ft-epochs", type=int, default=40)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    sp = split_subjects(scan_socofing(Path(args.data)))
    fids = sorted({fid for _, _, fid, _ in sp["train"]}); lm = {f: k for k, f in enumerate(fids)}
    train_items = [(p, lm[fid]) for p, _, fid, _ in sp["train"]]
    print("FVC databases:", flush=True)
    dbs = scan(Path(args.fvc)); real = [d for d in dbs if d not in SYNTHETIC]
    allf = [(d, k) for d in real for k in dbs[d]]
    res = {"device": DEV, "epochs": args.epochs, "ft_epochs": args.ft_epochs, "results": {}}

    for size in args.sizes:
        print(f"== size {size}: training on SOCOFing with strong augmentation", flush=True)
        t0 = time.time()
        model = train(train_items, len(fids), size, args.epochs, seed=0, log=f"R{size}")
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        torch.save({"encoder": state, "size": size}, out / f"encoder_aug_{size}.pt")
        r = {"train_min": (time.time() - t0) / 60, "socofing_test_eer": socofing_test(model, sp["test"], size)}
        r["fvc_zero_shot"], r["fvc_zero_shot_per_db"] = fvc_pooled(model, dbs, real, size)
        print(f"   SOCOFing test EER {r['socofing_test_eer'] * 100:.3f}%  FVC zero-shot EER {r['fvc_zero_shot']['eer'] * 100:.2f}%", flush=True)
        runs = []
        for seed in args.seeds:
            order = allf[:]; random.Random(seed).shuffle(order)
            folds = [order[0::2], order[1::2]]; G, I = [], []
            for f in (0, 1):
                keys = [f"{d}:{k}" for d, k in folds[f]]
                items = [(p, n) for n, (d, k) in enumerate(folds[f]) for p in dbs[d][k]]
                ft = train(items, len(keys), size, args.ft_epochs, seed * 10 + f, init=state, lr=3e-4, bs=64)
                te = {}
                for d, k in folds[1 - f]:
                    te.setdefault(d, {})[k] = dbs[d][k]
                for d, fingers in te.items():
                    if len(fingers) >= 2:
                        g, i = db_scores(ft, fingers, size); G.append(g); I.append(i)
            m = metrics(np.concatenate(G), np.concatenate(I)); runs.append(m)
            print(f"   fine-tuned seed {seed}: EER {m['eer'] * 100:.2f}%  FMR100 {m['fnmr_at_fmr_1e-2'] * 100:.2f}%", flush=True)
        r["fvc_fine_tuned_runs"] = runs
        r["fvc_fine_tuned"] = {k: {"mean": float(np.mean([x[k] for x in runs])),
                                   "std": float(np.std([x[k] for x in runs], ddof=1)) if len(runs) > 1 else 0.0}
                               for k in ("eer", "fnmr_at_fmr_1e-2", "fnmr_at_fmr_1e-3")}
        res["results"][str(size)] = r
        json.dump(res, open(out / "fvc_hr.json", "w"), indent=1)
    print(f"wrote {out}/fvc_hr.json")


if __name__ == "__main__":
    main()
