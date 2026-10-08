"""Train the ResNet18 + ArcFace encoder on SOCOFing and report verification accuracy.

Usage (Colab GPU recommended):
    python train.py --data /path/to/SOCOFing --epochs 30 --out models/

SOCOFing layout expected:
    SOCOFing/Real/1__M_Left_index_finger.BMP
    SOCOFing/Altered/Altered-Easy/1__M_Left_index_finger_CR.BMP   (also -Medium, -Hard)

Protocol:
  * subject-disjoint split 70/15/15 (420/90/90 subjects), identity = subject+hand+finger
  * genuine pairs: real impression vs each altered version of the same finger
  * impostor pairs: real impression vs altered image of a different finger (same count)
  * threshold chosen at the VALIDATION EER and then applied to the TEST set
Outputs: models/encoder.pt (weights + threshold), models/encoder.onnx, models/metrics.json
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.append(str(_Path(__file__).resolve().parents[1]))  # repo root, for the securefinger and server packages
import argparse
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from securefinger.encoder import ArcFaceHead, FingerprintEncoder, export_onnx
from securefinger.metrics import eer, rates_at
from securefinger.preprocess import load_and_preprocess

NAME_RE = re.compile(r"^(\d+)__([MF])_(Left|Right)_(\w+?)_finger(?:_(CR|Obl|Zcut))?$", re.I)


def scan_socofing(root: Path):
    """Return list of (path, subject, finger_id, level) where level in real/easy/medium/hard."""
    items = []
    for p in root.rglob("*"):
        if p.suffix.lower() not in {".bmp", ".png", ".jpg", ".tif"}:
            continue
        m = NAME_RE.match(p.stem)
        if not m:
            continue
        subj, _, hand, finger, _alt = m.groups()
        parts = str(p).lower()
        level = ("easy" if "easy" in parts else "medium" if "medium" in parts
                 else "hard" if "hard" in parts else "real")
        items.append((str(p), int(subj), f"{subj}_{hand}_{finger}".lower(), level))
    return items


def split_subjects(items, seed=42, frac=(0.70, 0.15, 0.15)):
    subjects = sorted({s for _, s, _, _ in items})
    random.Random(seed).shuffle(subjects)
    n = len(subjects)
    a, b = int(n * frac[0]), int(n * (frac[0] + frac[1]))
    groups = {"train": set(subjects[:a]), "val": set(subjects[a:b]), "test": set(subjects[b:])}
    return {k: [it for it in items if it[1] in v] for k, v in groups.items()}


_CACHE: dict[str, np.ndarray] = {}


def preload(items):
    """Preprocess each image once (float16) and keep it in memory for all epochs."""
    todo = [p for p, *_ in items if p not in _CACHE]
    for n, p in enumerate(todo, 1):
        _CACHE[p] = load_and_preprocess(p).astype(np.float16)
        if n % 5000 == 0:
            print(f"  preprocessed {n}/{len(todo)}")


class FingerDS(Dataset):
    def __init__(self, items, label_map=None, augment=False, shift=4):
        preload(items)
        self.shift = int(shift)
        self.x = np.stack([_CACHE[p] for p, *_ in items])
        self.y = np.array([label_map[fid] if label_map else 0 for _, _, fid, _ in items])
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.x[i].astype(np.float32)
        if self.augment and self.shift > 0:  # light augmentation: random shift
            dx, dy = np.random.randint(-self.shift, self.shift + 1, 2)
            x = np.roll(x, (dy, dx), axis=(1, 2))
        return torch.from_numpy(x), int(self.y[i])


def embed_all(model, items, device, bs=256):
    model.eval()
    dl = DataLoader(FingerDS(items), batch_size=bs, num_workers=0)
    out = []
    with torch.no_grad():
        for x, _ in dl:
            out.append(model(x.to(device)).cpu().numpy())
    return np.concatenate(out)


def make_pairs(items, seed=0):
    """Genuine: real vs altered of same finger. Impostor: real vs altered of another finger."""
    rng = random.Random(seed)
    real = {fid: i for i, (_, _, fid, lvl) in enumerate(items) if lvl == "real"}
    altered = [(i, fid, lvl) for i, (_, _, fid, lvl) in enumerate(items) if lvl != "real"]
    gen = [(real[fid], i, lvl) for i, fid, lvl in altered if fid in real]
    fids = list(real)
    imp = []
    for _, i, lvl in gen:
        fid_i = items[i][2]
        other = rng.choice(fids)
        while other == fid_i:
            other = rng.choice(fids)
        imp.append((real[other], i, lvl))
    return gen, imp


def scores(emb, pairs):
    return np.array([float(emb[a] @ emb[b]) for a, b, _ in pairs])


def evaluate(model, split_items, device):
    emb = embed_all(model, split_items, device)
    gen, imp = make_pairs(split_items)
    return scores(emb, gen), scores(emb, imp), gen, imp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="models")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
              else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    items = scan_socofing(Path(args.data))
    if not items:
        raise SystemExit(f"No SOCOFing images found under {args.data}")
    splits = split_subjects(items)
    fids = sorted({fid for _, _, fid, _ in splits["train"]})
    label_map = {f: i for i, f in enumerate(fids)}
    print(f"images={len(items)} train={len(splits['train'])} val={len(splits['val'])} "
          f"test={len(splits['test'])} train_identities={len(fids)} device={device}")

    model = FingerprintEncoder().to(device)
    head = ArcFaceHead(256, len(fids)).to(device)
    params = list(model.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    workers = 2 if device == "cuda" else 0   # macOS/MPS: worker processes would copy the whole dataset
    dl = DataLoader(FingerDS(splits["train"], label_map, augment=True), batch_size=args.bs,
                    shuffle=True, num_workers=workers, drop_last=True)

    best_eer, best_state = 1.0, None
    for ep in range(1, args.epochs + 1):
        model.train(); head.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            loss = head(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(y); n += len(y)
        sched.step()
        g, i, _, _ = evaluate(model, splits["val"], device)
        val_eer, _ = eer(g, i)
        print(f"epoch {ep:3d} loss {tot / n:.4f} val_EER {val_eer * 100:.2f}% ({time.time() - t0:.0f}s)")
        if val_eer < best_eer:
            best_eer, best_state = val_eer, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    g_val, i_val, _, _ = evaluate(model, splits["val"], device)
    val_eer, tau = eer(g_val, i_val)
    g, i, gen, imp = evaluate(model, splits["test"], device)
    test_eer, _ = eer(g, i)
    far, frr = rates_at(g, i, tau)

    per_level = {}
    for lvl in ("easy", "medium", "hard"):
        gl = np.array([s for s, p in zip(g, gen) if p[2] == lvl])
        il = np.array([s for s, p in zip(i, imp) if p[2] == lvl])
        if len(gl):
            e, _ = eer(gl, il)
            per_level[lvl] = {"eer": e, "frr_at_tau": float((gl < tau).mean()), "pairs": int(len(gl))}

    metrics = {
        "dataset": "SOCOFing", "split": "subject-disjoint 70/15/15",
        "val_eer": val_eer, "threshold_from_val": tau,
        "test_eer": test_eer, "test_far_at_tau": far, "test_frr_at_tau": frr,
        "test_genuine_pairs": int(len(g)), "test_impostor_pairs": int(len(i)),
        "per_level": per_level, "epochs": args.epochs,
    }
    print(json.dumps(metrics, indent=2))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    np.savez(out / "test_scores.npz", genuine=g, impostor=i)
    torch.save({"encoder": model.state_dict(), "threshold": tau}, out / "encoder.pt")
    try:
        export_onnx(model, str(out / "encoder.onnx"))
    except Exception as e:  # ONNX is optional (needs onnx + onnxscript)
        print(f"ONNX export skipped: {e}")
    print(f"saved to {out}/")


if __name__ == "__main__":
    main()
