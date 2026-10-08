"""Two follow-up attacks on ROP, on the SOCOFing test subjects (runs on a laptop in a few minutes).

    python attacks_v3.py --data ../socofing_raw/SOCOFing --weights models/encoder.pt --out results/attacks_v3.json

(c) Known-plaintext attack under PER-USER keys.
    In SecureFinger every user has their own key, so an attacker can only collect leaked
    (embedding, template) pairs of the victim's own captures. For every test subject we draw a key,
    leak n of that subject's images, estimate Q by orthogonal Procrustes, and invert the subject's
    remaining templates. Two leak scopes: captures of ONE finger only, and captures of ALL ten fingers.
    We report how well the inverted templates match plaintext probes (inversion EER) separately for
    the leaked finger and for the victim's other fingers.

(d) Collection-level linkage under a SHARED key.
    Two services each hold templates of the same N fingers (different impressions), each service
    using one key for all of its records. An orthogonal map preserves all inner products inside a
    collection, so the two collections can be aligned without knowing the key:
      * seeded: the attacker knows m linked records and aligns the rest by Procrustes;
      * blind: no linked records; candidate links are found from rotation-invariant signatures
        (each record's sorted similarities to the rest of its collection) and refined by Procrustes.
    The same attacks are run with per-record keys (SecureFinger's design) as a control.
    Reported: top-1 linkage accuracy and AUC of mated vs non-mated scores after alignment.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.append(str(_Path(__file__).resolve().parents[1]))  # repo root, for the securefinger and server packages
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from securefinger.cancelable import rop_matrix
from securefinger.encoder import FingerprintEncoder
from securefinger.keys import derive_key_vector, new_seed
from securefinger.metrics import eer, roc_auc
from train import embed_all, scan_socofing, split_subjects

DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def rand_Q():
    return rop_matrix(derive_key_vector(new_seed(), "attack", 1))


def procrustes(src, dst):
    """Orthogonal R minimising ||R src - dst|| for column-stacked vectors."""
    M = dst @ src.T
    try:
        U, _, Vt = np.linalg.svd(M)
    except np.linalg.LinAlgError:          # Accelerate's default SVD can fail on rank-deficient input
        try:
            u, _, vh = torch.linalg.svd(torch.from_numpy(M), driver=None)
            U, Vt = u.numpy(), vh.numpy()
        except Exception:
            M = M + 1e-12 * np.random.default_rng(0).standard_normal(M.shape)  # break exact degeneracy
            U, _, Vt = np.linalg.svd(M)
    return U @ Vt


def unit_rows(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


# ----------------------------------------------------------------------------- (c)
def per_user_attack(E, items, ns_one=(2, 4, 8), ns_all=(4, 8, 16, 32, 64), seed=0):
    rng = random.Random(seed)
    by_subj = {}
    for i, (_, s, fid, lvl) in enumerate(items):
        by_subj.setdefault(s, []).append(i)
    real_of = {fid: i for i, (_, _, fid, lvl) in enumerate(items) if lvl == "real"}
    all_real = list(real_of.items())
    out = {}
    for scope, ns in (("one_finger", ns_one), ("all_fingers", ns_all)):
        for n in ns:
            # for scope=all_fingers, "same_finger" means: the victim's other images (any finger)
            res = {"same_finger": ([], []), "other_fingers": ([], []), "cos_same": [], "cos_other": []}
            used = 0
            for s, idx in by_subj.items():
                fingers = {}
                for i in idx:
                    fingers.setdefault(items[i][2], []).append(i)
                if scope == "one_finger":
                    fid = rng.choice([f for f, v in fingers.items() if len(v) >= n + 2] or [None])
                    if fid is None:
                        continue
                    pool = fingers[fid][:]
                else:
                    fid = None
                    pool = idx[:]
                if len(pool) < n + 2:
                    continue
                rng.shuffle(pool)
                leak = pool[:n]
                leak_set = set(leak)
                Q = rand_Q()
                Qh = procrustes(E[leak].T, (Q @ E[leak].T))          # attacker's estimate from leaked pairs
                targets = [i for i in idx if i not in leak_set]
                for t in targets:
                    rec = Qh.T @ (Q @ E[t])
                    rec /= np.linalg.norm(rec)
                    tf = items[t][2]
                    key = ("same_finger" if tf == fid else "other_fingers") if scope == "one_finger" else "same_finger"
                    res["cos_same" if key == "same_finger" else "cos_other"].append(float(rec @ E[t]))
                    if tf not in real_of or real_of[tf] == t:
                        continue
                    g = float(rec @ E[real_of[tf]])
                    other = rng.choice(all_real)
                    while other[0] == tf:
                        other = rng.choice(all_real)
                    res[key][0].append(g)
                    res[key][1].append(float(rec @ E[other[1]]))
                used += 1
            row = {"victims": used}
            for key in ("same_finger", "other_fingers"):
                g, i = map(np.array, res[key])
                if len(g) > 10:
                    row[f"inversion_eer_{key}"] = float(eer(g, i)[0])
                    row[f"pairs_{key}"] = int(len(g))
            for key in ("cos_same", "cos_other"):
                if res[key]:
                    row[f"mean_{key}"] = float(np.mean(res[key]))
            out[f"{scope}/n={n}"] = row
            print(f"  (c) {scope:11s} n={n:3d} " + " ".join(
                f"{k}={v * 100:.2f}%" if "eer" in k else f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in row.items()), flush=True)
    return out


# ----------------------------------------------------------------------------- (d)
def linkage_scores(A, B):
    S = A @ B.T
    n = len(A)
    top1 = float(np.mean(S.argmax(1) == np.arange(n)))
    mated = np.diag(S)
    non = S[~np.eye(n, dtype=bool)]
    return top1, float(roc_auc(mated, non))


def seeded_alignment(TA, TB, m, rng):
    n = len(TA)
    known = rng.sample(range(n), m)
    R = procrustes(TB[known].T, TA[known].T)                        # maps service B space onto service A space
    rest = [i for i in range(n) if i not in set(known)]
    return linkage_scores(TA[rest], (R @ TB[rest].T).T)


def blind_alignment(TA, TB, k=96, iters=5):
    """No known links: rotation-invariant signatures give candidate links, Procrustes refines."""
    def sig(T):
        G = T @ T.T
        np.fill_diagonal(G, -np.inf)
        return np.sort(G, axis=1)[:, ::-1][:, :200]                  # top-200 similarities of each record
    SA, SB = sig(TA), sig(TB)
    D = -((SA[:, None, :] - SB[None, :, :]) ** 2).sum(-1)              # higher = more similar signature
    best = D.argmax(1)
    conf = D.max(1)
    seeds = np.argsort(-conf)[:k]
    pairs = [(a, best[a]) for a in seeds]
    for _ in range(iters):
        a_idx = [a for a, _ in pairs]; b_idx = [b for _, b in pairs]
        R = procrustes(TB[b_idx].T, TA[a_idx].T)
        S = TA @ (R @ TB.T)
        best = S.argmax(1); conf = S.max(1)
        seeds = np.argsort(-conf)[: max(k, len(TA) // 2)]
        pairs = [(a, best[a]) for a in seeds]
    return linkage_scores(TA, (R @ TB.T).T)


def collection_attack(E, items, seed=0, ms=(8, 16, 32, 64, 128)):
    rng = random.Random(seed)
    real_of, alt_of = {}, {}
    for i, (_, _, fid, lvl) in enumerate(items):
        (real_of.__setitem__(fid, i) if lvl == "real" else alt_of.setdefault(fid, []).append(i))
    fids = [f for f in real_of if f in alt_of]
    A = E[[real_of[f] for f in fids]]                                 # service A: real impressions
    B = E[[rng.choice(alt_of[f]) for f in fids]]                      # service B: another impression
    out = {"records": len(fids), "plaintext_upper_bound": dict(zip(("top1", "auc"), linkage_scores(A, B)))}
    QA, QB = rand_Q(), rand_Q()
    TA, TB = (QA @ A.T).T, (QB @ B.T).T                               # shared key per service
    out["shared_key/no_alignment"] = dict(zip(("top1", "auc"), linkage_scores(TA, TB)))
    ms = [m for m in ms if m < len(fids) - 10]
    for m in ms:
        out[f"shared_key/seeded_m={m}"] = dict(zip(("top1", "auc"), seeded_alignment(TA, TB, m, rng)))
    out["shared_key/blind"] = dict(zip(("top1", "auc"), blind_alignment(TA, TB)))
    # control: SecureFinger's per-record keys
    PA = np.stack([rand_Q() @ a for a in A]); PB = np.stack([rand_Q() @ b for b in B])
    out["per_user_keys/no_alignment"] = dict(zip(("top1", "auc"), linkage_scores(PA, PB)))
    for m in ms:
        out[f"per_user_keys/seeded_m={m}"] = dict(zip(("top1", "auc"), seeded_alignment(PA, PB, m, rng)))
    out["per_user_keys/blind"] = dict(zip(("top1", "auc"), blind_alignment(PA, PB)))
    for k, v in out.items():
        if isinstance(v, dict):
            print(f"  (d) {k:28s} top-1 linkage {v['top1'] * 100:6.2f}%   AUC {v['auc']:.3f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", default="models/encoder.pt")
    ap.add_argument("--out", default="results/attacks_v3.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--parts", nargs="+", default=["c", "d"], choices=["c", "d"])
    a = ap.parse_args()
    np.random.seed(a.seed)
    te = split_subjects(scan_socofing(Path(a.data)))["test"]
    print(f"test subjects: {len({s for _, s, _, _ in te})}, images: {len(te)}, device {DEV}", flush=True)
    model = FingerprintEncoder().to(DEV)
    model.load_state_dict(torch.load(a.weights, map_location=DEV)["encoder"])
    E = unit_rows(embed_all(model, te, DEV).astype(np.float64))
    res = {"weights": a.weights, "test_images": len(te)}
    if "c" in a.parts:
        print("== (c) known-plaintext attack with per-user keys", flush=True)
        res["per_user_known_plaintext"] = per_user_attack(E, te, seed=a.seed)
    if "d" in a.parts:
        print("== (d) collection-level linkage", flush=True)
        res["collection_linkage"] = collection_attack(E, te, seed=a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
