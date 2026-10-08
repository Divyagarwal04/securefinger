"""Evaluate a trained encoder: accuracy (plaintext / FiLM / ROP), linkability, and CKKS fidelity.

Usage:
    python evaluate.py --data /path/to/SOCOFing --weights models/encoder.pt --out results/

Reported settings
  * stolen-key: every template uses the SAME key (the impostor holds the victim's key).
    This isolates biometric discrimination and is the headline accuracy number.
  * legit-key:  each finger has its own key. Impostor scores collapse mostly because of key
    decorrelation, so this number flatters the system; reported for completeness only.
  * thresholds are chosen on the validation split and applied to the test split.
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
import torch

from securefinger import he
from securefinger.cancelable import make_template
from securefinger.keys import derive_key_vector, new_seed
from securefinger.metrics import eer, rates_at, roc_auc
from train import embed_all, make_pairs, scan_socofing, split_subjects
from securefinger.encoder import FingerprintEncoder


def templates(emb, keys, method):
    if method == "plain":
        return emb
    return np.stack([make_template(e, k, method) for e, k in zip(emb, keys)])


def pair_scores(t, pairs):
    return np.array([float(t[a] @ t[b]) for a, b, _ in pairs])


def accuracy(emb_val, items_val, emb_test, items_test, method, key_mode):
    res = {}
    for name, emb, items in (("val", emb_val, items_val), ("test", emb_test, items_test)):
        gen, imp = make_pairs(items)
        if key_mode == "stolen":
            k = derive_key_vector(new_seed(), "victim", 1)
            keys = [k] * len(items)
        else:  # one key per finger identity
            per = {fid: derive_key_vector(new_seed(), fid, 1) for fid in {it[2] for it in items}}
            keys = [per[it[2]] for it in items]
        t = templates(emb, keys, method)
        res[name] = (pair_scores(t, gen), pair_scores(t, imp))
    _, tau = eer(*res["val"])
    test_eer, _ = eer(*res["test"])
    far, frr = rates_at(*res["test"], tau)
    return {"test_eer": test_eer, "tau_val": tau, "test_far": far, "test_frr": frr}


def linkability(emb, items, method, n=1000, seed=0):
    """Mated = same finger under two different keys; non-mated = different fingers, different keys."""
    rng = random.Random(seed)
    real = [i for i, it in enumerate(items) if it[3] == "real"]
    mated_cos, non_cos, mated_abs, non_abs = [], [], [], []
    for _ in range(n):
        a, b = rng.sample(real, 2)
        k1, k2 = derive_key_vector(new_seed(), "u", 1), derive_key_vector(new_seed(), "u", 1)
        ta = make_template(emb[a], k1, method) if method != "plain" else emb[a]
        ta2 = make_template(emb[a], k2, method) if method != "plain" else emb[a]
        tb = make_template(emb[b], k2, method) if method != "plain" else emb[b]
        mated_cos.append(ta @ ta2); non_cos.append(ta @ tb)
        mated_abs.append(np.abs(ta) @ np.abs(ta2)); non_abs.append(np.abs(ta) @ np.abs(tb))
    return {"auc_cosine": roc_auc(mated_cos, non_cos), "auc_absolute": roc_auc(mated_abs, non_abs)}


def he_fidelity(emb, items, n=200):
    ctx = he.new_secret_context()
    pub = he.load_public(he.public_context_bytes(ctx))
    gen, imp = make_pairs(items)
    pairs = (gen[: n // 2] + imp[: n // 2])
    errs, times = [], []
    for a, b, _ in pairs:
        ea, eb = he.encrypt(ctx, emb[a]), he.encrypt(ctx, emb[b])
        t = time.perf_counter()
        enc = he.server_dot(pub, ea, eb)
        times.append((time.perf_counter() - t) * 1e3)
        errs.append(abs(he.decrypt_scalar(ctx, enc) - float(emb[a] @ emb[b])))
    return {"pairs": len(pairs), "max_abs_error": float(np.max(errs)),
            "mean_abs_error": float(np.mean(errs)), "server_dot_median_ms": float(np.median(times))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", default="models/encoder.pt")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")

    splits = split_subjects(scan_socofing(Path(args.data)))
    model = FingerprintEncoder().to(device)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state["encoder"])
    emb_val = embed_all(model, splits["val"], device).astype(np.float64)
    emb_test = embed_all(model, splits["test"], device).astype(np.float64)

    out = {"accuracy_stolen_key": {}, "accuracy_legit_key": {}, "linkability": {}}
    for m in ("plain", "film", "rop"):
        out["accuracy_stolen_key"][m] = accuracy(emb_val, splits["val"], emb_test, splits["test"], m, "stolen")
        print(m, "stolen-key", out["accuracy_stolen_key"][m])
    for m in ("film", "rop"):
        out["accuracy_legit_key"][m] = accuracy(emb_val, splits["val"], emb_test, splits["test"], m, "legit")
        out["linkability"][m] = linkability(emb_test, splits["test"], m)
        print(m, "legit-key", out["accuracy_legit_key"][m], "linkability", out["linkability"][m])
    out["ckks"] = he_fidelity(emb_test, splits["test"])
    print("ckks", out["ckks"])

    Path(args.out).mkdir(parents=True, exist_ok=True)
    Path(args.out, "eval.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}/eval.json")


if __name__ == "__main__":
    main()
