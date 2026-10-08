"""Extended evaluation for the final report (run on a GPU, e.g. Colab T4).

    python experiments.py --data /path/to/SOCOFing --out results_ext --seeds 0 1 2 --epochs 15

Experiments
  E1  Harder protocols: random impostors vs "hard" impostors (same hand and finger type,
      different subject), reported per alteration level, with TAR at fixed FAR and DET data.
  E2  Unseen-severity training: train on real + easy alterations only, test on medium/hard.
  E3  Seed variance: E1/E2 repeated for several training seeds -> mean +/- std.
  E4  Unlinkability: ISO/IEC 24745 system linkability D_sys (Gomez-Barrero et al.) with a
      permutation noise floor, for cosine, magnitude and a learned (MLP) linkage attacker,
      for FiLM and ROP, using both same-image and different-impression mated pairs.
  E5  Known-plaintext attack on ROP: recover Q by orthogonal Procrustes from n leaked
      (embedding, template) pairs; report matrix error and inversion EER versus n.
Everything is written to <out>/experiments.json and <out>/scores_*.npz.
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
from torch.utils.data import DataLoader

from securefinger.cancelable import make_template, rop_matrix
from securefinger.encoder import ArcFaceHead, FingerprintEncoder
from securefinger.keys import derive_key_vector, new_seed
from securefinger.metrics import eer, rates_at, roc_auc
from train import FingerDS, embed_all, make_pairs, scan_socofing, split_subjects

DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


# ----------------------------------------------------------------------------- helpers
def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def finger_type(fid):  # "12_left_index" -> "left_index"
    return fid.split("_", 1)[1]


def hard_pairs(items, seed=0):
    """Genuine: real vs altered of the same finger. Impostor: the same altered image vs the real
    impression of a DIFFERENT subject's finger of the same hand and finger type."""
    rng = random.Random(seed)
    real = {fid: i for i, (_, _, fid, lvl) in enumerate(items) if lvl == "real"}
    by_type = {}
    for fid in real:
        by_type.setdefault(finger_type(fid), []).append(fid)
    gen, imp = [], []
    for i, (_, _, fid, lvl) in enumerate(items):
        if lvl == "real" or fid not in real:
            continue
        gen.append((real[fid], i, lvl))
        cands = by_type[finger_type(fid)]
        other = rng.choice(cands)
        while other == fid:
            other = rng.choice(cands)
        imp.append((real[other], i, lvl))
    return gen, imp


def scores(emb, pairs):
    a = np.array([p[0] for p in pairs]); b = np.array([p[1] for p in pairs])
    return np.einsum("ij,ij->i", emb[a], emb[b])


def tar_at_far(g, i, far):
    thr = np.quantile(i, 1 - far) if len(i) else 1.0
    return float((g > thr).mean())


def det_curve(g, i, n=200):
    thr = np.quantile(np.concatenate([g, i]), np.linspace(0, 1, n))
    return [[float((i >= t).mean()), float((g < t).mean())] for t in thr]


def protocol_report(emb_val, val_items, emb_test, test_items):
    """Threshold from validation (random impostors); test on random and hard impostors."""
    gv, iv = make_pairs(val_items)
    _, tau = eer(scores(emb_val, gv), scores(emb_val, iv))
    rep = {"tau_val": tau}
    for name, fn in (("random_impostors", make_pairs), ("hard_impostors", hard_pairs)):
        gen, imp = fn(test_items)
        g, i = scores(emb_test, gen), scores(emb_test, imp)
        e, _ = eer(g, i)
        far, frr = rates_at(g, i, tau)
        r = {"pairs": len(g), "eer": e, "far_at_tau": far, "frr_at_tau": frr,
             "tar_at_far_1e-3": tar_at_far(g, i, 1e-3), "tar_at_far_1e-4": tar_at_far(g, i, 1e-4),
             "genuine_mean": float(g.mean()), "impostor_mean": float(i.mean()),
             "impostor_max": float(i.max()), "genuine_min": float(g.min()), "per_level": {}}
        for lvl in ("easy", "medium", "hard"):
            gl = np.array([s for s, p in zip(g, gen) if p[2] == lvl])
            il = np.array([s for s, p in zip(i, imp) if p[2] == lvl])
            if len(gl):
                el, _ = eer(gl, il)
                r["per_level"][lvl] = {"pairs": int(len(gl)), "eer": el, "frr_at_tau": float((gl < tau).mean()),
                                       "far_at_tau": float((il >= tau).mean())}
        rep[name] = r
        rep[name + "_scores"] = (g, i)
    return rep


# ----------------------------------------------------------------------------- training
def train_encoder(train_items, val_items, epochs, seed, bs=128, lr=1e-3, m=0.5, s=64.0, wd=5e-4, shift=4,
                  select_best=True):
    set_seed(seed)
    fids = sorted({fid for _, _, fid, _ in train_items})
    lm = {f: k for k, f in enumerate(fids)}
    model = FingerprintEncoder().to(DEV)
    head = ArcFaceHead(256, len(fids), s=s, m=m).to(DEV)
    opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    g = torch.Generator(); g.manual_seed(seed)
    dl = DataLoader(FingerDS(train_items, lm, augment=True, shift=shift), batch_size=bs, shuffle=True, drop_last=True,
                    num_workers=2 if DEV == "cuda" else 0, generator=g)
    best, best_state, curve = 1.0, None, []
    for ep in range(1, epochs + 1):
        model.train(); head.train(); t0, tot, n = time.time(), 0.0, 0
        for x, y in dl:
            x, y = x.to(DEV), y.to(DEV)
            loss = head(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(y); n += len(y)
        sched.step()
        if not select_best:
            print(f"  seed {seed} ep {ep:2d} loss {tot / n:.4f} ({time.time() - t0:.0f}s)", flush=True)
            curve.append({"epoch": ep, "loss": tot / n, "sec": time.time() - t0})
            continue
        ev = embed_all(model, val_items, DEV)
        gv, iv = make_pairs(val_items)
        v, _ = eer(scores(ev, gv), scores(ev, iv))
        curve.append({"epoch": ep, "loss": tot / n, "val_eer": v, "sec": time.time() - t0})
        print(f"  seed {seed} ep {ep:2d} loss {tot / n:.4f} val_EER {v * 100:.3f}% ({time.time() - t0:.0f}s)", flush=True)
        if v <= best:
            best, best_state = v, {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
    if select_best:
        model.load_state_dict(best_state)
    return model, curve


# ----------------------------------------------------------------------------- unlinkability
def dsys(mated, nonmated, bins=100):
    """Gomez-Barrero et al. (2018) global linkability D_sys with omega = 1 (histogram densities)."""
    lo, hi = min(mated.min(), nonmated.min()), max(mated.max(), nonmated.max())
    edges = np.linspace(lo, hi + 1e-12, bins + 1)
    pm, _ = np.histogram(mated, edges); pn, _ = np.histogram(nonmated, edges)
    pm = pm / pm.sum(); pn = pn / pn.sum()
    lr = np.divide(pm, pn, out=np.full_like(pm, np.inf), where=pn > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = np.where(lr > 1, 2 * lr / (1 + lr) - 1, 0.0)
    d = np.where(np.isinf(lr), 1.0, d)
    return float((pm * d).sum())


def dsys_with_floor(m, nm, perms=50, seed=0):
    rng = np.random.default_rng(seed)
    allv = np.concatenate([m, nm]); k = len(m); fl = []
    for _ in range(perms):
        p = rng.permutation(allv); fl.append(dsys(p[:k], p[k:]))
    return dsys(m, nm), float(np.percentile(fl, 95))


def pair_features(a, b):
    aa, bb = np.abs(a), np.abs(b)
    return np.concatenate([a * b, aa * bb, np.abs(aa - bb), np.abs(np.sort(aa, 1) - np.sort(bb, 1))], 1)


def linkage_pairs(emb, items, method, n, mode, rng):
    """Mated: same finger under two independent keys (same image, or real vs altered impression).
    Non-mated: different fingers under two independent keys."""
    real = [i for i, it in enumerate(items) if it[3] == "real"]
    alt_of = {}
    for i, it in enumerate(items):
        if it[3] != "real":
            alt_of.setdefault(it[2], []).append(i)
    real = [i for i in real if items[i][2] in alt_of] if mode == "impression" else real
    A, B, y = [], [], []
    for lab in (1, 0):
        for _ in range(n):
            a = rng.choice(real)
            if lab:
                b = a if mode == "same_image" else rng.choice(alt_of[items[a][2]])
            else:
                b = rng.choice(real)
                while items[b][2] == items[a][2]:
                    b = rng.choice(real)
            k1 = derive_key_vector(new_seed(), "u", 1); k2 = derive_key_vector(new_seed(), "u", 1)
            A.append(make_template(emb[a], k1, method)); B.append(make_template(emb[b], k2, method)); y.append(lab)
    return np.array(A), np.array(B), np.array(y)


def unlinkability(emb_train, train_items, emb_test, test_items, n_train=6000, n_test=1000, seed=0):
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    rng = random.Random(seed); out = {}
    for method in ("film", "rop"):
        for mode in ("same_image", "impression"):
            A, B, y = linkage_pairs(emb_test, test_items, method, n_test, mode, rng)
            res = {}
            att = {"cosine": np.einsum("ij,ij->i", A, B),
                   "magnitude": np.einsum("ij,ij->i", np.abs(A), np.abs(B))}
            At, Bt, yt = linkage_pairs(emb_train, train_items, method, n_train // 2, mode, rng)
            sc = StandardScaler().fit(pair_features(At, Bt))
            clf = MLPClassifier(hidden_layer_sizes=(256, 64), max_iter=200, early_stopping=True, random_state=seed)
            clf.fit(sc.transform(pair_features(At, Bt)), yt)
            att["learned_mlp"] = clf.predict_proba(sc.transform(pair_features(A, B)))[:, 1]
            for name, s in att.items():
                m, nm = s[y == 1], s[y == 0]
                d, floor = dsys_with_floor(m, nm)
                res[name] = {"auc": roc_auc(m, nm), "dsys": d, "dsys_noise_floor_p95": floor}
            out[f"{method}/{mode}"] = res
            print("  linkability", method, mode, {k: round(v["auc"], 3) for k, v in res.items()}, flush=True)
    return out


# ----------------------------------------------------------------------------- Procrustes attack
def procrustes_attack(emb_leak, emb_test, test_items, ns=(16, 64, 128, 192, 256, 384, 512), seed=0):
    rng = np.random.default_rng(seed)
    k = derive_key_vector(new_seed(), "victim", 1)
    Q = rop_matrix(k)
    gen, imp = make_pairs(test_items)
    out = []
    for n in ns:
        idx = rng.choice(len(emb_leak), n, replace=False)
        E = emb_leak[idx].T; T = Q @ E                     # attacker's leaked pairs (columns)
        U, _, Vt = np.linalg.svd(T @ E.T)
        Qh = U @ Vt
        rel = float(np.linalg.norm(Qh - Q) / np.linalg.norm(Q))
        # attack: templates of enrolled fingers (under Q) are inverted with Qh and matched with plaintext probes
        tmpl = (Q @ emb_test.T).T
        rec = (Qh.T @ tmpl.T).T
        rec /= np.linalg.norm(rec, axis=1, keepdims=True)
        g = np.array([rec[a] @ emb_test[b] for a, b, _ in gen]); i = np.array([rec[a] @ emb_test[b] for a, b, _ in imp])
        e, _ = eer(g, i)
        out.append({"n": int(n), "rel_frobenius_error": rel, "inversion_eer": e})
        print(f"  procrustes n={n:4d} rel_err={rel:.4f} inversion_EER={e * 100:.2f}%", flush=True)
    return out


# ----------------------------------------------------------------------------- main
def summarise(runs, key_path):
    vals = []
    for r in runs:
        v = r
        for k in key_path:
            v = v[k]
        vals.append(v)
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, "values": vals}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results_ext")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--skip", nargs="*", default=[], help="any of: full easyonly link procrustes")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    res = {"device": DEV, "epochs": args.epochs, "seeds": args.seeds}
    if torch.cuda.is_available():
        res["gpu"] = torch.cuda.get_device_name(0)

    splits = split_subjects(scan_socofing(Path(args.data)))
    tr, va, te = splits["train"], splits["val"], splits["test"]
    print(f"device={DEV} train={len(tr)} val={len(va)} test={len(te)}", flush=True)
    configs = {"full": tr, "easyonly": [it for it in tr if it[3] in ("real", "easy")]}
    first_model = None
    for cname, train_items in configs.items():
        if cname in args.skip:
            continue
        runs = []
        for s in args.seeds:
            print(f"== config {cname} seed {s} ({len(train_items)} training images)", flush=True)
            model, curve = train_encoder(train_items, va, args.epochs, s)
            ev, et = embed_all(model, va, DEV), embed_all(model, te, DEV)
            rep = protocol_report(ev, va, et, te)
            for prot in ("random_impostors", "hard_impostors"):
                g, i = rep.pop(prot + "_scores")
                np.savez(out / f"scores_{cname}_seed{s}_{prot}.npz", genuine=g, impostor=i)
                if s == args.seeds[0]:
                    rep[prot]["det"] = det_curve(g, i)
            rep["curve"] = curve
            runs.append(rep)
            print("  ", cname, s, "EER random", f"{rep['random_impostors']['eer'] * 100:.3f}%",
                  "hard", f"{rep['hard_impostors']['eer'] * 100:.3f}%", flush=True)
            if first_model is None and cname == "full":
                first_model = model
                torch.save({"encoder": model.state_dict(), "threshold": rep["tau_val"]}, out / "encoder_seed0.pt")
        res[cname] = {"runs": runs, "summary": {}}
        for prot in ("random_impostors", "hard_impostors"):
            for m in ("eer", "far_at_tau", "frr_at_tau", "tar_at_far_1e-3", "tar_at_far_1e-4"):
                res[cname]["summary"][f"{prot}/{m}"] = summarise(runs, [prot, m])
            for lvl in ("easy", "medium", "hard"):
                res[cname]["summary"][f"{prot}/{lvl}/eer"] = summarise(runs, [prot, "per_level", lvl, "eer"])
                res[cname]["summary"][f"{prot}/{lvl}/frr_at_tau"] = summarise(runs, [prot, "per_level", lvl, "frr_at_tau"])
        json.dump(res, open(out / "experiments.json", "w"), indent=1)

    if first_model is not None:
        e_tr, e_te = embed_all(first_model, tr, DEV).astype(np.float64), embed_all(first_model, te, DEV).astype(np.float64)
        if "link" not in args.skip:
            print("== unlinkability", flush=True)
            res["unlinkability"] = unlinkability(e_tr, tr, e_te, te)
            json.dump(res, open(out / "experiments.json", "w"), indent=1)
        if "procrustes" not in args.skip:
            print("== procrustes known-plaintext attack", flush=True)
            res["procrustes"] = procrustes_attack(e_tr, e_te, te)
    json.dump(res, open(out / "experiments.json", "w"), indent=1)
    print(f"wrote {out}/experiments.json")


if __name__ == "__main__":
    main()
