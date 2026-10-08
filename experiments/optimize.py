"""Metaheuristic optimisation of the encoder (Orca Predation Algorithm, OPA) on the HARD protocol.

    python optimize.py --data /path/to/SOCOFing --out results_opt --budget 25 --epochs 5

Fitness (minimised): validation EER against hard impostors (same hand and finger type), on the
medium and hard alterations only, for an encoder trained on real + easy images. The medium and hard
severities are therefore never seen in training, which is the non-saturated protocol of the report.

Search space (each mapped from [0, 1]):
    ArcFace margin m      0.20 .. 0.70
    ArcFace scale s       16 .. 64
    learning rate         1e-4 .. 3e-3   (log scale)
    weight decay          1e-5 .. 3e-3   (log scale)
    shift augmentation    0 .. 8 pixels  (integer)
The embedding size stays 256 because the cancelable transform and CKKS packing are built for it.

OPA follows the update rules of the Orca Predation Algorithm (Jiang et al., Expert Systems with
Applications 188, 116026, 2022) as implemented in mealpy's OriginalOrcaPA (p1=0.5, p2=0.1, q=0.9, F=2). A random search with the same number of evaluations is the baseline.
Finally the default and the OPA-best settings are retrained with 3 seeds and full epochs and compared
on the TEST partition (never used during the search).
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.append(str(_Path(__file__).resolve().parents[1]))  # repo root, for the securefinger and server packages
import argparse
import json
import time
from pathlib import Path

import numpy as np

from experiments import DEV, embed_all, hard_pairs, protocol_report, scores, train_encoder
from securefinger.metrics import eer
from train import scan_socofing, split_subjects

DEFAULT = {"m": 0.5, "s": 64.0, "lr": 1e-3, "wd": 5e-4, "shift": 4}


def decode(x):
    x = np.clip(np.asarray(x, dtype=float), 0, 1)
    return {"m": float(round(0.2 + 0.5 * x[0], 4)), "s": float(round(16 + 48 * x[1], 2)),
            "lr": float(10 ** (-4 + x[2] * np.log10(30))), "wd": float(10 ** (-5 + x[3] * np.log10(300))),
            "shift": int(round(8 * x[4]))}


class Fitness:
    def __init__(self, train_items, val_items, epochs, log_path):
        self.tr, self.epochs, self.log_path, self.history = train_items, epochs, log_path, []
        self.va = val_items
        keep = [i for i, it in enumerate(val_items) if it[3] in ("real", "medium", "hard")]
        self.va_sub = [val_items[i] for i in keep]

    def __call__(self, x, tag="opa"):
        hp = decode(x)
        t0 = time.time()
        model, _ = train_encoder(self.tr, self.va, self.epochs, seed=0, select_best=False, **hp)
        emb = embed_all(model, self.va_sub, DEV)
        gen, imp = hard_pairs(self.va_sub)
        val, _ = eer(scores(emb, gen), scores(emb, imp))
        rec = {"search": tag, "eval": len(self.history) + 1, "hp": hp, "val_eer_hard": val, "sec": time.time() - t0}
        self.history.append(rec)
        print(f"[{tag} #{rec['eval']}] {hp} -> val EER(hard) {val * 100:.3f}%  ({rec['sec']:.0f}s)", flush=True)
        json.dump(self.history, open(self.log_path, "w"), indent=1)
        return val


def run_opa(fit, budget, pop, seed):
    return opa(lambda x: fit(x, "opa"), 5, budget, pop, seed), "OPA (Jiang et al. 2022), update rules of mealpy OriginalOrcaPA"


def opa(f, dim, budget, pop, seed, p1=0.5, p2=0.1, q=0.9, F=2.0):
    """Orca Predation Algorithm with the update rules of mealpy's OriginalOrcaPA, on [0, 1]^dim,
    stopped after `budget` fitness evaluations (every candidate costs one encoder training)."""
    rng = np.random.default_rng(seed)
    T = max(1, int(np.ceil((budget - pop) / (2 * pop))))
    X = rng.random((pop, dim)); fx = np.array([f(x) for x in X]); used = pop

    def ev(x):
        nonlocal used
        used += 1
        return f(np.clip(x, 0, 1))

    for t in range(T):
        # ---- chase: driving (two methods) or encircling, greedy selection
        g, M = X[fx.argmin()].copy(), X.mean(0)
        XC, fc = X.copy(), fx.copy()
        for i in range(pop):
            if used >= budget:
                return XC[fc.argmin()]
            if rng.random() > p1:                       # driving
                if rng.random() > q:
                    a, b, d = rng.random(3)
                    xn = X[i] + a * (d * g - F * (b * M + (1 - b) * X[i]))
                else:
                    xn = X[i] + (rng.uniform(0, 2) * g - X[i])
            else:                                       # encircling
                j1, j2, j3 = rng.choice([k for k in range(pop) if k != i], 3, replace=False)
                u = 2 * (rng.random() - 0.5) * (1 - t / T)
                xn = X[j1] + u * (X[j2] - X[j3])
            xn = np.clip(xn, 0, 1); v = ev(xn)
            if v < fx[i]:
                XC[i], fc[i] = xn, v
        # ---- attack towards the mean of the four best, perturbed by three chase positions
        B = XC[np.argsort(fc)[:4]].mean(0)
        for i in range(pop):
            if used >= budget:
                return XC[fc.argmin()]
            j = rng.choice([k for k in range(pop) if k != i], 3, replace=False)
            xa = np.clip(XC[i] + rng.uniform(0, 2) * (B - XC[i]) + rng.uniform(-2.5, 2.5) * (XC[j].mean(0) - X[i]), 0, 1)
            v = ev(xa)
            if v < fc[i]:
                XC[i], fc[i] = xa, v
            elif rng.random() < p2 and used < budget:   # adjustment: reset some coordinates to the bound
                xr = np.where(rng.random(dim) < p2, 0.0, XC[i])
                if not np.allclose(xr, XC[i]):
                    vr = ev(xr)
                    if vr < fc[i]:
                        XC[i], fc[i] = xr, vr
        X, fx = XC, fc
    return X[fx.argmin()]


def run_random(fit, budget, seed):
    rng = np.random.default_rng(seed + 1000)
    best, bx = 1.0, None
    for _ in range(budget):
        x = rng.random(5)
        v = fit(x, "random")
        if v < best:
            best, bx = v, x
    return bx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results_opt")
    ap.add_argument("--budget", type=int, default=25, help="fitness evaluations per search method")
    ap.add_argument("--pop", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=5, help="epochs per fitness evaluation")
    ap.add_argument("--final-epochs", type=int, default=15)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--skip-random", action="store_true")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    sp = split_subjects(scan_socofing(Path(args.data)))
    tr = [it for it in sp["train"] if it[3] in ("real", "easy")]
    va, te = sp["val"], sp["test"]
    print(f"device={DEV} train(real+easy)={len(tr)} val={len(va)} test={len(te)}", flush=True)
    fit = Fitness(tr, va, args.epochs, out / "search_history.json")
    res = {"device": DEV, "budget": args.budget, "pop": args.pop, "epochs_per_eval": args.epochs,
           "default": DEFAULT}

    # the default configuration is evaluated with the same short budget for reference
    inv = [(DEFAULT["m"] - 0.2) / 0.5, (DEFAULT["s"] - 16) / 48, (np.log10(DEFAULT["lr"]) + 4) / np.log10(30),
           (np.log10(DEFAULT["wd"]) + 5) / np.log10(300), DEFAULT["shift"] / 8]
    res["default_val_eer_hard"] = fit(np.array(inv), "default")

    x_opa, impl = run_opa(fit, args.budget, args.pop, seed=0)
    res["opa"] = {"implementation": impl, "best_hp": decode(x_opa),
                  "best_val_eer_hard": min(h["val_eer_hard"] for h in fit.history if h["search"] == "opa")}
    if not args.skip_random:
        x_rand = run_random(fit, args.budget, seed=0)
        res["random"] = {"best_hp": decode(x_rand),
                         "best_val_eer_hard": min(h["val_eer_hard"] for h in fit.history if h["search"] == "random")}
    res["history"] = fit.history
    json.dump(res, open(out / "optimization.json", "w"), indent=1)

    # final comparison on the untouched TEST partition, full training, several seeds
    finals = {"default": DEFAULT, "opa_best": res["opa"]["best_hp"]}
    res["final"] = {}
    for name, hp in finals.items():
        runs = []
        for s in args.seeds:
            print(f"== final {name} seed {s} {hp}", flush=True)
            model, _ = train_encoder(tr, va, args.final_epochs, s, **hp)
            rep = protocol_report(embed_all(model, va, DEV), va, embed_all(model, te, DEV), te)
            rep.pop("random_impostors_scores"); rep.pop("hard_impostors_scores")
            runs.append(rep)
            print(f"   test EER random {rep['random_impostors']['eer'] * 100:.3f}%  hard {rep['hard_impostors']['eer'] * 100:.3f}%", flush=True)
        summ = {}
        for prot in ("random_impostors", "hard_impostors"):
            for k in ("eer", "frr_at_tau", "far_at_tau", "tar_at_far_1e-4"):
                v = [r[prot][k] for r in runs]
                summ[f"{prot}/{k}"] = {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, "values": v}
            for lvl in ("easy", "medium", "hard"):
                v = [r[prot]["per_level"][lvl]["eer"] for r in runs]
                summ[f"{prot}/{lvl}/eer"] = {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, "values": v}
        res["final"][name] = {"hp": hp, "summary": summ}
        json.dump(res, open(out / "optimization.json", "w"), indent=1)
    print(f"wrote {out}/optimization.json")


if __name__ == "__main__":
    main()
