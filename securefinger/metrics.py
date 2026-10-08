"""Verification metrics: EER, FAR/FRR at a threshold, and a simple cosine-based linkability score."""
import numpy as np


def eer(genuine: np.ndarray, impostor: np.ndarray) -> tuple[float, float]:
    """Return (EER, threshold at EER). Scores: higher = more similar."""
    genuine, impostor = np.asarray(genuine), np.asarray(impostor)
    thr = np.unique(np.concatenate([genuine, impostor]))
    if len(thr) > 4000:
        thr = np.quantile(thr, np.linspace(0, 1, 4000))
    far = np.array([(impostor >= t).mean() for t in thr])
    frr = np.array([(genuine < t).mean() for t in thr])
    k = int(np.argmin(np.abs(far - frr)))
    return float((far[k] + frr[k]) / 2), float(thr[k])


def rates_at(genuine, impostor, tau: float) -> tuple[float, float]:
    """(FAR, FRR) at threshold tau."""
    return float((np.asarray(impostor) >= tau).mean()), float((np.asarray(genuine) < tau).mean())


def roc_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """AUC via the Mann-Whitney statistic (used for linkage-attack scores)."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort() + 1
    rp = ranks[: len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
