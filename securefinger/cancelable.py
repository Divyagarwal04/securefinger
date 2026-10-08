"""Cancelable transforms: key-conditioned FiLM (as documented) and Random Orthogonal Projection.

FiLM:  t = gamma(k) * x + beta(k), then L2-normalized.
       gamma/beta come from two small MLPs applied to the key vector k. The MLP weights are
       fixed public system parameters; only k is secret. gamma carries the sign of k
       (bipolar FiLM) - without sign variation, templates under different keys stay highly
       correlated and the cross-key cosine target (< 0.15) cannot be met.
ROP:   t = Q x with Q a Haar-random orthogonal matrix seeded from the key (isometry, so
       same-key matching is exactly as accurate as plaintext matching).
"""
import hashlib

import numpy as np

from . import EMBED_DIM

_PARAM_SEED = 20240521  # public system parameter for the FiLM generator weights


class FiLMGenerator:
    def __init__(self, dim: int = EMBED_DIM, hidden: int = 256, seed: int = _PARAM_SEED):
        rng = np.random.default_rng(seed)
        s1, s2 = 1 / np.sqrt(dim), 1 / np.sqrt(hidden)
        self.g1, self.g2 = rng.normal(0, s1, (dim, hidden)), rng.normal(0, s2, (hidden, dim))
        self.b1, self.b2 = rng.normal(0, s1, (dim, hidden)), rng.normal(0, s2, (hidden, dim))

    def __call__(self, k: np.ndarray):
        h_g = np.tanh(k @ self.g1)
        h_b = np.tanh(k @ self.b1)
        gamma = np.sign(k) * (1.0 + 0.25 * np.tanh(h_g @ self.g2))   # bipolar scale
        beta = 0.02 * np.tanh(h_b @ self.b2)                           # small shift
        return gamma, beta


_FILM = FiLMGenerator()


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def film_transform(x: np.ndarray, k: np.ndarray) -> np.ndarray:
    gamma, beta = _FILM(k)
    return _unit(gamma * _unit(x) + beta)


_ROP_CACHE: dict[bytes, np.ndarray] = {}
_ROP_CACHE_MAX = 64


def rop_matrix(k: np.ndarray) -> np.ndarray:
    """Q depends only on the key, so it is cached per key (i.e. per user and key version).
    Rotation changes the key and therefore the cache entry; old entries age out."""
    digest = hashlib.sha256(k.tobytes()).digest()
    q = _ROP_CACHE.get(digest)
    if q is None:
        q = _rop_matrix_uncached(digest)
        if len(_ROP_CACHE) >= _ROP_CACHE_MAX:
            _ROP_CACHE.pop(next(iter(_ROP_CACHE)))
        _ROP_CACHE[digest] = q
    return q


def _rop_matrix_uncached(digest: bytes) -> np.ndarray:
    seed = int.from_bytes(digest, "big")
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((EMBED_DIM, EMBED_DIM))
    q, r = np.linalg.qr(a)
    return q * np.sign(np.diag(r))  # Mezzadri correction -> Haar distributed


def rop_transform(x: np.ndarray, k: np.ndarray) -> np.ndarray:
    return _unit(rop_matrix(k) @ _unit(x))


TRANSFORMS = {"film": film_transform, "rop": rop_transform}


def make_template(x: np.ndarray, k: np.ndarray, method: str = "film") -> np.ndarray:
    return TRANSFORMS[method](x, k)
