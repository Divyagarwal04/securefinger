"""CKKS homomorphic encryption helpers (TenSEAL, N=8192, 128-bit parameter set)."""
import numpy as np
import tenseal as ts

POLY_DEGREE = 8192
COEFF_BITS = [60, 40, 40, 60]  # 200 bits total, within the 218-bit bound for 128-bit security
SCALE = 2 ** 40


def new_secret_context() -> ts.Context:
    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=POLY_DEGREE,
                     coeff_mod_bit_sizes=COEFF_BITS)
    ctx.global_scale = SCALE
    ctx.generate_galois_keys()
    ctx.generate_relin_keys()
    return ctx


def public_context_bytes(ctx: ts.Context) -> bytes:
    """Context the server receives: public, relinearization and Galois keys, no secret key."""
    pub = ctx.copy()
    pub.make_context_public()
    return pub.serialize(save_public_key=True, save_secret_key=False,
                         save_galois_keys=True, save_relin_keys=True)


def encrypt(ctx: ts.Context, vec: np.ndarray) -> bytes:
    return ts.ckks_vector(ctx, list(map(float, vec))).serialize()


def load_public(public_ctx: bytes) -> ts.Context:
    """Parse a public context (expensive: Galois keys). Callers should cache the result."""
    return ts.context_from(public_ctx)


def server_dot(public_ctx: ts.Context, enc_a: bytes, enc_b: bytes) -> bytes:
    """Server side: encrypted inner product over ciphertexts; never sees plaintext."""
    a = ts.ckks_vector_from(public_ctx, enc_a)
    b = ts.ckks_vector_from(public_ctx, enc_b)
    return a.dot(b).serialize()


def decrypt_scalar(ctx: ts.Context, enc: bytes) -> float:
    return float(ts.ckks_vector_from(ctx, enc).decrypt()[0])
