"""SecureFinger client: capture -> embed -> cancelable template -> CKKS -> protocol with the server.

`http` can be an httpx.Client pointed at a running server, or FastAPI's TestClient
for a fully in-process demo.
"""
import base64
import time

import numpy as np

from . import he
from .cancelable import make_template
from .keys import KeyStore
from .preprocess import decode_and_preprocess, load_and_preprocess
from .secure_context import SoftwareSecureContext, secure_enclave_context

b64e = lambda b: base64.b64encode(b).decode()
b64d = lambda s: base64.b64decode(s.encode())


class SecureFingerClient:
    def __init__(self, http, embedder=None, method: str = "film", threshold: float = 0.3,
                 keystore: KeyStore | None = None, touchid: bool = False,
                 secure_enclave: bool = False, se_key_dir: str = "~/.securefinger/se_keys"):
        self.http = http
        self.embedder = embedder
        self.method = method
        self.keystore = keystore or KeyStore()
        self.sc = (secure_enclave_context(threshold, touchid, se_key_dir) if secure_enclave
                   else SoftwareSecureContext(threshold=threshold, require_touchid=touchid))
        self.timings: dict[str, float] = {}

    # ---------------------------------------------------------------- helpers
    def _embedding(self, image=None, embedding=None) -> np.ndarray:
        if embedding is not None:
            return np.asarray(embedding, dtype=np.float64)
        t = time.perf_counter()
        if isinstance(image, (bytes, bytearray)):
            x = decode_and_preprocess(bytes(image))
        else:
            x = load_and_preprocess(image)
        self.timings["preprocess_ms"] = (time.perf_counter() - t) * 1e3
        t = time.perf_counter()
        e = self.embedder.embed(x)
        self.timings["encoder_ms"] = (time.perf_counter() - t) * 1e3
        return e

    def _template(self, user_id: str, emb: np.ndarray) -> np.ndarray:
        t = time.perf_counter()
        k = self.keystore.key_vector(user_id)
        self.timings["key_ms"] = (time.perf_counter() - t) * 1e3
        t = time.perf_counter()
        tpl = make_template(emb, k, self.method)
        self.timings["transform_ms"] = (time.perf_counter() - t) * 1e3
        return tpl

    def _post(self, path: str, body: dict) -> dict:
        r = self.http.post(path, json=body)
        if r.status_code >= 400:
            raise PermissionError(f"{path} -> {r.status_code}: {r.json().get('detail')}")
        return r.json()

    # ---------------------------------------------------------------- protocol
    def enroll(self, user_id: str, image=None, embedding=None, rotate: bool = False) -> dict:
        version = self.keystore.rotate(user_id) if rotate else self.keystore.create(user_id)
        ctx = self.sc.provision_user(user_id, version)
        tpl = self._template(user_id, self._embedding(image, embedding))
        r = self._post("/enroll", {
            "user_id": user_id,
            "pubkey_pem": self.sc.public_key_pem(),
            "he_ctx": b64e(he.public_context_bytes(ctx)),
            "enc_template": b64e(he.encrypt(ctx, tpl)),
            "key_version": version,
            "accept_pubkey_pem": self.sc.accept_public_key_pem(),
        })
        if r.get("server_pubkey_pem"):
            self.sc.pin_server_key(user_id, r["server_pubkey_pem"])  # scores must be server-signed from now on
        return r

    def rotate(self, user_id: str, image=None, embedding=None) -> dict:
        """Revoke the old template: new seed + version, re-enroll the same finger."""
        return self.enroll(user_id, image, embedding, rotate=True)

    def verify(self, user_id: str, image=None, embedding=None) -> dict:
        tpl = self._template(user_id, self._embedding(image, embedding))
        ctx = self.sc.he_context(user_id)
        version = self.keystore.version(user_id)

        nonce = self._post("/verify/challenge", {"user_id": user_id})["nonce"]

        t = time.perf_counter()
        enc_probe = he.encrypt(ctx, tpl)
        self.timings["he_encrypt_ms"] = (time.perf_counter() - t) * 1e3

        t = time.perf_counter()
        resp = self._post("/verify/compute", {"user_id": user_id, "nonce": nonce, "enc_probe": b64e(enc_probe)})
        enc_score, server_sig = b64d(resp["enc_score"]), b64d(resp["server_sig"]) if resp.get("server_sig") else None
        self.timings["server_compute_ms"] = (time.perf_counter() - t) * 1e3

        t = time.perf_counter()
        payload, sig = self.sc.release(user_id, enc_score, nonce, version, server_sig)
        self.timings["secure_release_ms"] = (time.perf_counter() - t) * 1e3

        t = time.perf_counter()
        out = self._post("/verify/finalize", {"payload": b64e(payload), "signature": b64e(sig)})
        self.timings["finalize_ms"] = (time.perf_counter() - t) * 1e3
        out["local_score"] = round(self.sc._last_score, 4)  # shown locally only, never sent
        self._last = {"payload": payload, "sig": sig}
        return out
