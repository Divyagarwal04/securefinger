"""Key management: 32-byte seeds, HKDF-SHA256 key vectors, versioning, encrypted keystore file."""
import base64
import json
import os
from pathlib import Path

import numpy as np
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from . import EMBED_DIM


def new_seed() -> bytes:
    return os.urandom(32)


def derive_key_bytes(seed: bytes, user_id: str, version: int, length: int = 4 * EMBED_DIM) -> bytes:
    """HKDF-SHA256(seed) bound to user and key version."""
    info = f"securefinger|{user_id}|v{version}".encode()
    return HKDF(algorithm=hashes.SHA256(), length=length, salt=None, info=info).derive(seed)


def derive_key_vector(seed: bytes, user_id: str, version: int) -> np.ndarray:
    """k in [-1, 1]^256 from 1024 bytes of HKDF output (uint32 per coordinate)."""
    okm = derive_key_bytes(seed, user_id, version)
    u = np.frombuffer(okm, dtype=">u4").astype(np.float64)
    return u / (2 ** 32 - 1) * 2.0 - 1.0


class KeyStore:
    """Client-side store of per-user seeds and active key versions.

    Development stand-in for Android Keystore/StrongBox: the file is encrypted with
    AES-256-GCM under a PBKDF2-derived key (as in the documented prototype).
    """

    def __init__(self, path: str | None = None, password: str = "dev-password"):
        self.path = Path(path) if path else None
        self._key = self._kdf(password)
        self.users: dict[str, dict] = {}
        if self.path and self.path.exists():
            self._load()

    @staticmethod
    def _kdf(password: str) -> bytes:
        salt = b"securefinger-keystore-salt"  # fixed salt acceptable for a dev keystore
        return PBKDF2HMAC(hashes.SHA256(), 32, salt, 200_000).derive(password.encode())

    def _load(self):
        blob = json.loads(self.path.read_text())
        nonce, ct = base64.b64decode(blob["nonce"]), base64.b64decode(blob["ct"])
        self.users = json.loads(AESGCM(self._key).decrypt(nonce, ct, None))

    def _save(self):
        if not self.path:
            return
        nonce = os.urandom(12)
        ct = AESGCM(self._key).encrypt(nonce, json.dumps(self.users).encode(), None)
        self.path.write_text(json.dumps({"nonce": base64.b64encode(nonce).decode(),
                                         "ct": base64.b64encode(ct).decode()}))

    def create(self, user_id: str) -> int:
        self.users[user_id] = {"seed": new_seed().hex(), "version": 1}
        self._save()
        return 1

    def rotate(self, user_id: str) -> int:
        entry = self.users[user_id]
        entry["seed"] = new_seed().hex()
        entry["version"] += 1
        self._save()
        return entry["version"]

    def version(self, user_id: str) -> int:
        return self.users[user_id]["version"]

    def key_vector(self, user_id: str) -> np.ndarray:
        e = self.users[user_id]
        return derive_key_vector(bytes.fromhex(e["seed"]), user_id, e["version"])

    def key_bytes(self, user_id: str) -> bytes:
        e = self.users[user_id]
        return derive_key_bytes(bytes.fromhex(e["seed"]), user_id, e["version"], length=32)
