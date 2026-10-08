"""SecureContext: the only place the plaintext similarity score exists.

Software simulation of the TEE role (StrongBox / Secure Enclave / OP-TEE in production).
It holds the CKKS secret key and the ECDSA P-256 device key, decrypts the encrypted score,
applies the threshold, enforces nonce freshness and key version, and signs ONLY the
decision (accept/reject) bound to nonce and key version - the score never leaves.
"""
import hashlib
import json

import tenseal as ts
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec

from . import he, touchid
from .signers import SecureEnclaveSigner, SoftwareSigner


def canonical(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


class SoftwareSecureContext:
    def __init__(self, threshold: float = 0.3, require_touchid: bool = False, signer=None, accept_signer=None):
        """signer: signs every decision (software key by default, or a Secure Enclave key).
        accept_signer: optional presence-bound hardware key; if given, ACCEPT decisions are signed
        with it, and the hardware (not this code) enforces Touch ID before it signs."""
        self.threshold = threshold
        self.require_touchid = require_touchid
        self.signer = signer or SoftwareSigner()
        self.accept_signer = accept_signer
        self._he: dict[str, ts.Context] = {}          # per-user CKKS secret contexts
        self._active_version: dict[str, int] = {}
        self._seen_nonces: set[str] = set()
        self._server_keys: dict = {}                  # per-user pinned server public keys

    # ---------- provisioning ----------
    def public_key_pem(self) -> str:
        return self.signer.public_key_pem()

    def pin_server_key(self, user_id: str, pem: str) -> None:
        """Remember the server's public key at enrolment; later scores must carry its signature."""
        self._server_keys[user_id] = serialization.load_pem_public_key(pem.encode())

    def _check_server_sig(self, user_id, enc_score, nonce, version, server_sig):
        key = self._server_keys.get(user_id)
        if key is None:
            return  # no server key pinned (legacy enrolment): nothing to check against
        if not server_sig:
            raise PermissionError("score ciphertext not authenticated by the server")
        msg = json.dumps({"user_id": user_id, "nonce": nonce, "version": version,
                          "enc_score_sha256": hashlib.sha256(enc_score).hexdigest()},
                         sort_keys=True, separators=(",", ":")).encode()
        try:
            key.verify(server_sig, msg, ec.ECDSA(hashes.SHA256()))
        except InvalidSignature:
            raise PermissionError("score ciphertext not authenticated by the server")

    def accept_public_key_pem(self) -> str | None:
        return self.accept_signer.public_key_pem() if self.accept_signer else None

    def provision_user(self, user_id: str, version: int) -> ts.Context:
        """Create (or replace on rotation) the user's HE keys and record the active version."""
        ctx = he.new_secret_context()
        self._he[user_id] = ctx
        self._active_version[user_id] = version
        return ctx

    def revoke_version(self, user_id: str, new_version: int) -> None:
        self._active_version[user_id] = new_version

    def he_context(self, user_id: str) -> ts.Context:
        return self._he[user_id]

    # ---------- key release ----------
    def release(self, user_id: str, enc_score: bytes, nonce: str, version: int,
                server_sig: bytes | None = None) -> tuple[bytes, bytes]:
        """Return (payload, signature). Raises on stale version, replayed nonce, or a score
        ciphertext that the server did not sign for this user, nonce and version."""
        if version != self._active_version.get(user_id):
            raise PermissionError("key version is not active (rollback rejected)")
        if nonce in self._seen_nonces:
            raise PermissionError("nonce already used (replay rejected)")
        self._check_server_sig(user_id, enc_score, nonce, version, server_sig)
        self._seen_nonces.add(nonce)

        score = he.decrypt_scalar(self._he[user_id], enc_score)
        decision = bool(score >= self.threshold)
        self._last_score = score  # kept only for local display/benchmarking, never transmitted

        if decision and self.accept_signer is not None:
            # Hardware-enforced presence: the Secure Enclave refuses to sign without Touch ID.
            payload = canonical({"user_id": user_id, "nonce": nonce, "version": version,
                                 "decision": decision, "user_presence": True})
            return payload, self.accept_signer.sign(payload, f"approve SecureFinger sign-in for {user_id}")

        user_presence = False
        if decision and self.require_touchid:
            # Platform biometric authorization: the signing key is released only after Touch ID.
            if not touchid.authenticate(f"approve SecureFinger sign-in for {user_id}"):
                raise PermissionError("Touch ID not confirmed: key release denied")
            user_presence = True

        payload = canonical({"user_id": user_id, "nonce": nonce, "version": version,
                             "decision": decision, "user_presence": user_presence})
        sig = self.signer.sign(payload)
        return payload, sig


def secure_enclave_context(threshold: float, require_touchid: bool, key_dir: str) -> SoftwareSecureContext:
    """SecureContext whose device keys live in the Mac's Secure Enclave (non-exportable).
    With require_touchid, ACCEPT decisions use a second enclave key bound to Touch ID."""
    from pathlib import Path
    d = Path(key_dir).expanduser()
    signer = SecureEnclaveSigner(d / "decision.key", "none")
    accept = SecureEnclaveSigner(d / "accept.key", "biometry") if require_touchid else None
    return SoftwareSecureContext(threshold, require_touchid, signer=signer, accept_signer=accept)
