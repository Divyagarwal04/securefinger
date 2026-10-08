"""Device signing keys for the SecureContext.

SoftwareSigner      ECDSA P-256 key in process memory (development emulation).
SecureEnclaveSigner ECDSA P-256 key generated INSIDE the Mac's Secure Enclave via tools/se_helper.
                    The private key is non-exportable; with mode="biometry" the enclave itself
                    refuses to sign until Touch ID succeeds (hardware-enforced user presence).
Both produce DER ECDSA signatures over SHA-256(message), so the server verifies them identically.
"""
import base64
import os
import subprocess
import sys
from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec

HELPER = Path(os.environ.get("SF_SE_HELPER", Path(__file__).resolve().parents[1] / "tools" / "se_helper"))


def _pem_from_der_b64(b64: str) -> str:
    key = serialization.load_der_public_key(base64.b64decode(b64))
    return key.public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo).decode()


class SoftwareSigner:
    kind = "software"
    hardware_presence = False

    def __init__(self):
        self._key = ec.generate_private_key(ec.SECP256R1())

    def public_key_pem(self) -> str:
        return self._key.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo).decode()

    def sign(self, message: bytes, reason: str | None = None) -> bytes:
        return self._key.sign(message, ec.ECDSA(hashes.SHA256()))


def secure_enclave_available() -> tuple[bool, str]:
    if sys.platform != "darwin":
        return False, "Secure Enclave needs macOS"
    if not HELPER.exists():
        return False, f"build the helper first: swiftc -O tools/se_helper.swift -o {HELPER}"
    r = subprocess.run([str(HELPER), "check"], capture_output=True, text=True)
    return (r.returncode == 0, r.stdout.strip() or r.stderr.strip())


class SecureEnclaveSigner:
    kind = "secure-enclave"

    def __init__(self, keyfile: str | Path, mode: str = "none", create: bool = True):
        self.keyfile, self.mode = Path(keyfile), mode
        self.hardware_presence = mode in ("biometry", "presence")
        if not self.keyfile.exists():
            if not create:
                raise FileNotFoundError(self.keyfile)
            self.keyfile.parent.mkdir(parents=True, exist_ok=True)
            self._pem = _pem_from_der_b64(self._run("create", str(self.keyfile), mode))
        else:
            self._pem = _pem_from_der_b64(self._run("pubkey", str(self.keyfile)))

    def _run(self, *args, stdin: bytes | None = None) -> str:
        r = subprocess.run([str(HELPER), *args], input=stdin, capture_output=True)
        if r.returncode != 0:
            raise PermissionError(f"Secure Enclave: {r.stderr.decode().strip()}")
        return r.stdout.decode().strip()

    def public_key_pem(self) -> str:
        return self._pem

    def sign(self, message: bytes, reason: str | None = None) -> bytes:
        args = ["sign", str(self.keyfile)] + ([reason] if reason else [])
        return base64.b64decode(self._run(*args, stdin=message))


if __name__ == "__main__":  # self-test: python -m securefinger.signers
    ok, why = secure_enclave_available()
    print("Secure Enclave helper:", ok, "-", why)
    if ok:
        import tempfile
        d = Path(tempfile.mkdtemp())
        for mode in ("none", "biometry"):
            s = SecureEnclaveSigner(d / f"{mode}.key", mode)
            msg = b"securefinger self-test"
            if mode == "biometry":
                print("Touch the sensor to let the Secure Enclave sign...")
            sig = s.sign(msg, "SecureFinger Secure Enclave self-test")
            pub = serialization.load_pem_public_key(s.public_key_pem().encode())
            pub.verify(sig, msg, ec.ECDSA(hashes.SHA256()))
            print(f"mode={mode}: enclave signature verified with the exported public key")
