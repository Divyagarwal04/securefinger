"""SecureFinger backend (FastAPI).

Stores only CKKS ciphertexts, computes encrypted similarity, and finalizes authentication by
verifying the device's ECDSA signature over (decision, nonce, key version).
The encrypted score is returned with the server's own ECDSA signature over
(user, nonce, key version, SHA-256 of the ciphertext), so the secure context can check that the
ciphertext it decrypts really came from the server for this transaction.
The server never holds a plaintext template or a plaintext score.
"""
import base64
import hashlib
import json
import os
import secrets
import sqlite3
import time
from pathlib import Path

import jwt
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from securefinger import he

NONCE_TTL = 300  # seconds
REQUIRE_PRESENCE = os.environ.get("SF_REQUIRE_PRESENCE") == "1"  # demand Touch ID/BiometricPrompt
JWT_SECRET = os.environ.get("SF_JWT_SECRET", secrets.token_hex(32))
DB_PATH = os.environ.get("SF_DB", str(Path(__file__).with_name("securefinger.db")))

from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(_app):
    init_db()
    yield


app = FastAPI(title="SecureFinger backend", lifespan=lifespan)


# --------------------------------------------------------------------------- server signing key
_SERVER_KEY = {}


def server_key() -> ec.EllipticCurvePrivateKey:
    """P-256 key shared by all workers: created once next to the database, then loaded."""
    path = os.environ.get("SF_SERVER_KEY", DB_PATH + ".server_key.pem")
    if path not in _SERVER_KEY:
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            key = ec.generate_private_key(ec.SECP256R1())
            with os.fdopen(fd, "wb") as fh:
                fh.write(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                           serialization.NoEncryption()))
        except FileExistsError:
            for _ in range(50):  # another worker may still be writing it
                data = Path(path).read_bytes()
                if data:
                    break
                time.sleep(0.02)
            key = serialization.load_pem_private_key(data, password=None)
        _SERVER_KEY[path] = key
    return _SERVER_KEY[path]


def server_public_key_pem() -> str:
    return server_key().public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo).decode()


def score_message(user_id: str, nonce: str, version: int, enc_score: bytes) -> bytes:
    """What the server signs for each encrypted score (canonical JSON)."""
    return json.dumps({"user_id": user_id, "nonce": nonce, "version": version,
                       "enc_score_sha256": hashlib.sha256(enc_score).hexdigest()},
                      sort_keys=True, separators=(",", ":")).encode()


# --------------------------------------------------------------------------- storage
def db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH, timeout=30)
    con.row_factory = sqlite3.Row
    return con


def init_db(path: str | None = None) -> None:
    global DB_PATH
    if path:
        DB_PATH = path
    with db() as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS users(
            user_id TEXT PRIMARY KEY, pubkey_pem TEXT, he_ctx BLOB,
            enc_template BLOB, key_version INTEGER, accept_pubkey_pem TEXT);
        CREATE TABLE IF NOT EXISTS revoked(user_id TEXT, key_version INTEGER);
        CREATE TABLE IF NOT EXISTS nonces(
            nonce TEXT PRIMARY KEY, user_id TEXT, expires REAL, state TEXT);
        CREATE TABLE IF NOT EXISTS audit(
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, event TEXT,
            subject TEXT, detail TEXT, prev_hash TEXT, hash TEXT);
        """)
        cols = [r[1] for r in con.execute("PRAGMA table_info(users)")]
        if "accept_pubkey_pem" not in cols:  # databases created before the Secure Enclave upgrade
            con.execute("ALTER TABLE users ADD COLUMN accept_pubkey_pem TEXT")


def audit(event: str, user_id: str, detail: dict | None = None) -> None:
    """Hash-chained, PII-free audit log (user id is stored only as a salted hash)."""
    subject = hashlib.sha256(f"sf-audit|{user_id}".encode()).hexdigest()[:16]
    detail_s = json.dumps(detail or {}, sort_keys=True)
    # Read-last-hash and append must be one write transaction, otherwise concurrent server
    # workers could both chain onto the same previous record and fork the chain.
    con = sqlite3.connect(DB_PATH, timeout=30, isolation_level=None)
    con.row_factory = sqlite3.Row
    try:
        con.execute("BEGIN IMMEDIATE")
        row = con.execute("SELECT hash FROM audit ORDER BY id DESC LIMIT 1").fetchone()
        prev = row["hash"] if row else "0" * 64
        ts_ = time.time()
        h = hashlib.sha256(f"{prev}|{ts_}|{event}|{subject}|{detail_s}".encode()).hexdigest()
        con.execute("INSERT INTO audit(ts,event,subject,detail,prev_hash,hash) VALUES(?,?,?,?,?,?)",
                    (ts_, event, subject, detail_s, prev, h))
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    finally:
        con.close()


def verify_audit_chain() -> bool:
    with db() as con:
        prev = "0" * 64
        for r in con.execute("SELECT * FROM audit ORDER BY id"):
            h = hashlib.sha256(f"{prev}|{r['ts']}|{r['event']}|{r['subject']}|{r['detail']}".encode()).hexdigest()
            if r["prev_hash"] != prev or r["hash"] != h:
                return False
            prev = h
    return True


_CTX: dict[tuple[str, int], object] = {}


def _public_ctx(con, user_id: str, version: int):
    """Parsed public CKKS context per (user, key version); loaded from the DB once."""
    if (user_id, version) not in _CTX:
        blob = con.execute("SELECT he_ctx FROM users WHERE user_id=?", (user_id,)).fetchone()["he_ctx"]
        _CTX[(user_id, version)] = he.load_public(blob)
    return _CTX[(user_id, version)]


b64e = lambda b: base64.b64encode(b).decode()
b64d = lambda s: base64.b64decode(s.encode())


# --------------------------------------------------------------------------- schemas
class EnrollReq(BaseModel):
    user_id: str
    pubkey_pem: str
    he_ctx: str          # base64 public CKKS context (no secret key)
    enc_template: str    # base64 CKKS ciphertext
    key_version: int
    # Optional second device key that the hardware only lets sign after biometric user presence
    # (Secure Enclave key with .biometryCurrentSet). If registered, ACCEPT decisions must use it.
    accept_pubkey_pem: str | None = None


class ChallengeReq(BaseModel):
    user_id: str


class ComputeReq(BaseModel):
    user_id: str
    nonce: str
    enc_probe: str


class FinalizeReq(BaseModel):
    payload: str         # base64 canonical JSON signed by the device
    signature: str       # base64 DER ECDSA signature


# --------------------------------------------------------------------------- endpoints
@app.post("/enroll")
def enroll(req: EnrollReq):
    with db() as con:
        row = con.execute("SELECT key_version FROM users WHERE user_id=?", (req.user_id,)).fetchone()
        if row and req.key_version <= row["key_version"]:
            audit("enroll_rejected_rollback", req.user_id, {"v": req.key_version})
            raise HTTPException(409, "key version must increase (rollback rejected)")
        if row:  # re-enrollment after rotation: revoke old version
            con.execute("INSERT INTO revoked VALUES(?,?)", (req.user_id, row["key_version"]))
        con.execute("INSERT OR REPLACE INTO users VALUES(?,?,?,?,?,?)",
                    (req.user_id, req.pubkey_pem, b64d(req.he_ctx), b64d(req.enc_template), req.key_version,
                     req.accept_pubkey_pem))
    audit("enroll", req.user_id, {"v": req.key_version, "rotation": bool(row),
                                  "presence_key": req.accept_pubkey_pem is not None})
    return {"status": "enrolled", "key_version": req.key_version, "server_pubkey_pem": server_public_key_pem()}


@app.get("/server/pubkey")
def server_pubkey():
    return {"server_pubkey_pem": server_public_key_pem()}


@app.post("/verify/challenge")
def challenge(req: ChallengeReq):
    with db() as con:
        if not con.execute("SELECT 1 FROM users WHERE user_id=?", (req.user_id,)).fetchone():
            raise HTTPException(404, "unknown user")
        nonce = secrets.token_hex(16)
        con.execute("INSERT INTO nonces VALUES(?,?,?,?)", (nonce, req.user_id, time.time() + NONCE_TTL, "issued"))
    return {"nonce": nonce, "ttl": NONCE_TTL}


@app.post("/verify/compute")
def compute(req: ComputeReq):
    with db() as con:
        n = con.execute("SELECT * FROM nonces WHERE nonce=?", (req.nonce,)).fetchone()
        if not n or n["user_id"] != req.user_id or n["state"] != "issued" or n["expires"] < time.time():
            audit("compute_rejected", req.user_id)
            raise HTTPException(403, "invalid, expired or reused nonce")
        u = con.execute("SELECT enc_template, key_version FROM users WHERE user_id=?", (req.user_id,)).fetchone()
        ctx = _public_ctx(con, req.user_id, u["key_version"])
        enc_score = he.server_dot(ctx, u["enc_template"], b64d(req.enc_probe))
        con.execute("UPDATE nonces SET state='computed' WHERE nonce=?", (req.nonce,))
    audit("compute", req.user_id)
    sig = server_key().sign(score_message(req.user_id, req.nonce, u["key_version"], enc_score),
                            ec.ECDSA(hashes.SHA256()))
    return {"enc_score": b64e(enc_score), "server_sig": b64e(sig)}


@app.post("/verify/finalize")
def finalize(req: FinalizeReq):
    raw = b64d(req.payload)
    try:
        p = json.loads(raw)
        user_id, nonce, version, decision = p["user_id"], p["nonce"], p["version"], p["decision"]
    except Exception:
        raise HTTPException(400, "malformed payload")
    with db() as con:
        u = con.execute("SELECT pubkey_pem, key_version, accept_pubkey_pem FROM users WHERE user_id=?",
                        (user_id,)).fetchone()
        if not u:
            raise HTTPException(404, "unknown user")
        hw_presence = bool(decision) and bool(u["accept_pubkey_pem"])
        # An ACCEPT must be signed by the presence-bound hardware key when one is registered.
        pem = u["accept_pubkey_pem"] if hw_presence else u["pubkey_pem"]
        pub = serialization.load_pem_public_key(pem.encode())
        try:
            pub.verify(b64d(req.signature), raw, ec.ECDSA(hashes.SHA256()))
        except (InvalidSignature, ValueError):
            audit("finalize_rejected_signature", user_id)
            raise HTTPException(401, "bad signature")
        n = con.execute("SELECT * FROM nonces WHERE nonce=?", (nonce,)).fetchone()
        if not n or n["user_id"] != user_id or n["state"] != "computed" or n["expires"] < time.time():
            audit("finalize_rejected_nonce", user_id)
            raise HTTPException(403, "invalid, expired or replayed nonce")
        con.execute("UPDATE nonces SET state='used' WHERE nonce=?", (nonce,))
        if version != u["key_version"]:
            audit("finalize_rejected_version", user_id, {"v": version})
            raise HTTPException(403, "stale key version (rollback rejected)")
    if not decision:
        audit("verify_reject", user_id)
        return {"authenticated": False}
    if REQUIRE_PRESENCE and not (hw_presence or p.get("user_presence")):
        audit("verify_rejected_no_presence", user_id)
        raise HTTPException(403, "platform biometric confirmation required")
    presence = hw_presence or bool(p.get("user_presence"))
    token = jwt.encode({"sub": user_id, "ver": version, "uv": presence, "hw": hw_presence,
                        "exp": int(time.time()) + 900}, JWT_SECRET, "HS256")
    audit("verify_accept", user_id, {"user_presence": presence, "hardware_presence": hw_presence})
    return {"authenticated": True, "user_presence": presence, "hardware_presence": hw_presence, "token": token}


@app.get("/audit/verify")
def audit_verify():
    return {"chain_valid": verify_audit_chain()}
