"""Security and correctness tests for the SecureFinger prototype."""
import base64
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from securefinger import he  # noqa: E402
from securefinger.cancelable import film_transform, make_template, rop_transform  # noqa: E402
from securefinger.client import SecureFingerClient  # noqa: E402
from securefinger.keys import KeyStore, derive_key_vector, new_seed  # noqa: E402
from securefinger.metrics import eer, roc_auc  # noqa: E402
from server import app as srv  # noqa: E402

rng = np.random.default_rng(7)


def unit(v):
    return v / np.linalg.norm(v)


def rand_emb():
    return unit(rng.standard_normal(256))


def noisy(x, sigma=0.02):
    return unit(x + sigma * rng.standard_normal(256))


@pytest.fixture()
def client(tmp_path):
    srv.init_db(str(tmp_path / "t.db"))
    srv._CTX.clear()
    return SecureFingerClient(TestClient(srv.app), method="film", threshold=0.5)


# ---------------------------------------------------------------- keys
def test_hkdf_deterministic_and_version_bound():
    s = new_seed()
    k1, k1b, k2 = derive_key_vector(s, "u", 1), derive_key_vector(s, "u", 1), derive_key_vector(s, "u", 2)
    assert np.array_equal(k1, k1b)
    assert not np.array_equal(k1, k2)
    assert k1.shape == (256,) and k1.min() >= -1 and k1.max() <= 1


def test_keystore_file_is_encrypted(tmp_path):
    ks = KeyStore(str(tmp_path / "ks.json"), password="pw")
    ks.create("alice")
    raw = (tmp_path / "ks.json").read_text()
    assert ks.users["alice"]["seed"] not in raw
    assert KeyStore(str(tmp_path / "ks.json"), password="pw").version("alice") == 1


# ---------------------------------------------------------------- cancelability
@pytest.mark.parametrize("method", ["film", "rop"])
def test_cross_key_unlinkability_cosine(method):
    """Templates of the same finger under different keys: |cos| < 0.15 (documented target)."""
    cos = []
    for _ in range(200):
        x = rand_emb()
        k1, k2 = derive_key_vector(new_seed(), "u", 1), derive_key_vector(new_seed(), "u", 1)
        cos.append(make_template(x, k1, method) @ make_template(x, k2, method))
    assert np.mean(np.abs(cos)) < 0.15


@pytest.mark.parametrize("method", ["film", "rop"])
def test_same_key_preserves_matching(method):
    k = derive_key_vector(new_seed(), "u", 1)
    g = [make_template(x, k, method) @ make_template(noisy(x), k, method) for x in [rand_emb() for _ in range(100)]]
    i = [make_template(rand_emb(), k, method) @ make_template(rand_emb(), k, method) for _ in range(100)]
    assert eer(g, i)[0] == 0.0


def test_rop_is_isometry():
    k = derive_key_vector(new_seed(), "u", 1)
    a, b = rand_emb(), rand_emb()
    assert abs(rop_transform(a, k) @ rop_transform(b, k) - a @ b) < 1e-9


def test_film_magnitude_linkage_is_detectable():
    """Honest check: coordinate-wise FiLM leaks |x| structure (abs-value linkage attack).
    This documents the weakness discussed in the paper rather than hiding it."""
    same, diff = [], []
    for _ in range(150):
        x, y = rand_emb(), rand_emb()
        k1, k2 = derive_key_vector(new_seed(), "u", 1), derive_key_vector(new_seed(), "u", 1)
        t1, t2, t3 = film_transform(x, k1), film_transform(x, k2), film_transform(y, k2)
        same.append(np.abs(t1) @ np.abs(t2))
        diff.append(np.abs(t1) @ np.abs(t3))
    assert roc_auc(same, diff) > 0.9          # FiLM: linkable by magnitudes
    same_r, diff_r = [], []
    for _ in range(150):
        x, y = rand_emb(), rand_emb()
        k1, k2 = derive_key_vector(new_seed(), "u", 1), derive_key_vector(new_seed(), "u", 1)
        t1, t2, t3 = rop_transform(x, k1), rop_transform(x, k2), rop_transform(y, k2)
        same_r.append(np.abs(t1) @ np.abs(t2))
        diff_r.append(np.abs(t1) @ np.abs(t3))
    assert abs(roc_auc(same_r, diff_r) - 0.5) < 0.15   # ROP: near chance


# ---------------------------------------------------------------- homomorphic encryption
def test_ckks_dot_matches_plaintext():
    ctx = he.new_secret_context()
    pub = he.load_public(he.public_context_bytes(ctx))
    for _ in range(5):
        a, b = rand_emb(), rand_emb()
        enc = he.server_dot(pub, he.encrypt(ctx, a), he.encrypt(ctx, b))
        assert abs(he.decrypt_scalar(ctx, enc) - a @ b) < 1e-4


def test_public_context_has_no_secret_key():
    ctx = he.new_secret_context()
    pub = he.load_public(he.public_context_bytes(ctx))
    assert not pub.has_secret_key()


# ---------------------------------------------------------------- protocol
def test_genuine_accepted_impostor_rejected(client):
    x = rand_emb()
    client.enroll("alice", embedding=x)
    assert client.verify("alice", embedding=noisy(x))["authenticated"] is True
    assert client.verify("alice", embedding=rand_emb())["authenticated"] is False


def test_score_never_sent_to_server(client):
    x = rand_emb()
    client.enroll("alice", embedding=x)
    client.verify("alice", embedding=noisy(x))
    payload = json.loads(client._last["payload"])
    assert set(payload) == {"user_id", "nonce", "version", "decision", "user_presence"}


def test_replay_of_signed_payload_rejected(client):
    x = rand_emb()
    client.enroll("alice", embedding=x)
    client.verify("alice", embedding=noisy(x))
    body = {"payload": base64.b64encode(client._last["payload"]).decode(),
            "signature": base64.b64encode(client._last["sig"]).decode()}
    assert client.http.post("/verify/finalize", json=body).status_code == 403


def test_nonce_cannot_be_reused_for_compute(client):
    x = rand_emb()
    client.enroll("alice", embedding=x)
    nonce = client.http.post("/verify/challenge", json={"user_id": "alice"}).json()["nonce"]
    probe = base64.b64encode(he.encrypt(client.sc.he_context("alice"), x)).decode()
    body = {"user_id": "alice", "nonce": nonce, "enc_probe": probe}
    assert client.http.post("/verify/compute", json=body).status_code == 200
    assert client.http.post("/verify/compute", json=body).status_code == 403


def test_secure_context_rejects_replayed_nonce(client):
    x = rand_emb()
    client.enroll("alice", embedding=x)
    client.verify("alice", embedding=noisy(x))
    nonce = json.loads(client._last["payload"])["nonce"]
    with pytest.raises(PermissionError):
        client.sc.release("alice", b"", nonce, client.keystore.version("alice"))


def test_tampered_payload_rejected(client):
    """A genuine signature attached to a modified payload (impostor flips decision) fails."""
    x = rand_emb()
    client.enroll("alice", embedding=x)
    client.verify("alice", embedding=rand_emb())          # real signed reject
    nonce = client.http.post("/verify/challenge", json={"user_id": "alice"}).json()["nonce"]
    forged = json.dumps({"user_id": "alice", "nonce": nonce, "version": 1, "decision": True},
                        sort_keys=True, separators=(",", ":")).encode()
    body = {"payload": base64.b64encode(forged).decode(),
            "signature": base64.b64encode(client._last["sig"]).decode()}
    assert client.http.post("/verify/finalize", json=body).status_code == 401


def test_rotation_revokes_old_template(client):
    x = rand_emb()
    client.enroll("alice", embedding=x)
    old_version = client.keystore.version("alice")
    old_key = client.keystore.key_vector("alice")
    client.rotate("alice", embedding=x)
    assert client.keystore.version("alice") == old_version + 1
    # genuine still works under the new key
    assert client.verify("alice", embedding=noisy(x))["authenticated"] is True
    # an old-key template is unlinkable to the new enrollment
    assert abs(make_template(x, old_key) @ make_template(x, client.keystore.key_vector("alice"))) < 0.3
    # signing with the revoked version is refused by the secure context
    with pytest.raises(PermissionError):
        client.sc.release("alice", b"", "fresh-nonce", old_version)


def test_enrollment_rollback_rejected(client):
    x = rand_emb()
    client.enroll("alice", embedding=x)
    client.rotate("alice", embedding=x)
    ctx = client.sc.he_context("alice")
    body = {"user_id": "alice", "pubkey_pem": client.sc.public_key_pem(),
            "he_ctx": base64.b64encode(he.public_context_bytes(ctx)).decode(),
            "enc_template": base64.b64encode(he.encrypt(ctx, x)).decode(), "key_version": 1}
    assert client.http.post("/enroll", json=body).status_code == 409


def test_audit_chain_valid_and_pii_free(client, tmp_path):
    x = rand_emb()
    client.enroll("alice", embedding=x)
    client.verify("alice", embedding=noisy(x))
    assert client.http.get("/audit/verify").json()["chain_valid"] is True
    with srv.db() as con:
        rows = [dict(r) for r in con.execute("SELECT * FROM audit")]
    assert rows and all("alice" not in json.dumps(r) for r in rows)


# ---------------------------------------------------------------- Touch ID gate (mocked off-Mac)
def test_touchid_gate_denies_key_release(client, monkeypatch):
    from securefinger import touchid
    monkeypatch.setattr(touchid, "authenticate", lambda reason, timeout=60: False)
    client.sc.require_touchid = True
    x = rand_emb()
    client.enroll("alice", embedding=x)
    with pytest.raises(PermissionError, match="Touch ID"):
        client.verify("alice", embedding=noisy(x))


def test_touchid_gate_allows_and_marks_presence(client, monkeypatch):
    from securefinger import touchid
    monkeypatch.setattr(touchid, "authenticate", lambda reason, timeout=60: True)
    client.sc.require_touchid = True
    x = rand_emb()
    client.enroll("alice", embedding=x)
    r = client.verify("alice", embedding=noisy(x))
    assert r["authenticated"] and r["user_presence"] is True
    assert json.loads(client._last["payload"])["user_presence"] is True


def test_server_can_require_presence(client, monkeypatch):
    monkeypatch.setattr(srv, "REQUIRE_PRESENCE", True)
    x = rand_emb()
    client.enroll("alice", embedding=x)
    with pytest.raises(PermissionError, match="403"):
        client.verify("alice", embedding=noisy(x))


# ---------------------------------------------------------------- presence-bound (Secure Enclave style) accept key
class _FakeEnclaveKey:
    """Stands in for a Secure Enclave key off-Mac: a separate key that only signs when 'touched'."""
    hardware_presence = True

    def __init__(self, touched=True):
        from securefinger.signers import SoftwareSigner
        self._inner, self.touched, self.calls = SoftwareSigner(), touched, 0

    def public_key_pem(self):
        return self._inner.public_key_pem()

    def sign(self, message, reason=None):
        self.calls += 1
        if not self.touched:
            raise PermissionError("Secure Enclave: sign refused (user cancelled)")
        return self._inner.sign(message)


def test_accept_requires_presence_bound_key(client):
    client.sc.accept_signer = _FakeEnclaveKey()
    x = rand_emb()
    client.enroll("alice", embedding=x)
    r = client.verify("alice", embedding=noisy(x))
    assert r["authenticated"] and r["hardware_presence"] is True
    assert client.sc.accept_signer.calls == 1
    # an ACCEPT signed with the ordinary decision key is refused once a presence key is registered
    nonce = client.http.post("/verify/challenge", json={"user_id": "alice"}).json()["nonce"]
    probe = base64.b64encode(he.encrypt(client.sc.he_context("alice"), make_template(
        noisy(x), client.keystore.key_vector("alice"), "film"))).decode()
    client.http.post("/verify/compute", json={"user_id": "alice", "nonce": nonce, "enc_probe": probe})
    p = json.dumps({"user_id": "alice", "nonce": nonce, "version": client.keystore.version("alice"),
                    "decision": True, "user_presence": True}, sort_keys=True, separators=(",", ":")).encode()
    forged = client.sc.signer.sign(p)
    r = client.http.post("/verify/finalize", json={"payload": base64.b64encode(p).decode(),
                                                    "signature": base64.b64encode(forged).decode()})
    assert r.status_code == 401


def test_reject_never_touches_presence_key_and_refusal_blocks(client):
    client.sc.accept_signer = _FakeEnclaveKey(touched=False)
    x = rand_emb()
    client.enroll("alice", embedding=x)
    assert client.verify("alice", embedding=rand_emb())["authenticated"] is False   # impostor: no prompt
    assert client.sc.accept_signer.calls == 0
    with pytest.raises(PermissionError, match="Secure Enclave"):
        client.verify("alice", embedding=noisy(x))                                   # genuine but no touch


def test_rop_matrix_cached_per_key():
    from securefinger.cancelable import rop_matrix
    k1, k2 = derive_key_vector(new_seed(), "u", 1), derive_key_vector(new_seed(), "u", 1)
    assert rop_matrix(k1) is rop_matrix(k1)
    assert not np.allclose(rop_matrix(k1), rop_matrix(k2))


# ---------------------------------------------------------------- server-authenticated score ciphertext
def _compute(client, x):
    nonce = client.http.post("/verify/challenge", json={"user_id": "alice"}).json()["nonce"]
    probe = base64.b64encode(he.encrypt(client.sc.he_context("alice"), make_template(
        x, client.keystore.key_vector("alice"), "film"))).decode()
    r = client.http.post("/verify/compute", json={"user_id": "alice", "nonce": nonce, "enc_probe": probe}).json()
    return nonce, base64.b64decode(r["enc_score"]), base64.b64decode(r["server_sig"])


def test_forged_score_ciphertext_is_refused(client):
    """A tampered client app encrypts a high score itself: the secure context must refuse it."""
    x = rand_emb()
    client.enroll("alice", embedding=x)
    v = client.keystore.version("alice")
    nonce, real_score, server_sig = _compute(client, rand_emb())          # impostor attempt
    forged = he.encrypt(client.sc.he_context("alice"), np.array([1.0]))     # decrypts to a score of 1.0
    with pytest.raises(PermissionError, match="not authenticated"):
        client.sc.release("alice", forged, nonce, v, server_sig)
    with pytest.raises(PermissionError, match="not authenticated"):
        client.sc.release("alice", forged, nonce, v, None)


def test_score_signature_is_bound_to_nonce(client):
    """A genuine, server-signed score from one attempt cannot be replayed into another attempt."""
    x = rand_emb()
    client.enroll("alice", embedding=x)
    v = client.keystore.version("alice")
    _, old_score, old_sig = _compute(client, noisy(x))                     # genuine attempt 1
    nonce2, _, _ = _compute(client, rand_emb())                           # impostor attempt 2
    with pytest.raises(PermissionError, match="not authenticated"):
        client.sc.release("alice", old_score, nonce2, v, old_sig)
    nonce3, score3, sig3 = _compute(client, noisy(x))                      # a correct release still works
    payload, _ = client.sc.release("alice", score3, nonce3, v, sig3)
    assert json.loads(payload)["decision"] is True
