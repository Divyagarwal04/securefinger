"""End-to-end protocol demo that needs NO dataset or trained model.

Shows: enrollment, genuine accept, impostor reject, replay rejection, tamper rejection,
key rotation (revocation), rollback rejection and the audit chain.
Embeddings are synthetic unit vectors standing in for the encoder output; everything
after the encoder (FiLM/ROP, HKDF, CKKS, server, SecureContext, ECDSA) is the real code.

    python quickstart.py            # FiLM (documented design)
    python quickstart.py --method rop
    python quickstart.py --touchid    # Mac: Touch ID must approve every accepted sign-in
"""
import argparse
import base64
import json
import tempfile

import numpy as np
from fastapi.testclient import TestClient

from securefinger.client import SecureFingerClient
from server import app as srv


def unit(v):
    return v / np.linalg.norm(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="film", choices=["film", "rop"])
    ap.add_argument("--touchid", action="store_true", help="require Touch ID before key release (macOS)")
    ap.add_argument("--secure-enclave", action="store_true",
                    help="keep the device keys in the Mac's Secure Enclave (build tools/se_helper first)")
    args = ap.parse_args()
    if args.touchid:
        from securefinger import touchid
        ok, why = touchid.available()
        if not ok:
            raise SystemExit(f"--touchid unavailable: {why}")
        print("Touch ID ready: you will be asked to touch the sensor for each ACCEPTED sign-in.")

    if args.secure_enclave:
        from securefinger.signers import secure_enclave_available
        ok, why = secure_enclave_available()
        if not ok:
            raise SystemExit(f"--secure-enclave unavailable: {why}")
        print("Device keys: Secure Enclave (non-exportable)" + (", ACCEPT key bound to Touch ID" if args.touchid else ""))

    srv.init_db(tempfile.mktemp(suffix=".db"))
    http = TestClient(srv.app)
    c = SecureFingerClient(http, method=args.method, threshold=0.5, touchid=args.touchid,
                           secure_enclave=args.secure_enclave)
    rng = np.random.default_rng(1)
    finger = unit(rng.standard_normal(256))
    same_finger_again = unit(finger + 0.02 * rng.standard_normal(256))
    other_finger = unit(rng.standard_normal(256))

    step = lambda s: print(f"\n== {s}")
    step(f"Enroll alice (transform={args.method})")
    r = c.enroll("alice", embedding=finger); r.pop("server_pubkey_pem", None); print(r, "(server public key pinned)")

    step("Verify with alice's finger")
    r = c.verify("alice", embedding=same_finger_again)
    print({"authenticated": r["authenticated"], "touch_id_confirmed": r.get("user_presence"),
           "presence_enforced_by_hardware": r.get("hardware_presence"),
           "score_inside_secure_context": r["local_score"]})
    print("signed payload sent to server:", json.loads(c._last["payload"]))

    step("Verify with someone else's finger")
    r = c.verify("alice", embedding=other_finger)
    print({"authenticated": r["authenticated"], "score_inside_secure_context": r["local_score"]})

    step("Replay the last signed payload")
    body = {"payload": base64.b64encode(c._last["payload"]).decode(),
            "signature": base64.b64encode(c._last["sig"]).decode()}
    print(http.post("/verify/finalize", json=body).json())

    step("Tamper: flip decision to true, reuse signature")
    p = json.loads(c._last["payload"]); p["decision"] = True
    p["nonce"] = http.post("/verify/challenge", json={"user_id": "alice"}).json()["nonce"]
    forged = json.dumps(p, sort_keys=True, separators=(",", ":")).encode()
    print(http.post("/verify/finalize", json={"payload": base64.b64encode(forged).decode(),
                                               "signature": body["signature"]}).json())

    step("Tamper: client app swaps in its own encrypted 'score = 1.0'")
    from securefinger import he as _he
    n2 = http.post("/verify/challenge", json={"user_id": "alice"}).json()["nonce"]
    probe = _he.encrypt(c.sc.he_context("alice"), c._template("alice", other_finger))
    resp = http.post("/verify/compute", json={"user_id": "alice", "nonce": n2,
                                             "enc_probe": base64.b64encode(probe).decode()}).json()
    fake = _he.encrypt(c.sc.he_context("alice"), np.array([1.0]))
    try:
        c.sc.release("alice", fake, n2, c.keystore.version("alice"), base64.b64decode(resp["server_sig"]))
        print("forged score was accepted (should not happen)")
    except PermissionError as e:
        print("secure context:", e)

    step("Revoke: rotate key and re-enroll the same finger")
    r = c.rotate("alice", embedding=finger); r.pop("server_pubkey_pem", None); print(r)
    r = c.verify("alice", embedding=same_finger_again)
    print("verify after rotation:", r["authenticated"])
    try:
        c.sc.release("alice", b"", "any-nonce", version=1)
    except PermissionError as e:
        print("old key version 1 at secure context:", e)

    step("Audit log")
    print(http.get("/audit/verify").json())
    with srv.db() as con:
        for row in con.execute("SELECT event, subject FROM audit ORDER BY id"):
            print(" ", dict(row))


if __name__ == "__main__":
    main()
