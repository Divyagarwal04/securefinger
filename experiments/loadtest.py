"""Closed-loop throughput test of the SecureFinger server.

Start the server in another terminal (several workers, shared database):
    SF_DB=/tmp/sf_load.db uvicorn server.app:app --port 8000 --workers 4
Then:
    python loadtest.py --url http://127.0.0.1:8000 --users 8 --levels 1 2 4 8 16 --seconds 20

Each simulated client loops over complete verifications (challenge -> encrypted compute ->
secure-context decision and ECDSA signature -> finalize) with no think time, using a pre-encrypted
probe so that the measurement reflects the protocol and server cost rather than the encoder.
Reports throughput (verifications/s) and latency percentiles per concurrency level.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.append(str(_Path(__file__).resolve().parents[1]))  # repo root, for the securefinger and server packages
import argparse
import base64
import json
import os
import platform
import statistics
import threading
import time

import httpx
import numpy as np

from securefinger import he
from securefinger.client import SecureFingerClient

b64e = lambda b: base64.b64encode(b).decode()


def unit(v):
    return v / np.linalg.norm(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--users", type=int, default=8)
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--method", default="rop")
    ap.add_argument("--out", default="results/loadtest.json")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    http = httpx.Client(base_url=args.url, timeout=120)
    clients, probes = [], []
    tag = str(int(time.time()))
    for u in range(args.users):
        uid = f"load{tag}_{u}"
        c = SecureFingerClient(http, method=args.method, threshold=0.5)
        x = unit(rng.standard_normal(256))
        c.enroll(uid, embedding=x)
        tpl = c._template(uid, unit(x + 0.02 * rng.standard_normal(256)))
        probes.append((uid, b64e(he.encrypt(c.sc.he_context(uid), tpl))))
        clients.append(c)
    print(f"enrolled {args.users} users; warming up server caches...")
    for c, (uid, probe) in zip(clients, probes):
        for _ in range(2):
            one(c, uid, probe, http)

    results = []
    for level in args.levels:
        lat, errors, stop = [], [0], time.time() + args.seconds
        lock = threading.Lock()

        def worker(k):
            h = httpx.Client(base_url=args.url, timeout=120)
            c = clients[k % len(clients)]
            uid, probe = probes[k % len(probes)]
            while time.time() < stop:
                t = time.perf_counter()
                try:
                    ok = one(c, uid, probe, h)
                except Exception:
                    ok = False
                with lock:
                    if ok:
                        lat.append((time.perf_counter() - t) * 1e3)
                    else:
                        errors[0] += 1

        t0 = time.time()
        ths = [threading.Thread(target=worker, args=(k,)) for k in range(level)]
        [t.start() for t in ths]
        [t.join() for t in ths]
        wall = time.time() - t0
        lat.sort()
        r = {"clients": level, "verifications": len(lat), "errors": errors[0],
             "throughput_per_s": len(lat) / wall,
             "latency_ms_median": statistics.median(lat) if lat else None,
             "latency_ms_p95": lat[int(0.95 * (len(lat) - 1))] if lat else None}
        results.append(r)
        print(json.dumps(r))
    chain = http.get("/audit/verify").json()["chain_valid"]
    out = {"machine": platform.platform(), "cpu_count": os.cpu_count(), "method": args.method,
           "seconds_per_level": args.seconds, "users": args.users, "levels": results, "audit_chain_valid": chain}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print("audit chain valid after load:", chain, "| wrote", args.out)


def one(c, uid, probe, h):
    nonce = h.post("/verify/challenge", json={"user_id": uid}).json()["nonce"]
    r = h.post("/verify/compute", json={"user_id": uid, "nonce": nonce, "enc_probe": probe})
    if r.status_code != 200:
        return False
    enc_score = base64.b64decode(r.json()["enc_score"])
    server_sig = base64.b64decode(r.json()["server_sig"])
    with c._lock:
        payload, sig = c.sc.release(uid, enc_score, nonce, c.keystore.version(uid), server_sig)
    r = h.post("/verify/finalize", json={"payload": b64e(payload), "signature": b64e(sig)})
    return r.status_code == 200 and r.json().get("authenticated") is True


if __name__ == "__main__":
    SecureFingerClient._lock = threading.Lock()
    main()
