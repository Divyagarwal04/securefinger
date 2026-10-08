"""Per-stage verification latency over N runs (in-process server).

Usage:
    python benchmark.py --image path/to/finger.BMP --weights models/encoder.pt --runs 100
    python benchmark.py ... --size 192            # time the 192x192 encoder (speed does not depend on weights)
    python benchmark.py ... --secure-enclave      # sign with the Mac's Secure Enclave key (reject path, no prompt)
Without --image, a synthetic embedding is used and image/encoder stages are skipped.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.append(str(_Path(__file__).resolve().parents[1]))  # repo root, for the securefinger and server packages
import argparse
import json
import platform
import tempfile
import time

import numpy as np
from fastapi.testclient import TestClient

from securefinger.client import SecureFingerClient
from securefinger.encoder import Embedder
from server import app as srv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--weights", default="models/encoder.pt")
    ap.add_argument("--method", default="film", choices=["film", "rop"])
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--size", type=int, default=96, help="input resolution of the encoder")
    ap.add_argument("--secure-enclave", action="store_true",
                    help="decision signed by the Secure Enclave; threshold set above 1 so every attempt is a "
                         "reject and no Touch ID prompt appears (times the hardware signing path)")
    args = ap.parse_args()
    import securefinger.preprocess as pp
    pp.IMG_SIZE = args.size

    srv.init_db(tempfile.mktemp(suffix=".db"))
    emb = Embedder(args.weights) if args.image else None
    c = SecureFingerClient(TestClient(srv.app), embedder=emb, method=args.method,
                           threshold=1.5 if args.secure_enclave else 0.0, secure_enclave=args.secure_enclave)
    kw = {"image": args.image} if args.image else {"embedding": np.ones(256) / 16}
    c.enroll("bench", **kw)
    c.verify("bench", **kw)  # warm-up (context parsing, caches)

    rows = []
    for _ in range(args.runs):
        t = time.perf_counter()
        c.verify("bench", **kw)
        r = dict(c.timings)
        r["end_to_end_ms"] = (time.perf_counter() - t) * 1e3
        rows.append(r)

    keys = rows[0].keys()
    summary = {k: {"median": float(np.median([r[k] for r in rows])),
                   "p95": float(np.percentile([r[k] for r in rows], 95))} for k in keys}
    summary["_machine"] = {"platform": platform.platform(), "processor": platform.processor(),
                           "python": platform.python_version(), "runs": args.runs, "method": args.method,
                           "input_size": args.size, "signer": "secure_enclave" if args.secure_enclave else "software"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
