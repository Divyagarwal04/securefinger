"""SecureFinger viva demo (Streamlit, five tabs).

Run from the repo root:
    streamlit run demo/viva_app.py

Tabs: Authenticate · Attack lab · Privacy lab · Inside the pipeline · Results.
The backend runs in-process (same code as the FastAPI server). Uses the trained SOCOFing encoder in
models/encoder.pt and, if present, the SOCOFing images in ../socofing_raw/SOCOFing for the pickers.
"""
import base64
import json
import random
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from securefinger import he, touchid  # noqa: E402
from securefinger.cancelable import make_template  # noqa: E402
from securefinger.client import SecureFingerClient  # noqa: E402
from securefinger.encoder import Embedder  # noqa: E402
from securefinger.keys import derive_key_vector, new_seed  # noqa: E402
from securefinger.metrics import roc_auc  # noqa: E402
from securefinger.preprocess import decode_and_preprocess  # noqa: E402
from securefinger.signers import secure_enclave_available  # noqa: E402
from server import app as srv  # noqa: E402

ASSETS = Path(__file__).with_name("assets")
LEVELS = {"Real (same capture)": ("Real", ""), "Easy · central rotation": ("Altered/Altered-Easy", "_CR"),
          "Easy · obliteration": ("Altered/Altered-Easy", "_Obl"), "Easy · z-cut": ("Altered/Altered-Easy", "_Zcut"),
          "Medium · central rotation": ("Altered/Altered-Medium", "_CR"), "Medium · z-cut": ("Altered/Altered-Medium", "_Zcut"),
          "Hard · central rotation": ("Altered/Altered-Hard", "_CR"), "Hard · obliteration": ("Altered/Altered-Hard", "_Obl"),
          "Hard · z-cut": ("Altered/Altered-Hard", "_Zcut")}
b64 = lambda b: base64.b64encode(b).decode()

st.set_page_config(page_title="SecureFinger viva demo", page_icon="🔐", layout="wide")


# ----------------------------------------------------------------------------- cached resources
@st.cache_resource
def load_embedder(path: str):
    return Embedder(path)


@st.cache_data
def list_real(root: str):
    d = Path(root) / "Real"
    return sorted(p.name for p in d.glob("*.BMP")) if d.exists() else []


def variant_path(root: str, real_name: str, level: str):
    sub, suffix = LEVELS[level]
    p = Path(root) / sub / real_name.replace(".BMP", f"{suffix}.BMP")
    return p if p.exists() else None


@st.cache_data
def embed_bytes(_emb_id, data: bytes):
    return EMB.embed(decode_and_preprocess(data))


@st.cache_resource
def demo_he_context():
    return he.new_secret_context()


def unit(v):
    return v / np.linalg.norm(v)


def img01(data: bytes):
    x = decode_and_preprocess(data)[0]
    return (x - x.min()) / (x.max() - x.min() + 1e-9)


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    weights = st.text_input("Encoder weights", str(ROOT / "models" / "encoder.pt"))
    EMB = load_embedder(weights)
    if EMB.trained:
        st.success("Trained encoder loaded")
    else:
        st.error("No trained weights found")
    data_root = st.text_input("SOCOFing folder", str(ROOT.parent / "socofing_raw" / "SOCOFing"))
    reals = list_real(data_root)
    st.caption(f"{len(reals)} real fingerprints found" if reals else "Folder not found: use file upload instead")
    method = st.radio("Cancelable transform", ["rop", "film"], horizontal=True,
                      help="ROP = adopted orthogonal projection; FiLM = earlier design, kept as a baseline")
    default_tau = float(EMB.threshold) if EMB.threshold is not None else 0.6
    tau = st.slider("Decision threshold τ (fixed on validation data)", 0.0, 1.0, round(default_tau, 3), 0.001)
    tid_ok, tid_why = touchid.available()
    use_touchid = st.checkbox("Require Touch ID for accepts", value=False, disabled=not tid_ok,
                              help="" if tid_ok else tid_why)
    se_ok, se_why = secure_enclave_available()
    use_se = st.checkbox("Device keys in Secure Enclave", value=False, disabled=not se_ok,
                         help="" if se_ok else se_why)
    if st.button("Reset system"):
        st.session_state.clear()
        st.rerun()

config = (method, use_se, use_se and use_touchid, weights)
if "client" not in st.session_state or st.session_state.get("config") != config:
    srv.init_db(tempfile.mktemp(suffix=".db"))
    srv._CTX.clear()
    st.session_state.client = SecureFingerClient(TestClient(srv.app), embedder=EMB, method=method,
                                                 touchid=use_touchid, secure_enclave=use_se)
    st.session_state.config = config
    st.session_state.enrolled = {}
C: SecureFingerClient = st.session_state.client
C.sc.threshold = tau
C.sc.require_touchid = use_touchid


def pick_image(label, key, default_idx=0, allow_level=True, default_level="Real (same capture)"):
    """Return (bytes, caption) from the SOCOFing pickers or an upload."""
    if reals:
        cols = st.columns([3, 2]) if allow_level else [st.container()]
        name = cols[0].selectbox(label, reals, index=min(default_idx, len(reals) - 1), key=key + "_n")
        level = default_level
        if allow_level:
            level = cols[1].selectbox("Impression", list(LEVELS), index=list(LEVELS).index(default_level),
                                      key=key + "_l")
        p = variant_path(data_root, name, level)
        if p is None:
            st.warning("That alteration does not exist for this finger; pick another.")
            return None, None
        return p.read_bytes(), f"{name.replace('.BMP', '')} · {level}"
    up = st.file_uploader(label, type=["bmp", "png", "jpg", "tif"], key=key + "_u")
    return (up.getvalue(), up.name) if up else (None, None)


st.title("SecureFinger — cancelable, encrypted fingerprint authentication")
st.caption("Divya Agrawal · Omkaesh Kumar · Mohammad Ayaan — C.V. Raman Global University")
tabs = st.tabs(["🔐 Authenticate", "🛡️ Attack lab", "🕶️ Privacy lab", "🔬 Inside the pipeline", "📊 Results"])

# ============================================================================= 1. Authenticate
with tabs[0]:
    user = st.text_input("User ID", "alice")
    c1, c2, c3 = st.columns([1, 1.2, 1])
    with c1:
        st.subheader("1 · Enrol")
        enr, enr_cap = pick_image("Enrolment finger", "enr", 0, allow_level=False)
        if enr:
            st.image(img01(enr), caption=f"{enr_cap} (preprocessed 96×96)", width=170, clamp=True)
        if st.button("Enrol", disabled=enr is None, type="primary"):
            try:
                r = C.enroll(user, image=enr)
                st.session_state.enrolled[user] = enr
                st.success(f"Enrolled · key version {r['key_version']} · server key pinned")
            except Exception as e:
                st.error(str(e))
    with c2:
        st.subheader("2 · Verify")
        mode = st.radio("Probe", ["Same finger, another impression", "Someone else's finger"], horizontal=True)
        if mode.startswith("Same") and reals and enr_cap:
            idx = reals.index(enr_cap.split(" · ")[0] + ".BMP") if enr_cap.split(" · ")[0] + ".BMP" in reals else 0
            prb, prb_cap = pick_image("Probe finger", "prb", idx, default_level="Easy · central rotation")
        else:
            prb, prb_cap = pick_image("Probe finger", "prb2", 7, default_level="Real (same capture)")
        if prb:
            st.image(img01(prb), caption=prb_cap, width=170, clamp=True)
        if st.button("Verify", disabled=prb is None or user not in st.session_state.enrolled, type="primary"):
            if use_touchid:
                st.info("If the fingerprint matches, touch the Touch ID sensor to approve.")
            try:
                t0 = time.perf_counter()
                r = C.verify(user, image=prb)
                total = (time.perf_counter() - t0) * 1e3
                if r["authenticated"]:
                    extra = (" · Touch ID enforced by the Secure Enclave" if r.get("hardware_presence")
                             else " · Touch ID confirmed" if r.get("user_presence") else "")
                    st.success(f"✅ Accepted — session token issued{extra}")
                else:
                    st.error("❌ Rejected")
                a, b = st.columns(2)
                a.markdown("**Client (secure context) sees**")
                a.metric("Decrypted similarity", f"{r['local_score']:.4f}", help=f"threshold τ = {tau:.3f}")
                b.markdown("**Server sees**")
                b.code(json.dumps(json.loads(C._last["payload"]), indent=1), language="json")
                b.caption("Signed decision only — no score, no template.")
                tm = {k.replace("_ms", ""): v for k, v in C.timings.items()}
                st.caption(f"Where the time goes ({total:.1f} ms end to end, incl. any Touch ID wait)")
                st.bar_chart(pd.DataFrame({"ms": tm}), horizontal=True, height=230)
            except Exception as e:
                st.error(str(e))
    with c3:
        st.subheader("3 · Revoke")
        st.caption("New seed and key version → a new, unlinkable template from the same finger. "
                   "The old version is refused.")
        if st.button("Rotate key & re-enrol", disabled=user not in st.session_state.enrolled):
            r = C.rotate(user, image=st.session_state.enrolled[user])
            st.success(f"Rotated · now key version {r['key_version']}")
        st.subheader("Audit log")
        chain = C.http.get("/audit/verify").json()["chain_valid"]
        st.write("Hash chain valid:", "✅" if chain else "❌")
        with srv.db() as con:
            rows = [dict(x) for x in con.execute("SELECT event, subject FROM audit ORDER BY id DESC LIMIT 8")]
        st.dataframe(rows, hide_index=True, width="stretch")
    with st.expander("What the server stores"):
        with srv.db() as con:
            u = con.execute("SELECT user_id, key_version, length(enc_template) AS template_ciphertext_bytes, "
                            "length(he_ctx) AS public_context_bytes FROM users").fetchall()
        st.dataframe([dict(x) for x in u], hide_index=True)
        st.caption("Only CKKS ciphertexts and public keys: no image, no embedding, no score.")

# ============================================================================= 2. Attack lab
with tabs[1]:
    st.subheader("Try to break it")
    st.caption("Each button performs a real attack against the running server and secure context.")
    users = list(st.session_state.enrolled)
    if not users:
        st.info("Enrol a user in the Authenticate tab first.")
    else:
        au = st.selectbox("Target user", users)
        v = C.keystore.version(au)

        def blocked(msg):
            st.success(f"🛡️ Blocked — {msg}")

        def broke(msg):
            st.error(f"⚠️ Attack succeeded — {msg}")

        a1, a2 = st.columns(2)
        with a1:
            st.markdown("**A · Replay a signed accept**")
            st.caption("Send the last signed decision to the server a second time.")
            if st.button("Replay", key="atk1"):
                if not getattr(C, "_last", None):
                    st.warning("Run one verification first.")
                else:
                    r = C.http.post("/verify/finalize", json={"payload": b64(C._last["payload"]),
                                                              "signature": b64(C._last["sig"])})
                    if r.status_code != 200:
                        blocked(r.json().get("detail"))
                    else:
                        broke("replay accepted")
            st.markdown("**B · Flip reject → accept**")
            st.caption("Take a signed decision, change it to accept, keep the old signature.")
            if st.button("Tamper", key="atk2"):
                if not getattr(C, "_last", None):
                    st.warning("Run one verification first.")
                else:
                    p = json.loads(C._last["payload"]); p["decision"] = True
                    p["nonce"] = C.http.post("/verify/challenge", json={"user_id": au}).json()["nonce"]
                    forged = json.dumps(p, sort_keys=True, separators=(",", ":")).encode()
                    r = C.http.post("/verify/finalize", json={"payload": b64(forged), "signature": b64(C._last["sig"])})
                    if r.status_code != 200:
                        blocked(r.json().get("detail"))
                    else:
                        broke("tampered payload accepted")
            st.markdown("**C · Reuse a nonce**")
            st.caption("Ask the server to compute twice with the same one-time challenge.")
            if st.button("Reuse nonce", key="atk3"):
                n = C.http.post("/verify/challenge", json={"user_id": au}).json()["nonce"]
                probe = he.encrypt(C.sc.he_context(au), C._template(au, unit(np.random.randn(256))))
                body = {"user_id": au, "nonce": n, "enc_probe": b64(probe)}
                first = C.http.post("/verify/compute", json=body).status_code
                r = C.http.post("/verify/compute", json=body)
                if r.status_code != 200:
                    blocked(f"first use {first}, second use {r.status_code}: {r.json().get('detail')}")
                else:
                    broke("nonce reused")
        with a2:
            st.markdown("**D · Hacked client injects 'score = 1.0'**")
            st.caption("A tampered app runs an impostor attempt, then swaps the server's encrypted score for its "
                       "own encryption of 1.0 before handing it to the secure context.")
            if st.button("Inject fake score", key="atk4"):
                n = C.http.post("/verify/challenge", json={"user_id": au}).json()["nonce"]
                probe = he.encrypt(C.sc.he_context(au), C._template(au, unit(np.random.randn(256))))
                resp = C.http.post("/verify/compute", json={"user_id": au, "nonce": n, "enc_probe": b64(probe)}).json()
                fake = he.encrypt(C.sc.he_context(au), np.array([1.0]))
                try:
                    C.sc.release(au, fake, n, v, base64.b64decode(resp["server_sig"]))
                    broke("fake score was signed")
                except PermissionError as e:
                    blocked(str(e))
            st.markdown("**E · Roll back to an old key version**")
            st.caption("Use a key version that is no longer active (rotate first to make an old one).")
            if st.button("Roll back", key="atk5"):
                old = v - 1 if v > 1 else v + 7
                try:
                    C.sc.release(au, b"", "fresh-nonce", old)
                    broke("old version accepted")
                except PermissionError as e:
                    blocked(f"version {old}: {e}")
            st.markdown("**F · Stolen database**")
            st.caption("What an attacker who copies the server's database actually gets.")
            if st.button("Dump the database", key="atk6"):
                with srv.db() as con:
                    row = con.execute("SELECT enc_template FROM users WHERE user_id=?", (au,)).fetchone()
                ct = row["enc_template"]
                st.code(ct[:96].hex(" ", 8) + " …", language="text")
                st.success(f"🛡️ {len(ct):,} bytes of CKKS ciphertext. Without the user's secret key it reveals "
                           "nothing, and the user can revoke it by rotating the key.")

# ============================================================================= 3. Privacy lab
with tabs[2]:
    st.subheader("Can templates from two services be linked?")
    st.caption("The same finger is enrolled at two services with two different keys. An attacker holding both "
               "templates tries to tell whether they belong to the same person.")
    p1, p2 = st.columns(2)
    with p1:
        fa, fa_cap = pick_image("Finger at service 1", "pl1", 0, allow_level=False)
    with p2:
        fb, fb_cap = pick_image("Same finger at service 2", "pl2", 0, default_level="Easy · central rotation")
    if fa and fb and st.button("Compare across keys", type="primary"):
        ea, eb = embed_bytes(weights, fa), embed_bytes(weights, fb)
        other = embed_bytes(weights, Path(data_root, "Real", reals[(reals.index(fa_cap.split(" · ")[0] + ".BMP") + 37)
                                                                   % len(reals)]).read_bytes()) if reals else unit(np.random.randn(256))
        k1, k2 = derive_key_vector(new_seed(), "svc1", 1), derive_key_vector(new_seed(), "svc2", 1)
        rows = []
        for m in ("rop", "film"):
            t1, t2, to = make_template(ea, k1, m), make_template(eb, k2, m), make_template(other, k2, m)
            rows.append({"transform": m.upper(),
                         "same key (service matching)": float(make_template(ea, k1, m) @ make_template(eb, k1, m)),
                         "cosine, same finger": float(t1 @ t2), "cosine, other finger": float(t1 @ to),
                         "magnitude attack, same finger": float(np.abs(t1) @ np.abs(t2)),
                         "magnitude attack, other finger": float(np.abs(t1) @ np.abs(to))})
        st.dataframe(pd.DataFrame(rows).set_index("transform").round(3), width="stretch")
        st.caption(f"Plaintext similarity of the two impressions: {float(ea @ eb):.3f}. Same-key matching keeps it; "
                   "across keys the cosine is near 0 for both transforms, but FiLM's magnitude score still separates "
                   "the same finger from a different one.")
    st.divider()
    st.markdown("**Measure it on many fingers**")
    n_f = st.slider("Number of fingers", 50, 400, 150, 50)
    if st.button("Run linkage attacks", disabled=not reals):
        rng = random.Random(0)
        names = rng.sample(reals, min(n_f, len(reals)))
        bar = st.progress(0.0, "embedding images…")
        pairs = []
        for i, nm in enumerate(names):
            alt = variant_path(data_root, nm, "Easy · central rotation") or variant_path(data_root, nm, "Easy · z-cut")
            if alt is None:
                continue
            pairs.append((embed_bytes(weights, Path(data_root, "Real", nm).read_bytes()), embed_bytes(weights, alt.read_bytes())))
            bar.progress((i + 1) / len(names), "embedding images…")
        res = []
        for m in ("rop", "film"):
            mated, non = {"cosine": [], "magnitude": []}, {"cosine": [], "magnitude": []}
            for i, (a, b) in enumerate(pairs):
                c = pairs[(i + 1) % len(pairs)][1]
                k1, k2 = derive_key_vector(new_seed(), "a", 1), derive_key_vector(new_seed(), "b", 1)
                ta, tb, tc = make_template(a, k1, m), make_template(b, k2, m), make_template(c, k2, m)
                mated["cosine"].append(ta @ tb); non["cosine"].append(ta @ tc)
                mated["magnitude"].append(np.abs(ta) @ np.abs(tb)); non["magnitude"].append(np.abs(ta) @ np.abs(tc))
            res.append({"transform": m.upper(), "cosine attack AUC": roc_auc(mated["cosine"], non["cosine"]),
                        "magnitude attack AUC": roc_auc(mated["magnitude"], non["magnitude"])})
        bar.empty()
        st.dataframe(pd.DataFrame(res).set_index("transform").round(3), width="stretch")
        st.caption(f"{len(pairs)} mated and {len(pairs)} non-mated pairs, every template under its own fresh key. "
                   "AUC 0.5 = guessing, 1.0 = perfect linking. Report: FiLM magnitude 1.00, ROP ≈ 0.5.")

# ============================================================================= 4. Inside the pipeline
with tabs[3]:
    st.subheader("Follow one fingerprint through the system")
    pi, pi_cap = pick_image("Fingerprint", "pipe", 3)
    if pi:
        k = derive_key_vector(new_seed(), "demo", 1)
        e = embed_bytes(weights, pi)
        t = make_template(e, k, method)
        ctx = demo_he_context()
        s1, s2, s3 = st.columns(3)
        s1.markdown("**1 · Raw capture**"); s1.image(pi, caption=pi_cap, width=170)
        s2.markdown("**2 · Preprocessed**"); s2.image(img01(pi), caption="ROI crop · CLAHE · 96×96 · normalised",
                                                       width=170, clamp=True)
        s3.markdown("**3 · Key**"); s3.caption("HKDF-SHA256(seed, user, version) → 256 numbers in [−1, 1]")
        s3.line_chart(pd.DataFrame({"key": k}), height=150)
        s4, s5 = st.columns(2)
        s4.markdown("**4 · Embedding** (ResNet-18 + ArcFace, length 1)")
        s4.line_chart(pd.DataFrame({"embedding": e}), height=170)
        s5.markdown(f"**5 · Cancelable template** ({method.upper()} under the key)")
        s5.line_chart(pd.DataFrame({"template": t}), height=170)
        st.caption(f"Similarity of embedding and template: {float(e @ t):+.3f} — the template looks unrelated to the "
                   "embedding, yet two templates under the same key compare exactly like the embeddings.")
        if ctx is not None:
            ct = he.encrypt(ctx, t)
            pub = he.public_context_bytes(ctx)
            st.markdown("**6 · What leaves the device: CKKS ciphertext**")
            st.code(ct[:128].hex(" ", 8) + " …", language="text")
            m1, m2, m3 = st.columns(3)
            m1.metric("Template ciphertext", f"{len(ct) / 1024:.0f} KiB")
            m2.metric("Public context (once per enrolment)", f"{len(pub) / 1e6:.1f} MB")
            m3.metric("Plaintext numbers visible to the server", "0")

# ============================================================================= 5. Results
with tabs[4]:
    st.subheader("Key results")
    r1 = st.columns(4)
    r1[0].metric("SOCOFing test EER", "0.027%"); r1[0].caption("2 errors in 7,431")
    r1[1].metric("Real multi-session EER", "16.8%"); r1[1].caption("nested selection")
    r1[2].metric("Minutiae matcher, same pairs", "5.8%"); r1[2].caption("SourceAFIS")
    r1[3].metric("Verification time", "33.8 ms"); r1[3].caption("laptop, 96 px")
    r2 = st.columns(4)
    r2[0].metric("Linkability D↔sys", "FiLM 1.00"); r2[0].caption("ROP below noise floor")
    r2[1].metric("Collection linkage, 32 known links", "96% vs 0.2%"); r2[1].caption("shared vs per-user keys")
    r2[2].metric("Throughput", "≈ 59 / s"); r2[2].caption("0 errors in 5,562")
    r2[3].metric("Automated tests", "27 / 27"); r2[3].caption("Linux and macOS")
    st.markdown("**Real-fingerprint accuracy: 28.7% → 24.0% → 16.2% → 16.8% (nested)**")
    figs = [("F6_fvc2.png", "EER per real database: zero-shot, first fine-tuning, 192 px recipe"),
            ("F6_minutiae.png", "Our encoder vs the SourceAFIS minutiae matcher"),
            ("F6_4x.png", "Harder SOCOFing protocols (3 seeds)"),
            ("F6_6x.png", "Linkability D↔sys: FiLM fully linkable, ROP below the noise floor"),
            ("F6_collection.png", "Linking whole collections: shared key vs per-user keys"),
            ("F6_peruser.png", "Known-plaintext attack within one user's key"),
            ("F6_7x.png", "Known-plaintext attack with a service-wide key"),
            ("F6_5c.png", "Latency of each verification stage"),
            ("F6_load.png", "Throughput and latency under load")]
    for i in range(0, len(figs), 2):
        cols = st.columns(2)
        for c, (f, cap) in zip(cols, figs[i:i + 2]):
            if (ASSETS / f).exists():
                c.image(str(ASSETS / f), caption=cap, width="stretch")
