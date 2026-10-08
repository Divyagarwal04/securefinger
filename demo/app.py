"""SecureFinger live demo (Streamlit).

Run from the repo root:
    streamlit run demo/app.py
The backend runs in-process; the same client code works against a real server via httpx.
"""
import sys
import tempfile
from pathlib import Path

import streamlit as st
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from securefinger import touchid  # noqa: E402
from securefinger.client import SecureFingerClient  # noqa: E402
from securefinger.encoder import Embedder  # noqa: E402
from securefinger.preprocess import decode_and_preprocess  # noqa: E402
from server import app as srv  # noqa: E402

st.set_page_config(page_title="SecureFinger", page_icon="🔐", layout="wide")
st.title("SecureFinger — cancelable, encrypted fingerprint authentication")


@st.cache_resource
def load_embedder(path: str):
    return Embedder(path)


with st.sidebar:
    st.header("Settings")
    weights = st.text_input("Encoder weights", str(ROOT / "models" / "encoder.pt"))
    embedder = load_embedder(weights)
    if embedder.trained:
        st.success("Trained encoder loaded")
    else:
        st.error("No trained weights found — scores are meaningless until you run train.py")
    method = st.radio("Cancelable transform", ["film", "rop"], horizontal=True,
                      help="FiLM = documented design; ROP = orthogonal projection (resists magnitude linkage)")
    default_tau = float(embedder.threshold) if embedder.threshold is not None else 0.3
    tau = st.slider("Decision threshold τ", -1.0, 1.0, round(default_tau, 3), 0.001)
    tid_ok, tid_why = touchid.available()
    use_touchid = st.checkbox("Require Touch ID for key release", value=False, disabled=not tid_ok,
                              help="macOS only. Touch ID approves the ECDSA signing of an ACCEPT decision. "
                                   "It does not read the fingerprint image." + ("" if tid_ok else f" ({tid_why})"))
    from securefinger.signers import secure_enclave_available
    se_ok, se_why = secure_enclave_available()
    use_se = st.checkbox("Device keys in Secure Enclave", value=False, disabled=not se_ok,
                         help="Non-exportable ECDSA keys created inside the Mac's Secure Enclave. With Touch ID "
                              "required, ACCEPT decisions use a second enclave key that the hardware only lets "
                              "sign after Touch ID." + ("" if se_ok else f" ({se_why})"))
    if st.button("Reset system"):
        st.session_state.clear()
        st.rerun()

config = (method, use_se, use_se and use_touchid)
if "client" not in st.session_state or st.session_state.get("config") != config:
    srv.init_db(tempfile.mktemp(suffix=".db"))
    srv._CTX.clear()
    st.session_state.client = SecureFingerClient(TestClient(srv.app), embedder=embedder, method=method,
                                                 touchid=use_touchid, secure_enclave=use_se)
    st.session_state.config = config
    st.session_state.log = []
client: SecureFingerClient = st.session_state.client
client.sc.threshold = tau
client.sc.require_touchid = use_touchid


def show_image(col, data, caption):
    x = decode_and_preprocess(data)[0]
    col.image(((x - x.min()) / (x.max() - x.min() + 1e-9)), caption=caption, width=160, clamp=True)


user = st.text_input("User ID", "alice")
c1, c2, c3 = st.columns(3)

with c1:
    st.subheader("1 · Enroll")
    f_enroll = st.file_uploader("Enrollment fingerprint", type=["bmp", "png", "jpg", "tif"], key="enr")
    if f_enroll:
        show_image(c1, f_enroll.getvalue(), "preprocessed (CLAHE, 96×96)")
    if st.button("Enroll", disabled=not f_enroll):
        try:
            r = client.enroll(user, image=f_enroll.getvalue())
            st.success(f"Enrolled · key version {r['key_version']}")
            st.session_state.log.append(f"enroll {user} v{r['key_version']}")
        except Exception as e:
            st.error(str(e))

with c2:
    st.subheader("2 · Verify")
    f_probe = st.file_uploader("Probe fingerprint", type=["bmp", "png", "jpg", "tif"], key="prb")
    if f_probe:
        show_image(c2, f_probe.getvalue(), "preprocessed probe")
    if st.button("Verify", disabled=not f_probe):
        if use_touchid:
            st.info("If the fingerprint matches, touch the Touch ID sensor to approve.")
        try:
            r = client.verify(user, image=f_probe.getvalue())
            if r["authenticated"]:
                extra = (" · Touch ID enforced by the Secure Enclave" if r.get("hardware_presence")
                         else " · Touch ID confirmed" if r.get("user_presence") else "")
                st.success(f"✅ Authenticated — session token issued{extra}")
            else:
                st.error("❌ Rejected")
            st.metric("Similarity (decrypted inside SecureContext only)", f"{r['local_score']:.4f}",
                      help="Shown here for the demo. The server receives only the signed accept/reject bit.")
            st.caption("Per-stage latency (ms)")
            st.json({k: round(v, 2) for k, v in client.timings.items()})
            st.session_state.log.append(f"verify {user}: {'accept' if r['authenticated'] else 'reject'}")
        except Exception as e:
            st.error(str(e))

with c3:
    st.subheader("3 · Revoke (rotate key)")
    st.caption("New seed + key version → new, unlinkable template from the same finger. "
               "The old version is refused by the secure context and the server.")
    if st.button("Rotate key & re-enroll", disabled=not f_enroll):
        try:
            r = client.rotate(user, image=f_enroll.getvalue())
            st.success(f"Rotated · now key version {r['key_version']}")
            st.session_state.log.append(f"rotate {user} -> v{r['key_version']}")
        except Exception as e:
            st.error(str(e))
    st.subheader("Audit")
    chain = client.http.get("/audit/verify").json()["chain_valid"]
    st.write("Hash chain valid:", "✅" if chain else "❌")
    with srv.db() as con:
        rows = [dict(r) for r in con.execute("SELECT event, subject, hash FROM audit ORDER BY id DESC LIMIT 8")]
    st.dataframe(rows, width='stretch', hide_index=True)

with st.expander("What the server stores (no plaintext biometrics)"):
    with srv.db() as con:
        u = con.execute("SELECT user_id, key_version, length(enc_template) AS ct_bytes, "
                        "length(he_ctx) AS public_ctx_bytes FROM users").fetchall()
    st.dataframe([dict(r) for r in u], hide_index=True)
