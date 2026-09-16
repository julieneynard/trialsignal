"""Streamlit demo frontend. Deliberately thin: it only calls the TrialSignal
API (trialsignal.serving.api) and renders the response — no scoring logic
lives here, so the demo can never drift from what the API actually does.
"""

from __future__ import annotations

import os

import httpx
import streamlit as st


def _api_url() -> str:
    """Streamlit Community Cloud's "Secrets" panel is the documented,
    first-class way to configure a deployed app (TOML, injected as
    st.secrets) -- it is not guaranteed to also land in os.environ, unlike a
    plain env var on Render or a local `TRIALSIGNAL_API_URL=... streamlit
    run`. Check st.secrets first, then fall back to os.environ, so both
    deployment paths work without relying on undocumented behavior."""
    try:
        return str(st.secrets["TRIALSIGNAL_API_URL"])
    except (KeyError, FileNotFoundError):
        return os.environ.get("TRIALSIGNAL_API_URL", "http://localhost:8000")


API_URL = _api_url()

st.set_page_config(page_title="TrialSignal", page_icon="🧬")
st.title("TrialSignal")
st.caption("Clinical trial progression risk scoring for drug/target/disease hypotheses.")
st.warning(
    "**Research/portfolio project built entirely on public data.** Not a "
    "validated clinical, investment, or regulatory decision tool. "
    "See [`docs/LIMITATIONS.md`]"
    "(https://github.com/julieneynard/trialsignal/blob/main/docs/LIMITATIONS.md) "
    "for what this model does and doesn't support."
)

with st.sidebar:
    st.markdown(f"**API:** `{API_URL}`")
    try:
        health = httpx.get(f"{API_URL}/health", timeout=5.0).json()
        st.success("API reachable") if health["status"] == "ok" else st.error("API unhealthy")
        st.markdown(f"Model loaded: **{health['model_loaded']}**")
    except httpx.HTTPError:
        st.error("Cannot reach API — is it running? (`trialsignal serve`)")

st.caption(
    "Live scoring is restricted to a small curated set of hand-verified hypotheses "
    "(see `trialsignal list-hypotheses`) — this is not a general gene/disease lookup."
)
gene_symbol = st.text_input("Target gene symbol", value="EGFR")
disease_name = st.text_input("Disease", value="non-small cell lung carcinoma")
drug_name = st.text_input("Drug (optional)", value="")

if st.button("Score", type="primary"):
    payload = {"gene_symbol": gene_symbol, "disease_name": disease_name}
    if drug_name:
        payload["drug_name"] = drug_name

    try:
        # Generous timeout: a cache-miss request triggers live Open Targets +
        # ChEMBL fetches server-side (see serving/api.py) before it can score.
        with st.spinner("Scoring (first request per hypothesis fetches live reference data)..."):
            response = httpx.post(f"{API_URL}/score", json=payload, timeout=60.0)
    except httpx.HTTPError as exc:
        st.error(f"Request failed: {exc}")
    else:
        if response.status_code == 200:
            result = response.json()
            st.metric("Risk score", f"{result['risk_score']:.2f}")
            st.caption(f"Model version: {result['model_version']}")
            for warning in result.get("warnings", []):
                st.info(warning)
            st.subheader("Top contributing features (SHAP)")
            st.table(result["top_contributions"])
        else:
            st.warning(response.json().get("detail", f"Request failed ({response.status_code})"))
