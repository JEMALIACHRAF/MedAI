# =============================================================================
# FRONTEND — MEDAI CLINICAL PLATFORM
# Streamlit · Professional Medical AI Interface
#
# REQUIREMENTS:
#   pip install streamlit requests
#
# RUN:
#   streamlit run frontend.py
#
# BACKEND must be running on http://localhost:8000
# =============================================================================

import streamlit as st
import requests
import json

# =============================================================================
# PAGE CONFIG
# =============================================================================

BACKEND = "http://localhost:8000"

st.set_page_config(
    page_title="MedAI Clinical",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# GLOBAL STYLES — clean, dark, clinical
# =============================================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

/* ── Reset & base ─────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0d0f12;
    color: #c9d1d9;
}

/* ── Hide Streamlit chrome ─────────────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem; max-width: 1200px; }

/* ── Top bar ────────────────────────────────────────────────────────── */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.2rem 0 2rem;
    border-bottom: 1px solid #21262d;
    margin-bottom: 2.5rem;
}
.topbar-brand {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.05rem;
    font-weight: 500;
    letter-spacing: 0.04em;
    color: #e6edf3;
}
.topbar-brand span { color: #58a6ff; }
.topbar-status {
    font-size: 0.72rem;
    font-family: 'IBM Plex Mono', monospace;
    color: #3fb950;
    letter-spacing: 0.08em;
}
.topbar-status.offline { color: #f85149; }

/* ── Tab navigation ─────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: transparent;
    border-bottom: 1px solid #21262d;
    padding: 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.82rem;
    font-weight: 500;
    letter-spacing: 0.03em;
    color: #8b949e;
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    padding: 0.7rem 1.2rem;
    margin: 0;
    transition: color 0.15s, border-color 0.15s;
}
.stTabs [data-baseweb="tab"]:hover { color: #c9d1d9; }
.stTabs [aria-selected="true"] {
    color: #e6edf3 !important;
    border-bottom: 2px solid #58a6ff !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab-panel"] { padding: 2rem 0 0; }

/* ── Section headings ──────────────────────────────────────────────── */
.section-title {
    font-size: 0.72rem;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #8b949e;
    margin-bottom: 1.2rem;
}

/* ── Input fields ──────────────────────────────────────────────────── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
    color: #c9d1d9 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.88rem !important;
    padding: 0.65rem 0.9rem !important;
    transition: border-color 0.15s !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #58a6ff !important;
    box-shadow: 0 0 0 3px rgba(88,166,255,0.1) !important;
}
label { font-size: 0.82rem !important; color: #8b949e !important; margin-bottom: 0.3rem !important; }

/* ── Buttons ──────────────────────────────────────────────────────── */
.stButton > button {
    background: #21262d !important;
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
    color: #c9d1d9 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    padding: 0.5rem 1.2rem !important;
    transition: background 0.15s, border-color 0.15s !important;
}
.stButton > button:hover {
    background: #30363d !important;
    border-color: #58a6ff !important;
    color: #e6edf3 !important;
}
.stButton > button[kind="primary"] {
    background: #1f6feb !important;
    border-color: #1f6feb !important;
    color: #ffffff !important;
}
.stButton > button[kind="primary"]:hover {
    background: #388bfd !important;
    border-color: #388bfd !important;
}

/* ── File uploader ────────────────────────────────────────────────── */
.stFileUploader > div {
    background: #161b22 !important;
    border: 1px dashed #30363d !important;
    border-radius: 8px !important;
    padding: 1.5rem !important;
}
.stFileUploader > div:hover { border-color: #58a6ff !important; }
.stFileUploader label { font-size: 0.82rem !important; color: #8b949e !important; }

/* ── Result card ──────────────────────────────────────────────────── */
.result-block {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 1.4rem 1.6rem;
    margin: 1rem 0;
}
.result-label {
    font-size: 0.68rem;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #8b949e;
    margin-bottom: 0.5rem;
}
.result-value {
    font-size: 1.5rem;
    font-weight: 600;
    color: #e6edf3;
    font-family: 'IBM Plex Sans', sans-serif;
}
.result-sub {
    font-size: 0.78rem;
    color: #8b949e;
    margin-top: 0.4rem;
    font-family: 'IBM Plex Mono', monospace;
}

/* ── Prediction badge ─────────────────────────────────────────────── */
.badge-malin {
    display: inline-block;
    background: rgba(248,81,73,0.15);
    border: 1px solid rgba(248,81,73,0.4);
    color: #f85149;
    border-radius: 4px;
    padding: 2px 10px;
    font-size: 0.78rem;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 500;
    letter-spacing: 0.05em;
}
.badge-benin {
    display: inline-block;
    background: rgba(63,185,80,0.15);
    border: 1px solid rgba(63,185,80,0.4);
    color: #3fb950;
    border-radius: 4px;
    padding: 2px 10px;
    font-size: 0.78rem;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 500;
    letter-spacing: 0.05em;
}

/* ── Confidence bar ───────────────────────────────────────────────── */
.conf-bar-wrap {
    background: #21262d;
    border-radius: 3px;
    height: 4px;
    margin-top: 0.6rem;
    overflow: hidden;
}
.conf-bar-fill {
    height: 4px;
    border-radius: 3px;
    transition: width 0.4s ease;
}

/* ── Chat ──────────────────────────────────────────────────────────── */
.chat-wrap {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 1.2rem;
    max-height: 480px;
    overflow-y: auto;
    margin-bottom: 1rem;
}
.msg-user {
    text-align: right;
    margin: 0.5rem 0;
}
.msg-user span {
    display: inline-block;
    background: #1f6feb;
    color: #ffffff;
    border-radius: 10px 10px 2px 10px;
    padding: 0.55rem 1rem;
    font-size: 0.85rem;
    max-width: 75%;
    text-align: left;
    line-height: 1.5;
}
.msg-bot {
    text-align: left;
    margin: 0.5rem 0;
}
.msg-bot span {
    display: inline-block;
    background: #21262d;
    color: #c9d1d9;
    border-radius: 10px 10px 10px 2px;
    border-left: 3px solid #58a6ff;
    padding: 0.55rem 1rem;
    font-size: 0.85rem;
    max-width: 75%;
    text-align: left;
    line-height: 1.5;
}
.msg-init {
    text-align: left;
    margin: 0.5rem 0;
}
.msg-init span {
    display: inline-block;
    background: #21262d;
    color: #8b949e;
    border-radius: 10px 10px 10px 2px;
    padding: 0.55rem 1rem;
    font-size: 0.82rem;
    font-style: italic;
    max-width: 75%;
    line-height: 1.5;
}

/* ── Source chip ──────────────────────────────────────────────────── */
.source-chip {
    display: inline-block;
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.72rem;
    font-family: 'IBM Plex Mono', monospace;
    color: #8b949e;
    margin: 2px 4px 2px 0;
}

/* ── Answer block ─────────────────────────────────────────────────── */
.answer-block {
    background: #161b22;
    border: 1px solid #21262d;
    border-left: 3px solid #58a6ff;
    border-radius: 0 8px 8px 0;
    padding: 1.2rem 1.4rem;
    font-size: 0.88rem;
    line-height: 1.7;
    color: #c9d1d9;
    margin: 1rem 0;
}

/* ── Disclaimer ───────────────────────────────────────────────────── */
.disclaimer {
    background: rgba(210,153,34,0.08);
    border: 1px solid rgba(210,153,34,0.25);
    border-radius: 6px;
    padding: 0.65rem 1rem;
    font-size: 0.75rem;
    color: #d29922;
    font-family: 'IBM Plex Mono', monospace;
    margin-top: 1.2rem;
    letter-spacing: 0.02em;
}

/* ── Spinner override ────────────────────────────────────────────── */
.stSpinner > div { border-top-color: #58a6ff !important; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# BACKEND STATUS
# =============================================================================

@st.cache_data(ttl=10)
def check_backend():
    try:
        r = requests.get(f"{BACKEND}/health", timeout=3)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None

health = check_backend()
connected = health is not None

status_class = "topbar-status" if connected else "topbar-status offline"
status_text  = "SYSTEM OPERATIONAL" if connected else "BACKEND OFFLINE"

st.markdown(f"""
<div class="topbar">
    <div class="topbar-brand">Med<span>AI</span> Clinical</div>
    <div class="{status_class}">{status_text}</div>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# TABS
# =============================================================================

tab1, tab2, tab3 = st.tabs([
    "Lesion Classification",
    "Image Analysis",
    "Clinical Knowledge"
])

# =============================================================================
# TAB 1 — CNN LESION CLASSIFICATION
# =============================================================================

with tab1:
    st.markdown('<div class="section-title">Skin Lesion — Binary Classification (CNN · HAM10000)</div>', unsafe_allow_html=True)

    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        uploaded = st.file_uploader(
            "Upload dermoscopy image",
            type=["jpg", "jpeg", "png"],
            key="cnn_upload",
            label_visibility="visible"
        )
        if uploaded:
            st.image(uploaded, use_container_width=True)

        run_cnn = st.button("Run classification", type="primary", key="btn_cnn", disabled=not uploaded)

    with col_result:
        if uploaded and run_cnn:
            with st.spinner("Running inference..."):
                try:
                    files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
                    r = requests.post(f"{BACKEND}/cnn/predict", files=files, timeout=60)
                    d = r.json()

                    pred  = d.get("prediction", "N/A")
                    conf  = d.get("confidence", 0.0)
                    ms    = d.get("execution_ms", 0)
                    pct   = int(conf * 100)
                    bar_color = "#f85149" if pred == "malin" else "#3fb950"
                    badge_cls = f"badge-{pred}" if pred in ("malin", "benin") else "badge-benin"

                    st.markdown(f"""
                    <div class="result-block">
                        <div class="result-label">Prediction</div>
                        <div class="result-value">
                            <span class="{badge_cls}">{pred.upper()}</span>
                        </div>
                        <div class="conf-bar-wrap">
                            <div class="conf-bar-fill" style="width:{pct}%;background:{bar_color};"></div>
                        </div>
                        <div class="result-sub">Confidence {pct}% &nbsp;·&nbsp; {ms} ms</div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown(
                        '<div class="disclaimer">This output is not a clinical diagnosis. '
                        'Consult a licensed dermatologist for interpretation.</div>',
                        unsafe_allow_html=True
                    )

                except Exception as e:
                    st.error(f"Request failed: {e}")
        elif not connected:
            st.markdown(
                '<div class="result-block"><div class="result-label">Status</div>'
                '<div class="result-sub">Backend unreachable — start uvicorn backend:app --reload</div></div>',
                unsafe_allow_html=True
            )

# =============================================================================
# TAB 2 — HUGGINGFACE MULTIMODAL (VQA + texte)
# =============================================================================

with tab2:
    st.markdown('<div class="section-title">Multimodal Analysis — Visual QA + Text Classification (HuggingFace)</div>', unsafe_allow_html=True)

    sub_vqa, sub_text = st.tabs(["Image + Question", "Text Query"])

    # ── VQA ──
    with sub_vqa:
        c1, c2 = st.columns([1, 1], gap="large")
        with c1:
            img_vqa = st.file_uploader(
                "Upload image",
                type=["jpg", "jpeg", "png"],
                key="hf_img"
            )
            if img_vqa:
                st.image(img_vqa, use_container_width=True)

        with c2:
            question = st.text_input(
                "Clinical question",
                value="Is this skin lesion benign or malignant? Describe its characteristics.",
                key="hf_q"
            )
            run_vqa = st.button("Analyze", type="primary", key="btn_vqa", disabled=not img_vqa)

            if img_vqa and run_vqa:
                with st.spinner("Processing..."):
                    try:
                        files   = {"file": (img_vqa.name, img_vqa.getvalue(), img_vqa.type)}
                        params  = {"question": question}
                        r = requests.post(f"{BACKEND}/hf/analyze", files=files, params=params, timeout=120)
                        d = r.json()
                        answer = d.get("answer", "N/A")
                        conf   = int(d.get("confidence", 0) * 100)
                        ms     = d.get("execution_ms", 0)

                        st.markdown(f"""
                        <div class="result-block">
                            <div class="result-label">Model Response</div>
                            <div style="font-size:1.1rem;color:#e6edf3;margin:0.4rem 0;">{answer}</div>
                            <div class="result-sub">Confidence {conf}% &nbsp;·&nbsp; {ms} ms &nbsp;·&nbsp; {d.get('model','')}</div>
                        </div>
                        """, unsafe_allow_html=True)

                        st.markdown(
                            '<div class="disclaimer">Model output for research purposes only. '
                            'Not a substitute for clinical evaluation.</div>',
                            unsafe_allow_html=True
                        )

                    except Exception as e:
                        st.error(f"Request failed: {e}")

    # ── Text classification ──
    with sub_text:
        query_hf = st.text_area(
            "Describe patient symptoms or clinical context",
            placeholder="e.g. Patient presents with dry cough, low-grade fever and progressive fatigue for 5 days...",
            height=120,
            key="hf_text"
        )
        run_hf_text = st.button("Classify", type="primary", key="btn_hf_text")

        if run_hf_text and query_hf.strip():
            with st.spinner("Classifying..."):
                try:
                    r = requests.post(f"{BACKEND}/hf/chat", json={"text": query_hf}, timeout=90)
                    d = r.json()
                    preds = d.get("top_predictions", [])

                    st.markdown('<div class="result-block"><div class="result-label">Top Predictions</div>', unsafe_allow_html=True)
                    for p in preds:
                        pct = int(p["score"] * 100)
                        bar_color = "#58a6ff"
                        st.markdown(f"""
                        <div style="margin:0.5rem 0;">
                            <div style="display:flex;justify-content:space-between;font-size:0.82rem;color:#c9d1d9;margin-bottom:3px;">
                                <span>{p['label']}</span>
                                <span style="font-family:'IBM Plex Mono',monospace;color:#8b949e;">{pct}%</span>
                            </div>
                            <div class="conf-bar-wrap">
                                <div class="conf-bar-fill" style="width:{pct}%;background:{bar_color};"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    st.markdown(f'<div class="result-sub" style="margin-top:0.8rem;">{d.get("model","")} &nbsp;·&nbsp; {d.get("execution_ms",0)} ms</div></div>', unsafe_allow_html=True)

                    st.markdown(
                        '<div class="disclaimer">Zero-shot classification — indicative only. '
                        'Clinical confirmation required.</div>',
                        unsafe_allow_html=True
                    )
                except Exception as e:
                    st.error(f"Request failed: {e}")

# =============================================================================
# TAB 3 — RAG + GPT-3.5 (medical knowledge base)
# =============================================================================

with tab3:
    st.markdown('<div class="section-title">Clinical Knowledge — Retrieval-Augmented Generation (GPT-3.5)</div>', unsafe_allow_html=True)

    col_q, col_a = st.columns([1, 1], gap="large")

    with col_q:
        rag_query = st.text_area(
            "Patient question or clinical query",
            placeholder=(
                "e.g. What are the diagnostic criteria for type 2 diabetes?\n"
                "How should I manage stage 1 hypertension in a 55-year-old?"
            ),
            height=160,
            key="rag_input"
        )
        run_rag = st.button("Search knowledge base", type="primary", key="btn_rag")

        if "rag_result" in st.session_state and st.session_state.rag_result:
            d = st.session_state.rag_result
            st.markdown('<div class="section-title" style="margin-top:1.5rem;">Sources retrieved</div>', unsafe_allow_html=True)
            for src in d.get("sources", []):
                score_pct = int(src["relevance"] * 100)
                st.markdown(
                    f'<span class="source-chip">{src["title"]} — {score_pct}%</span>',
                    unsafe_allow_html=True
                )
            st.markdown(
                f'<div class="result-sub" style="margin-top:0.6rem;">{d.get("execution_ms",0)} ms</div>',
                unsafe_allow_html=True
            )

    with col_a:
        if run_rag and rag_query.strip():
            with st.spinner("Retrieving and generating..."):
                try:
                    r = requests.post(f"{BACKEND}/rag/query", json={"text": rag_query}, timeout=60)
                    d = r.json()
                    st.session_state["rag_result"] = d

                    answer = d.get("answer", "No answer returned.")
                    st.markdown(f'<div class="answer-block">{answer}</div>', unsafe_allow_html=True)
                    st.markdown(
                        '<div class="disclaimer">Generated from curated medical knowledge base. '
                        'Always verify with current clinical guidelines and a qualified physician.</div>',
                        unsafe_allow_html=True
                    )

                except Exception as e:
                    st.error(f"Request failed: {e}")

        elif "rag_result" in st.session_state and st.session_state.rag_result and not run_rag:
            d = st.session_state.rag_result
            answer = d.get("answer", "")
            if answer:
                st.markdown(f'<div class="answer-block">{answer}</div>', unsafe_allow_html=True)
                st.markdown(
                    '<div class="disclaimer">Generated from curated medical knowledge base. '
                    'Always verify with current clinical guidelines and a qualified physician.</div>',
                    unsafe_allow_html=True
                )