"""
Streamlit Web Application for Hallucination Detection System
A beautiful, interactive interface for detecting hallucinations in LLM responses
"""
import streamlit as st
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from knowledge_base.document_loader import DocumentLoader
from knowledge_base.vector_store import get_vector_store
from rag.generator import get_generator
from detection.hallucination_detector import get_hallucination_detector


# Page configuration
st.set_page_config(
    page_title="Hallucination Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


def inject_css():
    """Inject custom CSS in smaller chunks to avoid Streamlit rendering issues."""

    # Font import
    st.markdown(
        '<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">',
        unsafe_allow_html=True,
    )

    # Base styles
    st.markdown("""<style>
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: #4A4036; /* Deep taupe base text */
}
.block-container { padding: 1.5rem 2rem 2rem !important; max-width: 1280px; }
header[data-testid="stHeader"] { background: transparent !important; }
footer { display: none !important; }
hr { border: none; height: 1px; background: #E8E3DD; margin: 1.5rem 0; } /* Soft beige border */
</style>""", unsafe_allow_html=True)

    # Hero
    st.markdown("""<style>
.hero {
    background: #FFFCF9; /* Very warm off-white */
    border-radius: 24px;
    padding: 3rem 2.5rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    border: 1px solid #F0EEE9;
    box-shadow: 0 0 0 1px rgba(0,0,0,0.01), 0 8px 30px rgba(139,115,85,0.06); /* Soft brown shadow */
}
.hero::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0;
    height: 5px;
    background: linear-gradient(90deg, #D4C4B7, #E8DCCB, #C2B29F, #E8DCCB, #D4C4B7); /* Beige/Gold shimmer */
    background-size: 200% 100%;
    animation: shimmer 4s linear infinite;
}
@keyframes shimmer { 0%{background-position:200% 0} 100%{background-position:-200% 0} }
@keyframes fadeInUp { 0%{opacity:0;transform:translateY(12px)} 100%{opacity:1;transform:translateY(0)} }
.animate-in { animation: fadeInUp 0.45s cubic-bezier(.4,0,.2,1) both; }
.hero::after {
    content: '';
    position: absolute; top: 30px; right: -60px;
    width: 260px; height: 260px; border-radius: 50%;
    background: radial-gradient(circle, rgba(212,196,183,0.15) 0%, transparent 70%); /* Soft beige radial glow */
    pointer-events: none;
}
.hero-icon {
    display: inline-flex; align-items: center; justify-content: center;
    width: 56px; height: 56px; border-radius: 16px;
    background: linear-gradient(135deg, #F5EFEB, #FFFDFB);
    font-size: 1.6rem; margin-bottom: 1rem; border: 1px solid #EAE3DC;
}
.hero h1 {
    font-size: 2.2rem; font-weight: 800; color: #3A322B;
    margin: 0 0 0.4rem 0; letter-spacing: -0.02em; line-height: 1.2;
}
.hero h1 span {
    background: linear-gradient(135deg, #A48B71, #8B7355); /* Deep earthy gold */
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.hero p { color: #8A7E71; font-size: 1.05rem; margin: 0; font-weight: 400; }
</style>""", unsafe_allow_html=True)

    # Sidebar styles
    st.markdown("""<style>
section[data-testid="stSidebar"] {
    background: #FDFBFA !important; border-right: 1px solid #F0EEE9;
}
section[data-testid="stSidebar"] .stButton > button {
    background: #FFFCF9 !important; color: #7A6F62 !important;
    border: 1px solid #E8E3DD !important; box-shadow: 0 1px 2px rgba(0,0,0,0.02) !important;
    border-radius: 10px !important; font-size: 0.82rem !important; font-weight: 600 !important;
    transition: all 0.2s ease !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: #F5EFEB !important; color: #4A4036 !important; border-color: #D4C4B7 !important;
}
.sidebar-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: #F3ECE1; color: #7C6851; font-size: 0.78rem; font-weight: 600;
    padding: 6px 12px; border-radius: 8px; border: 1px solid #E0D3C1;
}
.sidebar-section {
    font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.1em; color: #A49B8F; margin-bottom: 0.5rem; margin-top: 0.5rem;
}
.tech-list { list-style: none; padding: 0; margin: 0; }
.tech-list li {
    display: flex; align-items: center; gap: 8px;
    padding: 6px 0; font-size: 0.85rem; color: #6D6154; border-bottom: 1px solid #F5EFEB;
}
.tech-list li:last-child { border-bottom: none; }
.tech-dot { width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; }
</style>""", unsafe_allow_html=True)

    # Section titles, input card, buttons
    st.markdown("""<style>
.section-title {
    display: flex; align-items: center; gap: 10px; margin-bottom: 1rem;
}
.section-title .icon-box {
    width: 36px; height: 36px; border-radius: 10px;
    display: flex; align-items: center; justify-content: center; font-size: 1rem;
}
.section-title h3 { font-size: 1.05rem; font-weight: 700; color: #3A322B; margin: 0; }

.input-card {
    background: #FFFCF9; border: 1px solid #F0EEE9; border-radius: 20px;
    padding: 1.75rem; box-shadow: 0 1px 2px rgba(0,0,0,0.02), 0 6px 20px rgba(139,115,85,0.04);
}

.stTextArea textarea {
    border-radius: 14px !important; border: 2px solid #EAE3DC !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important; font-size: 0.95rem !important;
    padding: 1rem !important; background: #FDFBFA !important;
    transition: all 0.25s ease !important; color: #4A4036 !important; line-height: 1.5 !important;
}
.stTextArea textarea:focus {
    border-color: #C2B29F !important; box-shadow: 0 0 0 4px rgba(194,178,159,0.15) !important;
    background: #FFFCF9 !important;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #A48B71 0%, #8B7355 100%) !important;
    color: #ffffff !important; border: none !important;
    padding: 0.7rem 1.5rem !important; border-radius: 14px !important;
    font-weight: 700 !important; font-size: 0.92rem !important;
    box-shadow: 0 4px 14px rgba(139, 115, 85, 0.25) !important;
    transition: all 0.3s cubic-bezier(.4,0,.2,1) !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 6px 18px rgba(139, 115, 85, 0.35) !important;
    transform: translateY(-2px) !important;
}

div[data-testid="stHorizontalBlock"] .stButton > button {
    background: #FFFCF9 !important; color: #8B7355 !important;
    border: 1.5px solid #EAE3DC !important; box-shadow: 0 1px 3px rgba(0,0,0,0.02) !important;
    font-size: 0.8rem !important; font-weight: 600 !important;
    padding: 0.5rem 0.8rem !important; border-radius: 12px !important;
    transition: all 0.2s ease !important;
}
div[data-testid="stHorizontalBlock"] .stButton > button:hover {
    background: #F5EFEB !important; border-color: #D4C4B7 !important;
    transform: translateY(-1px) !important; box-shadow: 0 3px 10px rgba(139,115,85,0.08) !important;
}
</style>""", unsafe_allow_html=True)

    # Results panel, score card, metrics, response
    st.markdown("""<style>
.results-panel {
    background: #FFFCF9; border: 1px solid #F0EEE9; border-radius: 20px;
    padding: 1.75rem; box-shadow: 0 1px 2px rgba(0,0,0,0.02), 0 6px 20px rgba(139,115,85,0.04);
    min-height: 300px;
}

.empty-state { text-align: center; padding: 3rem 1.5rem; }
.empty-state .empty-icon {
    width: 72px; height: 72px; border-radius: 20px;
    background: linear-gradient(135deg, #F5EFEB, #FFFDFB);
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 1.8rem; margin-bottom: 1rem; border: 1px solid #EAE3DC;
}
.empty-state h4 { color: #5C5346; font-size: 1rem; font-weight: 600; margin: 0 0 0.3rem 0; }
.empty-state p { color: #A49B8F; font-size: 0.85rem; margin: 0; }

.score-ring {
    border-radius: 20px; padding: 1.75rem; text-align: center;
    position: relative; overflow: hidden;
}
.score-ring .verdict-label {
    font-size: 0.75rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.12em; opacity: 0.85; margin-bottom: 0.5rem;
}
.score-ring .score-number { font-size: 3.2rem; font-weight: 800; line-height: 1; margin: 0.25rem 0; }
.score-ring .risk-badge {
    display: inline-block; padding: 4px 14px; border-radius: 999px;
    font-size: 0.75rem; font-weight: 700; background: rgba(255,255,255,0.25);
    margin-top: 0.5rem; letter-spacing: 0.05em;
}
.score-low-bg { background: linear-gradient(135deg, #6B8E7B, #8FAD9C); color: #fff; } /* Earthy Sage Green */
.score-medium-bg { background: linear-gradient(135deg, #C49A6C, #D8B48B); color: #fff; } /* Earthy Ochre/Mustard */
.score-high-bg { background: linear-gradient(135deg, #BC6C6C, #D48989); color: #fff; } /* Earthy Terracotta/Rose */

[data-testid="stMetric"] {
    background: #FDFBFA; border: 1px solid #F0EEE9; border-radius: 14px;
    padding: 0.75rem 1rem; text-align: center;
}
[data-testid="stMetricLabel"] {
    font-size: 0.72rem !important; color: #A49B8F !important;
    text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600 !important;
}
[data-testid="stMetricValue"] { font-weight: 800 !important; font-size: 1.5rem !important; color: #3A322B !important; }

.response-card {
    background: #FFFCF9; border: 1px solid #F0EEE9; border-radius: 18px;
    padding: 1.75rem; box-shadow: 0 1px 2px rgba(0,0,0,0.01), 0 4px 12px rgba(139,115,85,0.03);
}
.response-card .response-text { font-size: 0.95rem; line-height: 1.75; color: #4A4036; }

.context-block {
    background: #FDFBFA; border: 1px solid #F0EEE9; border-radius: 14px;
    padding: 1.25rem; font-size: 0.88rem; color: #7A6F62;
    line-height: 1.65; max-height: 350px; overflow-y: auto;
}
</style>""", unsafe_allow_html=True)

    # Claim cards, footer, animations
    st.markdown("""<style>
.claim {
    border-radius: 14px; padding: 1rem 1.25rem; margin-bottom: 0.75rem;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    position: relative; overflow: hidden;
}
.claim:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(139,115,85,0.05); }
.claim::before { content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 4px; }

.claim-ok { background: #F6F8F6; border: 1px solid #E1EBE2; } /* Earthy light green */
.claim-ok::before { background: #6B8E7B; }
.claim-bad { background: #FCF5F5; border: 1px solid #F3E4E4; } /* Earthy light red */
.claim-bad::before { background: #BC6C6C; }
.claim-idk { background: #FDF9F3; border: 1px solid #F6EDDF; } /* Earthy light amber */
.claim-idk::before { background: #C49A6C; }

.claim .claim-head {
    display: flex; align-items: center; gap: 6px;
    font-weight: 700; font-size: 0.82rem; margin-bottom: 0.3rem;
    color: #4A4036;
}
.claim .claim-body { font-size: 0.88rem; color: #5C5346; line-height: 1.55; padding-left: 4px; }
.claim .claim-foot {
    font-size: 0.76rem; color: #A49B8F; margin-top: 0.4rem;
    padding-left: 4px; font-style: italic;
}
.claim-tag {
    display: inline-block; font-size: 0.68rem; font-weight: 700;
    padding: 2px 8px; border-radius: 6px; text-transform: uppercase; letter-spacing: 0.05em;
}
.tag-ok { background: #E1EBE2; color: #436451; }
.tag-bad { background: #F3E4E4; color: #8F4949; }
.tag-idk { background: #F6EDDF; color: #9A7144; }

.app-footer { text-align: center; padding: 1.2rem 0 0.5rem; font-size: 0.8rem; color: #A49B8F; }

.streamlit-expanderHeader { font-weight: 600 !important; font-size: 0.88rem !important; color: #6D6154 !important; }
.stSpinner > div { border-top-color: #A48B71 !important; } /* Beige spinner */
</style>""", unsafe_allow_html=True)


# ── HELPER FUNCTIONS ─────────────────────────────────────────────────────────

def initialize_knowledge_base():
    """Initialize the knowledge base with documents."""
    vector_store = get_vector_store()
    if vector_store.get_count() > 0:
        return vector_store.get_count()

    loader = DocumentLoader()
    kb_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "knowledge_base.txt")
    if os.path.exists(kb_path):
        with st.spinner("Loading knowledge base..."):
            chunks = loader.load_and_chunk(file_path=kb_path)
            vector_store.add_documents(chunks)
            return len(chunks)
    return 0


def get_score_bg(score: float) -> str:
    if score <= 0.3:
        return "score-low-bg"
    elif score <= 0.7:
        return "score-medium-bg"
    return "score-high-bg"


def get_risk_emoji(risk_level: str) -> str:
    return {"LOW": "✅", "MEDIUM_LOW": "🟢", "MEDIUM": "🟡", "MEDIUM_HIGH": "🟠", "HIGH": "🔴"}.get(risk_level, "❓")


# ── MAIN APP ─────────────────────────────────────────────────────────────────

def main():
    """Main application."""

    # Inject all CSS
    inject_css()

    # ── Hero ──────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero animate-in">
        <div class="hero-icon">🔍</div>
        <h1>Hallucination <span>Detector</span></h1>
        <p>Verify factual claims in AI-generated text — powered by RAG &amp; NLI</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Settings")

        st.markdown('<div class="sidebar-section">Knowledge Base</div>', unsafe_allow_html=True)
        doc_count = initialize_knowledge_base()
        st.markdown(f'<div class="sidebar-badge">📚 {doc_count} chunks loaded</div>', unsafe_allow_html=True)
        st.markdown("")

        if st.button("🔄 Rebuild Knowledge Base"):
            vector_store = get_vector_store()
            vector_store.clear()
            st.rerun()

        st.divider()

        st.markdown('<div class="sidebar-section">Detection Mode</div>', unsafe_allow_html=True)
        use_rag = st.checkbox("Use RAG (Recommended)", value=True)
        if not use_rag:
            st.warning("⚠️ Without RAG, responses may be less grounded.")

        st.divider()

        st.markdown('<div class="sidebar-section">Technology Stack</div>', unsafe_allow_html=True)
        st.markdown("""
        <ul class="tech-list">
            <li><span class="tech-dot" style="background:#A48B71;"></span> Google Gemini — LLM</li>
            <li><span class="tech-dot" style="background:#6B8E7B;"></span> ChromaDB — Vectors</li>
            <li><span class="tech-dot" style="background:#C49A6C;"></span> Sentence-Transformers</li>
            <li><span class="tech-dot" style="background:#BC6C6C;"></span> NLI — Fact-checking</li>
        </ul>
        """, unsafe_allow_html=True)

    # ── Two-Column Layout ─────────────────────────────────────────────────
    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown("""
        <div class="section-title">
            <div class="icon-box" style="background:#F5EFEB; border:1px solid #EAE3DC;">💬</div>
            <h3>Ask a Question</h3>
        </div>
        """, unsafe_allow_html=True)

        query = st.text_area(
            "Your question",
            placeholder="e.g., What is hallucination in large language models?",
            height=130,
            label_visibility="collapsed",
        )

        st.caption("**Quick examples**")
        ex_cols = st.columns(3)
        examples = [
            "What is hallucination in LLMs?",
            "How does RAG work?",
            "What is transformer architecture?",
        ]
        for i, (col, ex) in enumerate(zip(ex_cols, examples)):
            with col:
                if st.button(f"📌 {ex[:22]}…", key=f"ex_{i}"):
                    query = ex
                    st.session_state["query"] = query

        st.markdown("")

        if st.button("🚀  Analyze Response", type="primary", use_container_width=True) or query:
            if query:
                with st.spinner("Generating and verifying…"):
                    try:
                        generator = get_generator()
                        result = generator.generate(query, use_rag=use_rag)
                        detector = get_hallucination_detector()
                        context = result.get("context", "No context available")
                        detection_result = detector.detect(result["answer"], context)
                        st.session_state["result"] = result
                        st.session_state["detection"] = detection_result

                        # Self-Learning / Auto-Update Mechanism
                        if detection_result["risk_level"] == "LOW" and result.get("answer"):
                            kb_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "knowledge_base.txt")
                            try:
                                new_fact = f"\n\nQ: {query}\nA: {result['answer']}"
                                
                                # Append to persistent file
                                with open(kb_path, "a", encoding="utf-8") as f:
                                    f.write(new_fact)
                                
                                # Update vector store dynamically without complete rebuild
                                from langchain_core.documents import Document
                                vector_store = get_vector_store()
                                doc = Document(page_content=new_fact, metadata={"source": "self_learned"})
                                vector_store.add_documents([doc])
                                
                                st.toast("🧠 Knowledge base successfully updated with verified fact!")
                            except Exception as e:
                                print(f"Failed to auto-update knowledge base: {e}")

                    except Exception as e:
                        st.error(f"❌ {e}")

    with right:
        st.markdown("""
        <div class="section-title">
            <div class="icon-box" style="background:#FDFBFA; border:1px solid #F0EEE9;">📊</div>
            <h3>Detection Results</h3>
        </div>
        """, unsafe_allow_html=True)

        if "detection" in st.session_state:
            det = st.session_state["detection"]
            score = det["overall_score"]
            risk = det["risk_level"]
            bg = get_score_bg(score)
            emoji = get_risk_emoji(risk)

            st.markdown(f"""
            <div class="score-ring {bg} animate-in">
                <div class="verdict-label">{emoji} {det['overall_verdict'].replace('_', ' ')}</div>
                <div class="score-number">{score:.0%}</div>
                <div class="risk-badge">Risk: {risk.replace('_', ' ')}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("")

            summary = det["summary"]
            m_cols = st.columns(4)
            m_cols[0].metric("Total", summary["total_claims"])
            m_cols[1].metric("✅ Supported", summary["supported"])
            m_cols[2].metric("❌ Contradicted", summary["contradicted"])
            m_cols[3].metric("❓ Unknown", summary["not_enough_info"])
        else:
            st.markdown("""
            <div class="empty-state">
                <div class="empty-icon">🧪</div>
                <h4>Waiting for input</h4>
                <p>Type a question and hit <strong>Analyze</strong> to see results</p>
            </div>
            """, unsafe_allow_html=True)

    # ── Full-Width Results ────────────────────────────────────────────────
    if "result" in st.session_state:
        st.divider()
        result = st.session_state["result"]
        det = st.session_state["detection"]

        st.markdown("""
        <div class="section-title">
            <div class="icon-box" style="background:#FDF9F3; border:1px solid #F6EDDF;">💡</div>
            <h3>Generated Response</h3>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="response-card animate-in">
            <div class="response-text">{result['answer']}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")

        ctx_col, claims_col = st.columns(2, gap="large")

        with ctx_col:
            st.markdown("""
            <div class="section-title">
                <div class="icon-box" style="background:#F6F8F6; border:1px solid #E1EBE2;">📚</div>
                <h3>Retrieved Context</h3>
            </div>
            """, unsafe_allow_html=True)
            with st.expander("View documents", expanded=False):
                st.markdown(f"""
                <div class="context-block">{result.get('context', 'No context available')}</div>
                """, unsafe_allow_html=True)

        with claims_col:
            st.markdown("""
            <div class="section-title">
                <div class="icon-box" style="background:#FCF5F5; border:1px solid #F3E4E4;">🔎</div>
                <h3>Claim Verification</h3>
            </div>
            """, unsafe_allow_html=True)

            if det["verification_results"]:
                for i, cr in enumerate(det["verification_results"], 1):
                    v = cr["verdict"]
                    if v == "SUPPORTED":
                        cls, tag_cls, icon = "claim-ok", "tag-ok", "✅"
                    elif v == "CONTRADICTED":
                        cls, tag_cls, icon = "claim-bad", "tag-bad", "❌"
                    else:
                        cls, tag_cls, icon = "claim-idk", "tag-idk", "❓"

                    st.markdown(f"""
                    <div class="claim {cls} animate-in" style="animation-delay:{i * 0.08}s">
                        <div class="claim-head">
                            {icon} Claim {i}
                            <span class="claim-tag {tag_cls}">{v}</span>
                        </div>
                        <div class="claim-body">{cr['claim']}</div>
                        <div class="claim-foot">Confidence: {cr['confidence']} — {cr['explanation']}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No verifiable claims found.")

    # ── Footer ────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("""
    <div class="app-footer">
        🎓 Final Year Project — Hallucination Detection using LLM &amp; RAG
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
