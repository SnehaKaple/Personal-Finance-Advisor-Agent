"""
Personal Finance Advisor — Streamlit UI
Agentic AI Capstone Project 2026


Run with: streamlit run capstone_streamlit.py
"""

import uuid
import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # Loads GROQ_API_KEY from .env

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Personal Finance Advisor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Layout ── */
    .main { background-color: #0f1117; }
    .block-container { padding-top: 1.5rem; }

    /* ── Header ── */
    .main-header {
        background: linear-gradient(135deg, #0d2137 0%, #1a3a2a 100%);
        padding: 1.4rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.2rem;
        border: 1px solid #1e3a50;
    }
    .main-header h1 { margin: 0; font-size: 1.9rem; }
    .main-header p  { margin: 0.25rem 0 0; opacity: 0.8; font-size: 0.9rem; }

    /* ── Chat messages ── */
    .stChatMessage { border-radius: 12px; margin-bottom: 6px; }

    /* ── Route badges ── */
    .route-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 700;
        margin-right: 6px;
        letter-spacing: 0.5px;
    }
    .badge-ADVICE    { background:#1a3a2a; color:#4caf50; border:1px solid #4caf50; }
    .badge-CALCULATE { background:#1a2a3a; color:#2196f3; border:1px solid #2196f3; }
    .badge-SKIP      { background:#2a2a1a; color:#ff9800; border:1px solid #ff9800; }
    .badge-unknown   { background:#2a1a1a; color:#f44336; border:1px solid #f44336; }

    /* ── Confidence score colours ── */
    .score-high { color: #4caf50; font-weight: 700; font-size: 12px; }
    .score-mid  { color: #ff9800; font-weight: 700; font-size: 12px; }
    .score-low  { color: #f44336; font-weight: 700; font-size: 12px; }

    /* ── Source tags ── */
    .source-tag {
        display: inline-block;
        background: #1e2530;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 2px 8px;
        font-size: 11px;
        color: #94a3b8;
        margin: 2px;
    }

    /* ── Sidebar ── */
    .sidebar-topic { font-size: 12.5px; color: #cbd5e1; margin: 2px 0; }
</style>
""", unsafe_allow_html=True)

# ── Session State Initialisation ──────────────────────────────────────────────
def _init_state() -> None:
    defaults = {
        "agent":            None,
        "messages":         [],
        "thread_id":        str(uuid.uuid4()),
        "total_queries":    0,
        "total_confidence": 0.0,
        "user_name":        None,
        "prefill":          "",
        "init_error":       "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ── Lazy-load agent (cached per session) ─────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_agent():
    from agent import FinanceAdvisorAgent
    return FinanceAdvisorAgent()

if st.session_state.agent is None:
    try:
        st.session_state.agent = load_agent()
    except EnvironmentError as e:
        st.session_state.init_error = str(e)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💰 Finance Advisor")
    st.markdown("*Agentic AI Capstone 2026*")
    st.divider()

    st.markdown("### 📚 Topics Covered")
    topics = [
        "💼 Budgeting & Saving",
        "📈 Investing Basics",
        "🏦 Fixed Deposits & Bonds",
        "💳 Credit Cards & Debt",
        "🏠 Home Loan / EMI Planning",
        "📊 Mutual Funds & SIP",
        "🧾 Tax Saving (80C, HRA, NPS)",
        "🎯 Goal-Based Planning",
        "📉 Stock Market Basics",
        "🛡️ Insurance Planning",
        "💰 Emergency Fund",
        "🔢 Financial Calculations",
    ]
    for t in topics:
        st.markdown(f"<div class='sidebar-topic'>{t}</div>", unsafe_allow_html=True)

    st.divider()
    st.markdown("### 📊 Session Stats")
    avg_conf = (
        st.session_state.total_confidence / st.session_state.total_queries
        if st.session_state.total_queries > 0 else 0.0
    )
    c1, c2 = st.columns(2)
    c1.metric("Queries", st.session_state.total_queries)
    c2.metric("Avg Confidence", f"{avg_conf:.2f}")

    st.divider()

    # New conversation button
    if st.button("🔄 New Conversation", use_container_width=True):
        for key in ["messages", "total_queries", "total_confidence", "user_name"]:
            st.session_state[key] = [] if key == "messages" else (0.0 if key == "total_confidence" else (0 if key == "total_queries" else None))
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.agent = None
        st.cache_resource.clear()
        st.rerun()

    st.divider()
    st.markdown("### ⚡ Quick Questions")
    sample_qs = [
        "How should I start investing with ₹5000/month?",
        "What is the 50-30-20 budgeting rule?",
        "How does SIP work in mutual funds?",
        "Calculate EMI for ₹10 lakh loan at 8% for 5 years",
        "What are the best tax-saving options under 80C?",
        "How much emergency fund should I maintain?",
        "Difference between term and whole life insurance?",
    ]
    for q in sample_qs:
        if st.button(q, use_container_width=True, key=f"sq_{hash(q)}"):
            st.session_state.prefill = q
            st.rerun()

    st.divider()
    st.markdown(
        "<small style='color:#475569;'>Personal Finance Advisor · Agentic AI Capstone 2026<br>"
        "Subhang Raj · Roll No. 2328131 · KIIT DU</small>",
        unsafe_allow_html=True,
    )

# ── Main Header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>💰 Personal Finance Advisor</h1>
    <p>Your AI-powered assistant for budgeting, investing, tax saving, loans, and more — tailored for India.</p>
</div>
""", unsafe_allow_html=True)

# ── Init Error Banner ─────────────────────────────────────────────────────────
if st.session_state.init_error:
    st.error(
        f"⚠️ **Setup Required:** {st.session_state.init_error}\n\n"
        "**Fix:** Open the `.env` file and replace `your_groq_api_key_here` with your actual Groq API key.\n"
        "Get a free key at: https://console.groq.com"
    )
    st.stop()

# ── Helper: render metadata row ───────────────────────────────────────────────
def render_meta(route: str, confidence: float, sources: list[str]) -> None:
    badge_cls = f"badge-{route}" if route in ("ADVICE", "CALCULATE", "SKIP") else "badge-unknown"
    badge_html = f'<span class="route-badge {badge_cls}">{route}</span>'

    conf_cls = "score-high" if confidence >= 0.75 else ("score-mid" if confidence >= 0.5 else "score-low")
    conf_emoji = "🟢" if confidence >= 0.75 else ("🟡" if confidence >= 0.5 else "🔴")
    conf_html = f'<span class="{conf_cls}">{conf_emoji} Confidence: {confidence:.2f}</span>'

    src_html = " ".join(f'<span class="source-tag">📌 {s}</span>' for s in sources)

    st.markdown(f"{badge_html} {conf_html} &nbsp; {src_html}", unsafe_allow_html=True)

# ── Render Chat History ───────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "meta" in msg:
            meta = msg["meta"]
            render_meta(meta.get("route", "ADVICE"), meta.get("confidence", 0.0), meta.get("sources", []))

# ── Chat Input ────────────────────────────────────────────────────────────────
prefill = st.session_state.pop("prefill", "")
user_input = st.chat_input("Ask a personal finance question... e.g. 'How do I start a SIP?'")
if not user_input and prefill:
    user_input = prefill

if user_input and user_input.strip():
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Analysing your query..."):
            try:
                result = st.session_state.agent.run(user_input)
                answer     = result.get("answer", "Sorry, I could not generate a response.")
                route      = result.get("route", "ADVICE")
                confidence = result.get("confidence", 0.8)
                sources    = result.get("sources", [])

                if result.get("user_name"):
                    st.session_state.user_name = result["user_name"]

                st.markdown(answer)
                render_meta(route, confidence, sources)

                # Persist to session state
                st.session_state.messages.append({
                    "role":    "assistant",
                    "content": answer,
                    "meta":    {"route": route, "confidence": confidence, "sources": sources},
                })
                st.session_state.total_queries    += 1
                st.session_state.total_confidence += confidence

            except Exception as exc:
                error_msg = f"❌ Error: {exc}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

    st.rerun()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#475569; font-size:0.78rem;'>"
    "Personal Finance Advisor · Agentic AI Capstone 2026 · "
    "Subhang Raj · Roll No. 2328131 · KIIT DU · "
    "Powered by Groq LLaMA 3.3 70B + LangGraph + Streamlit"
    "</p>",
    unsafe_allow_html=True,
)
