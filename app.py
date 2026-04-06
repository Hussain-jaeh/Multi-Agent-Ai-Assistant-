"""
Multi-Agent AI Academic Assistant — Streamlit Application
"""

import json
import os
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# ── Bootstrap ─────────────────────────────────────────────────────────────────
load_dotenv()

import os as _os
# Ensure GEMINI_API_KEY is set — prefer explicit GEMINI_API_KEY, fall back to GOOGLE_API_KEY
if not _os.getenv("GEMINI_API_KEY") and _os.getenv("GOOGLE_API_KEY"):
    _os.environ["GEMINI_API_KEY"] = _os.getenv("GOOGLE_API_KEY")

st.set_page_config(
    page_title="Academic AI Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design System ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Reset & Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] ul li,
[data-testid="stMarkdownContainer"] ol li,
[data-testid="stMarkdownContainer"] td,
[data-testid="stMarkdownContainer"] th { color: #111827 !important; }
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4 { color: #030712 !important; }
[data-testid="stMarkdownContainer"] strong { color: #030712 !important; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Page background ── */
.stApp { background: #e8edf2; }
.main .block-container {
    background: white;
    border-radius: 16px;
    padding: 2rem 2.5rem 4rem !important;
    max-width: 1100px;
    margin-top: 1rem;
    margin-bottom: 1rem;
    min-height: calc(100vh - 2rem);
    box-shadow: 0 2px 12px rgba(0,0,0,.06);
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: none;
}
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
section[data-testid="stSidebar"] .stTextInput input {
    background: #1a1f2e !important;
    border: 1px solid #1e2130 !important;
    color: #e2e8f0 !important;
    border-radius: 10px !important;
    font-size: 0.9rem !important;
    padding: 0.5rem 0.85rem !important;
}
section[data-testid="stSidebar"] .stTextInput input:focus {
    border-color: #3b82f6 !important;
    box-shadow: none !important;
}
section[data-testid="stSidebar"] hr {
    border-color: #1e2130 !important;
    margin: 0.75rem 0 !important;
}

/* ── Sidebar nav radio → nav items ── */
section[data-testid="stSidebar"] [data-testid="stRadio"] > div {
    gap: 2px !important;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label {
    background: transparent;
    border-radius: 12px;
    padding: 0.7rem 0.9rem !important;
    display: flex !important;
    align-items: center;
    gap: 0.75rem;
    margin: 0 !important;
    cursor: pointer;
    transition: background 0.15s;
    width: 100%;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label > div:first-child {
    display: none !important;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label p {
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    color: #94a3b8 !important;
    margin: 0 !important;
    line-height: 1.4 !important;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: #1a1f2e !important;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
    background: #3b82f6 !important;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p {
    color: white !important;
    font-weight: 600 !important;
}

/* ── Page header ── */
.page-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 0 0 2rem 0;
    border-bottom: 1px solid #f1f5f9;
    margin-bottom: 2rem;
}
.page-header-icon {
    width: 48px; height: 48px;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 22px;
}
.page-header h1 {
    margin: 0; font-size: 1.5rem; font-weight: 700;
    color: #0f172a; letter-spacing: -0.02em;
}
.page-header p { margin: 0.2rem 0 0; font-size: 0.85rem; color: #64748b; }

/* ── Metric cards ── */
.metric-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 2rem; }
.metric-card {
    background: white;
    border: 1px solid #f1f5f9;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,.04), 0 1px 2px rgba(0,0,0,.06);
}
.metric-label { font-size: 0.75rem; font-weight: 500; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem; }
.metric-value { font-size: 1.75rem; font-weight: 700; color: #0f172a; letter-spacing: -0.02em; }
.metric-delta { font-size: 0.75rem; margin-top: 0.25rem; font-weight: 500; }
.delta-up { color: #10b981; }
.delta-neutral { color: #64748b; }
.delta-down { color: #ef4444; }

/* ── Section headers ── */
.section-header {
    font-size: 0.75rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.08em; color: #94a3b8; margin: 0 0 0.75rem;
}

/* ── Chat / Q&A area ── */
.question-chips { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 1.5rem; }
.chip {
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 20px; padding: 0.4rem 0.9rem;
    font-size: 0.8rem; color: #475569; cursor: pointer;
    transition: all 0.15s;
}
.chip:hover { background: #eef2ff; border-color: #6366f1; color: #6366f1; }

/* ── Response card ── */
.response-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 1.25rem 1.5rem 0.5rem;
    box-shadow: 0 1px 4px rgba(0,0,0,.04);
    margin-top: 1.5rem;
    line-height: 1.7;
    color: #1e293b;
}
.response-meta {
    display: flex; align-items: center; gap: 0.5rem;
    margin-bottom: 1rem; padding-bottom: 1rem;
    border-bottom: 1px solid #f1f5f9;
}
.response-badge {
    background: #eef2ff; color: #6366f1;
    font-size: 0.7rem; font-weight: 600;
    padding: 0.25rem 0.6rem; border-radius: 20px;
    text-transform: uppercase; letter-spacing: 0.05em;
}
.response-model { font-size: 0.75rem; color: #94a3b8; margin-left: auto; }

/* ── Agent log ── */
.log-container {
    background: #0f1117;
    border-radius: 10px;
    padding: 1.25rem;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 0.72rem;
    line-height: 1.6;
    color: #94a3b8;
    max-height: 280px;
    overflow-y: auto;
    margin-top: 0.75rem;
}
.log-agent { color: #818cf8; font-weight: 600; }
.log-action { color: #34d399; }
.log-thought { color: #fb923c; }
.log-result { color: #e2e8f0; }

/* ── Progress bars ── */
.prog-bar-wrap { background: #f1f5f9; border-radius: 99px; height: 6px; margin: 0.4rem 0 1rem; }
.prog-bar { background: linear-gradient(90deg, #6366f1, #8b5cf6); border-radius: 99px; height: 6px; }

/* ── Topic pills ── */
.topic-pill {
    display: inline-block;
    background: #f0fdf4; color: #16a34a;
    border: 1px solid #bbf7d0;
    border-radius: 20px; padding: 0.25rem 0.75rem;
    font-size: 0.78rem; font-weight: 500; margin: 0.2rem;
}

/* ── Strength / weakness rows ── */
.sw-row { display: flex; align-items: center; gap: 0.6rem; padding: 0.5rem 0; border-bottom: 1px solid #f8fafc; font-size: 0.85rem; color: #334155; }
.sw-icon { font-size: 1rem; }

/* ── Buttons ── */
div.stButton > button {
    background: #3b82f6;
    color: white !important;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1.4rem;
    font-weight: 600;
    font-size: 0.85rem;
    letter-spacing: 0.01em;
    transition: background 0.15s, transform 0.1s;
    width: 100%;
}
div.stButton > button:hover { background: #2563eb; transform: translateY(-1px); }
div.stButton > button:active { transform: translateY(0); }

/* ── Text area ── */
.stTextArea textarea {
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    font-size: 0.9rem !important;
    color: #64748b !important;
    background: white !important;
    padding: 0.75rem 1rem !important;
    transition: border-color 0.15s !important;
    resize: none !important;
}
.stTextArea textarea::placeholder { color: #cbd5e1 !important; }
.stTextArea textarea:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,.1) !important;
}

/* ── Alerts ── */
.alert {
    border-radius: 10px; padding: 0.9rem 1.2rem;
    font-size: 0.85rem; display: flex; align-items: flex-start; gap: 0.6rem;
    margin: 1rem 0;
}
.alert-info { background: #eff6ff; border: 1px solid #bfdbfe; color: #1e40af; }
.alert-success { background: #f0fdf4; border: 1px solid #bbf7d0; color: #166534; }
.alert-warning { background: #fffbeb; border: 1px solid #fde68a; color: #92400e; }
.alert-icon { font-size: 1rem; flex-shrink: 0; margin-top: 1px; }

/* ── Divider ── */
.divider { height: 1px; background: #f1f5f9; margin: 1.5rem 0; }

/* ── Sidebar labels ── */
.sidebar-label {
    font-size: 0.65rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.1em; color: #4a5568; margin: 0.25rem 0 0.6rem;
}

/* ── Status dot ── */
.status-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-right: 6px; }
.dot-green { background: #10b981; box-shadow: 0 0 0 2px rgba(16,185,129,.2); }
.dot-red { background: #ef4444; box-shadow: 0 0 0 2px rgba(239,68,68,.2); }

/* ── Score chart ── */
.stVegaLiteChart, .stArrowVegaLiteChart { border-radius: 12px !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    border: 1.5px dashed #cbd5e1 !important;
    border-radius: 10px !important;
    padding: 0.75rem !important;
}

/* ── Expander ── */
details { border: 1px solid #f1f5f9 !important; border-radius: 10px !important; }
details summary { font-size: 0.82rem !important; font-weight: 500 !important; color: #475569 !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: #6366f1 !important; }

/* ── Download button ── */
div.stDownloadButton > button {
    background: white !important;
    color: #6366f1 !important;
    border: 1.5px solid #6366f1 !important;
}
div.stDownloadButton > button:hover { background: #eef2ff !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
STUDENT_DATA_DIR = Path("data/student_data")
DOCUMENTS_DIR    = Path("data/documents")


# ── Utilities ─────────────────────────────────────────────────────────────────

def api_key_ok() -> bool:
    key = os.getenv("GOOGLE_API_KEY", "")
    return bool(key and key != "your-gemini-api-key-here" and len(key) > 10)

def load_student(student_id: str) -> dict:
    path = STUDENT_DATA_DIR / f"{student_id}.json"
    return json.loads(path.read_text()) if path.exists() else {}

def save_pdf(uploaded_file) -> str:
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    dest = DOCUMENTS_DIR / uploaded_file.name
    dest.write_bytes(uploaded_file.getbuffer())
    return str(dest)

def score_color(score: float) -> str:
    if score >= 85: return "#10b981"
    if score >= 70: return "#f59e0b"
    return "#ef4444"

@st.cache_resource(show_spinner=False)
def get_crew():
    from crew import AcademicAssistantCrew
    return AcademicAssistantCrew()


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar() -> tuple[str, str]:
    with st.sidebar:
        # ── Brand ────────────────────────────────────────────────────────────
        st.markdown("""
        <div style="padding:1.25rem 0.25rem 0.75rem; display:flex; align-items:center; gap:0.75rem;">
            <div style="width:40px; height:40px; background:#3b82f6; border-radius:10px; flex-shrink:0;
                        display:flex; align-items:center; justify-content:center; font-size:1.25rem;">🎓</div>
            <div>
                <div style="font-size:1rem; font-weight:700; color:#e2e8f0; letter-spacing:-0.01em;">ILM Assistant</div>
                <div style="font-size:0.7rem; color:#475569; margin-top:0.1rem;">Multi-Agent AI System</div>
            </div>
        </div>
        <hr style="border:none; border-top:1px solid #1e2130; margin:0.5rem 0 1rem;">
        """, unsafe_allow_html=True)

        # ── Student ID ────────────────────────────────────────────────────────
        st.markdown('<div class="sidebar-label">STUDENT ID</div>', unsafe_allow_html=True)
        student_id = st.text_input("", value="student_001", placeholder="student_001", label_visibility="collapsed")

        st.markdown('<hr style="border:none; border-top:1px solid #1e2130; margin:1rem 0;">', unsafe_allow_html=True)

        # ── Mode navigation ───────────────────────────────────────────────────
        st.markdown('<div class="sidebar-label">MODE</div>', unsafe_allow_html=True)

        NAV_ITEMS = {
            "Tutoring":        ("📖", "Ask academic questions"),
            "Admin Support":   ("❓", "Policies & schedules"),
            "Progress Report": ("📊", "Performance analytics"),
        }

        def nav_label(name):
            icon, sub = NAV_ITEMS[name]
            return f"{icon}  {name}"

        mode = st.radio(
            "",
            list(NAV_ITEMS.keys()),
            index=2,
            label_visibility="collapsed",
            format_func=nav_label,
            key="mode_radio",
        )

        st.markdown('<hr style="border:none; border-top:1px solid #1e2130; margin:1rem 0;">', unsafe_allow_html=True)

        # ── Footer ────────────────────────────────────────────────────────────
        online = 3 if api_key_ok() else 0
        dot_color = "#10b981" if api_key_ok() else "#ef4444"
        st.markdown(f"""
        <div style="padding:0.5rem 0.25rem;">
            <span style="width:9px; height:9px; background:{dot_color}; border-radius:50%;
                         display:inline-block; margin-right:7px;
                         box-shadow:0 0 0 2px rgba(16,185,129,.25);"></span>
            <span style="font-size:0.82rem; color:#94a3b8; font-weight:500;">{online} Agents Online</span>
        </div>
        """, unsafe_allow_html=True)

    return mode, student_id


# ── Agent log renderer ────────────────────────────────────────────────────────

def render_log(log: str):
    if not log.strip():
        return
    with st.expander("View agent reasoning", expanded=False):
        lines = []
        for line in log.split("\n"):
            l = line.strip()
            if not l:
                continue
            if "Agent:" in l or "Task:" in l:
                lines.append(f'<span class="log-agent">{l}</span>')
            elif "Action:" in l or "Tool:" in l:
                lines.append(f'<span class="log-action">{l}</span>')
            elif "Thought:" in l:
                lines.append(f'<span class="log-thought">{l}</span>')
            else:
                lines.append(f'<span class="log-result">{l}</span>')
        st.markdown(
            f'<div class="log-container">{"<br>".join(lines[:150])}</div>',
            unsafe_allow_html=True,
        )


# ── MODE: TUTORING ────────────────────────────────────────────────────────────

def tutoring_mode(crew, student_id: str):
    student = load_student(student_id)
    sname = student.get("name", "Student") if student else "Student"
    st.markdown(f"""
    <div class="page-header">
        <div class="page-header-icon">📚</div>
        <div>
            <h1>Tutoring Session</h1>
            <p>{sname} ({student_id})</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if student:
        scores = student.get("scores", [])
        avg = sum(scores) / len(scores) if scores else 0
        comp = student.get("completion_rate", 0)
        topics = student.get("topics_covered", [])
        delta = scores[-1] - scores[-2] if len(scores) >= 2 else 0
        delta_html = f'<span class="metric-delta {"delta-up" if delta >= 0 else "delta-down"}">{"↑" if delta >= 0 else "↓"} {abs(delta):.0f}% vs last</span>'

        st.markdown(f"""
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Average Score</div>
                <div class="metric-value" style="color:{score_color(avg)};">{avg:.1f}%</div>
                {delta_html}
            </div>
            <div class="metric-card">
                <div class="metric-label">Completion Rate</div>
                <div class="metric-value">{comp}%</div>
                <div class="prog-bar-wrap"><div class="prog-bar" style="width:{comp}%"></div></div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Topics Covered</div>
                <div class="metric-value">{len(topics)}</div>
                <span class="metric-delta delta-neutral">{", ".join(topics[:2])}{"…" if len(topics)>2 else ""}</span>
            </div>
            <div class="metric-card">
                <div class="metric-label">Assignments</div>
                <div class="metric-value">{len(scores)}</div>
                <span class="metric-delta delta-neutral">submitted</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Quick questions
    st.markdown('<div class="section-header">Suggested Questions</div>', unsafe_allow_html=True)
    examples = [
        "Explain photosynthesis in simple terms",
        "How do I solve quadratic equations?",
        "What is the difference between mitosis and meiosis?",
        "Explain Newton's laws with real-world examples",
    ]
    cols = st.columns(4)
    for i, ex in enumerate(examples):
        if cols[i].button(ex, key=f"chip_{i}"):
            st.session_state["tutor_q"] = ex

    st.markdown("<div style='height:.75rem'></div>", unsafe_allow_html=True)

    # Question input
    question = st.text_area(
        "Your question",
        value=st.session_state.get("tutor_q", ""),
        placeholder="What would you like to understand today?",
        height=100,
        label_visibility="collapsed",
    )

    col_ask, col_clear = st.columns([5, 1])
    ask = col_ask.button("Ask Tutor →", key="ask_btn")
    if col_clear.button("Clear", key="clear_btn"):
        st.session_state.pop("tutor_q", None)
        st.session_state.pop("tutor_resp", None)
        st.rerun()

    if ask and question.strip():
        _placeholder = st.empty()
        steps = [
            "Analysing your learning profile…",
            "Searching course materials…",
            "Crafting your personalised explanation…",
        ]
        with st.spinner(""):
            for msg in steps:
                _placeholder.markdown(
                    f'<div class="alert alert-info"><span class="alert-icon">⟳</span>{msg}</div>',
                    unsafe_allow_html=True,
                )
                time.sleep(0.7)
            _placeholder.empty()
            try:
                result, log = crew.handle_tutoring_request(student_id, question)
                if result and result.strip():
                    st.session_state["tutor_resp"] = {"q": question, "a": result, "log": log}
                    st.rerun()
                else:
                    st.markdown('<div class="alert alert-warning"><span class="alert-icon">⚠</span>Got an empty response — please try again.</div>', unsafe_allow_html=True)
            except Exception as e:
                st.markdown(
                    f'<div class="alert alert-warning"><span class="alert-icon">⚠</span>Agent error: {e}</div>',
                    unsafe_allow_html=True,
                )

    if resp := st.session_state.get("tutor_resp"):
        st.markdown(f"""
        <div class="response-card">
            <div class="response-meta">
                <span class="response-badge">Tutor Response</span>
                <span style="font-size:.8rem;color:#64748b;font-style:italic;">"{resp['q']}"</span>
                <span class="response-model">gemini-2.5-flash</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(resp['a'])
        render_log(resp.get("log", ""))

        st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
        c1, c2, _ = st.columns([1, 1, 6])
        if c1.button("👍 Helpful", key="fb_yes"):
            st.markdown('<div class="alert alert-success"><span class="alert-icon">✓</span>Glad it helped!</div>', unsafe_allow_html=True)
        if c2.button("👎 Not quite", key="fb_no"):
            st.markdown('<div class="alert alert-info"><span class="alert-icon">ℹ</span>Try rephrasing or uploading relevant course material.</div>', unsafe_allow_html=True)


# ── MODE: ADMIN SUPPORT ───────────────────────────────────────────────────────

def support_mode(crew):
    st.markdown("""
    <div class="page-header">
        <div class="page-header-icon">❓</div>
        <div>
            <h1>Admin Support</h1>
            <p>Policies &amp; schedules</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Quick-access tiles
    st.markdown('<div class="section-header">Common Questions</div>', unsafe_allow_html=True)
    quick = [
        ("📋", "Admission requirements"),
        ("💳", "Tuition & fees"),
        ("📅", "Academic calendar"),
        ("📐", "Grading policies"),
        ("💰", "Financial aid"),
        ("🏥", "Campus services"),
    ]
    cols = st.columns(3)
    for i, (icon, label) in enumerate(quick):
        with cols[i % 3]:
            if st.button(f"{icon}  {label}", key=f"sq_{i}"):
                st.session_state["support_q"] = f"What are the {label.lower()}?"

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    question = st.text_area(
        "",
        value=st.session_state.get("support_q", ""),
        placeholder="Ask anything about admissions, fees, schedules, or academic rules…",
        height=90,
        label_visibility="collapsed",
    )

    if st.button("Ask Support →", key="support_btn") and question.strip():
        with st.spinner(""):
            _p = st.empty()
            _p.markdown('<div class="alert alert-info"><span class="alert-icon">⟳</span>Searching policies…</div>', unsafe_allow_html=True)
            try:
                result, log = crew.handle_support_request(question)
                _p.empty()
                st.session_state["support_resp"] = {"q": question, "a": result, "log": log}
                st.rerun()
            except Exception as e:
                _p.empty()
                st.markdown(f'<div class="alert alert-warning"><span class="alert-icon">⚠</span>{e}</div>', unsafe_allow_html=True)

    if resp := st.session_state.get("support_resp"):
        st.markdown(f"""
        <div class="response-card">
            <div class="response-meta">
                <span class="response-badge" style="background:#fef3c7;color:#92400e;">Support Response</span>
                <span style="font-size:.8rem;color:#64748b;font-style:italic;">"{resp['q']}"</span>
                <span class="response-model">gemini-2.5-flash</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(resp['a'])
        render_log(resp.get("log", ""))
        st.markdown("""
        <div class="alert alert-info" style="margin-top:1rem;">
            <span class="alert-icon">ℹ</span>
            Need further help? Visit Student Services or call <strong>(555) 123-4567</strong>
        </div>
        """, unsafe_allow_html=True)


# ── MODE: PROGRESS REPORT ─────────────────────────────────────────────────────

def progress_mode(crew, student_id: str):
    student = load_student(student_id)
    student_name = student.get("name", "Student") if student else "Student"

    hdr_left, hdr_right = st.columns([3, 1])
    with hdr_left:
        st.markdown(f"""
        <div class="page-header" style="border-bottom:none; padding-bottom:0; margin-bottom:0;">
            <div class="page-header-icon">📊</div>
            <div>
                <h1>Progress Report</h1>
                <p>{student_name} ({student_id})</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with hdr_right:
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        generate_header = st.button("📄  Generate Report", key="gen_report_header", use_container_width=True)

    st.markdown('<hr style="border:none; border-top:1px solid #f1f5f9; margin:1rem 0 2rem;">', unsafe_allow_html=True)
    if not student:
        st.markdown('<div class="alert alert-warning"><span class="alert-icon">⚠</span>No profile found for this student ID.</div>', unsafe_allow_html=True)
        student = {"scores": [], "completion_rate": 0, "topics_covered": [], "strengths": [], "weaknesses": []}

    scores     = student.get("scores", [])
    avg        = sum(scores) / len(scores) if scores else 0
    comp       = student.get("completion_rate", 0)
    topics     = student.get("topics_covered", [])
    strengths  = student.get("strengths", [])
    weaknesses = student.get("weaknesses", [])
    delta      = scores[-1] - scores[-2] if len(scores) >= 2 else 0

    # ── Metrics ───────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-label">Average Score</div>
            <div class="metric-value" style="color:{score_color(avg)};">{avg:.1f}%</div>
            <span class="metric-delta {"delta-up" if delta>=0 else "delta-down"}">{"↑" if delta>=0 else "↓"} {abs(delta):.0f}% trend</span>
        </div>
        <div class="metric-card">
            <div class="metric-label">Completion Rate</div>
            <div class="metric-value">{comp}%</div>
            <div class="prog-bar-wrap"><div class="prog-bar" style="width:{comp}%"></div></div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Topics Covered</div>
            <div class="metric-value">{len(topics)}</div>
            <span class="metric-delta delta-neutral">{len(scores)} assignments</span>
        </div>
        <div class="metric-card">
            <div class="metric-label">Standing</div>
            <div class="metric-value" style="font-size:1.1rem;padding-top:.3rem;">{"Excelling" if avg>=85 else "On Track" if avg>=70 else "At Risk"}</div>
            <span class="metric-delta {"delta-up" if avg>=85 else "delta-neutral" if avg>=70 else "delta-down"}">{"●" * min(5, int(avg/20))} {"●" * max(0, 5-int(avg/20))}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Two-column layout ─────────────────────────────────────────────────────
    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown('<div class="section-header">Score History</div>', unsafe_allow_html=True)
        if scores:
            df = pd.DataFrame({"Score": scores}, index=[f"#{i+1}" for i in range(len(scores))])
            st.line_chart(df, use_container_width=True, height=200, color="#6366f1")
        else:
            st.markdown('<div style="height:200px;display:flex;align-items:center;justify-content:center;color:#94a3b8;font-size:.85rem;">No scores recorded yet</div>', unsafe_allow_html=True)

        st.markdown('<div style="height:.5rem"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Topics</div>', unsafe_allow_html=True)
        if topics:
            pills = "".join(f'<span class="topic-pill">{t}</span>' for t in topics)
            st.markdown(pills, unsafe_allow_html=True)
        else:
            st.markdown('<span style="color:#94a3b8;font-size:.85rem;">No topics recorded</span>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-header">Strengths</div>', unsafe_allow_html=True)
        if strengths:
            rows = "".join(f'<div class="sw-row"><span class="sw-icon">✦</span>{s}</div>' for s in strengths)
            st.markdown(f'<div style="background:white;border:1px solid #f1f5f9;border-radius:12px;padding:.75rem 1rem;">{rows}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<span style="color:#94a3b8;font-size:.85rem;">None recorded</span>', unsafe_allow_html=True)

        st.markdown('<div style="height:.75rem"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Areas to Improve</div>', unsafe_allow_html=True)
        if weaknesses:
            rows = "".join(f'<div class="sw-row"><span class="sw-icon" style="color:#f59e0b;">◆</span>{w}</div>' for w in weaknesses)
            st.markdown(f'<div style="background:white;border:1px solid #f1f5f9;border-radius:12px;padding:.75rem 1rem;">{rows}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<span style="color:#94a3b8;font-size:.85rem;">None identified</span>', unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Generate report (triggered from header button) ────────────────────────
    _, ccol = st.columns([5, 1])
    if ccol.button("Clear", key="clear_report"):
        st.session_state.pop("progress_resp", None)
        st.rerun()

    if generate_header:
        bar = st.progress(0)
        status = st.empty()
        steps = [
            (20, "Retrieving performance history…"),
            (45, "Identifying patterns and trends…"),
            (70, "Building personalised learning plan…"),
            (90, "Finalising your report…"),
        ]
        with st.spinner(""):
            try:
                for pct, msg in steps:
                    bar.progress(pct)
                    status.markdown(f'<div class="alert alert-info"><span class="alert-icon">⟳</span>{msg}</div>', unsafe_allow_html=True)
                    time.sleep(0.6)
                result, log = crew.generate_progress_report(student_id)
                bar.progress(100)
                bar.empty(); status.empty()
                st.session_state["progress_resp"] = {"result": result, "log": log, "sid": student_id}
                st.rerun()
            except Exception as e:
                bar.empty(); status.empty()
                st.markdown(f'<div class="alert alert-warning"><span class="alert-icon">⚠</span>{e}</div>', unsafe_allow_html=True)

    if resp := st.session_state.get("progress_resp"):
        st.markdown(f"""
        <div class="response-card">
            <div class="response-meta">
                <span class="response-badge" style="background:#f0fdf4;color:#166534;">AI Progress Report</span>
                <span style="font-size:.78rem;color:#94a3b8;">{resp['sid']}</span>
                <span class="response-model">gemini-2.5-flash</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(resp['result'])
        render_log(resp.get("log", ""))
        st.markdown("<div style='height:.75rem'></div>", unsafe_allow_html=True)
        st.download_button(
            "Download Report",
            data=resp["result"],
            file_name=f"report_{resp['sid']}.txt",
            mime="text/plain",
        )


# ── No-key preview ────────────────────────────────────────────────────────────

def demo_preview(student_id: str):
    student = load_student(student_id) or {
        "scores": [85, 90, 78, 92, 88],
        "completion_rate": 85,
        "topics_covered": ["Algebra", "Geometry", "Biology", "Chemistry"],
        "strengths": ["Problem solving", "Critical thinking"],
        "weaknesses": ["Time management", "Essay writing"],
    }
    scores = student["scores"]
    avg = sum(scores) / len(scores)
    comp = student["completion_rate"]

    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-label">Average Score</div>
            <div class="metric-value" style="color:{score_color(avg)};">{avg:.1f}%</div>
            <span class="metric-delta delta-up">↑ 4% trend</span>
        </div>
        <div class="metric-card">
            <div class="metric-label">Completion Rate</div>
            <div class="metric-value">{comp}%</div>
            <div class="prog-bar-wrap"><div class="prog-bar" style="width:{comp}%"></div></div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Topics Covered</div>
            <div class="metric-value">{len(student['topics_covered'])}</div>
            <span class="metric-delta delta-up">↑ 1 new</span>
        </div>
        <div class="metric-card">
            <div class="metric-label">Standing</div>
            <div class="metric-value" style="font-size:1.1rem;padding-top:.3rem;">On Track</div>
            <span class="metric-delta delta-neutral">●●●●○</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    df = pd.DataFrame({"Score": scores}, index=[f"#{i+1}" for i in range(len(scores))])
    st.line_chart(df, use_container_width=True, height=180, color="#6366f1")

    pills = "".join(f'<span class="topic-pill">{t}</span>' for t in student["topics_covered"])
    st.markdown(pills, unsafe_allow_html=True)
    st.markdown("""
    <div class="alert alert-info" style="margin-top:1.5rem;">
        <span class="alert-icon">ℹ</span>
        Add your <strong>GOOGLE_API_KEY</strong> to <code>.env</code> to activate the AI agents.
    </div>
    """, unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    for k in ("tutor_q", "support_q"):
        st.session_state.setdefault(k, "")

    mode, student_id = render_sidebar()

    if not api_key_ok():
        st.markdown("""
        <div class="page-header">
            <div class="page-header-icon">🎓</div>
            <div>
                <h1>ILM Assistant</h1>
                <p>Configure your API key to activate the multi-agent system</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        demo_preview(student_id)
        return

    try:
        crew = get_crew()
    except Exception as e:
        st.markdown(f'<div class="alert alert-warning"><span class="alert-icon">⚠</span>Failed to initialise agents: {e}</div>', unsafe_allow_html=True)
        return

    if mode == "Tutoring":
        tutoring_mode(crew, student_id)
    elif mode == "Admin Support":
        support_mode(crew)
    else:
        progress_mode(crew, student_id)


if __name__ == "__main__":
    main()
