import streamlit as st
import os
import json
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List
from dotenv import load_dotenv
from utils.stt import transcribe_audio
from utils.intent import classify_intent
from utils.memory import get_relevant_context, save_interaction, get_learned_facts
from tools.executor import execute_intent

load_dotenv()

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VoiceAgent",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

:root {
  --bg: #0a0a0f;
  --surface: #12121a;
  --border: #1e1e2e;
  --accent: #7c3aed;
  --accent2: #06b6d4;
  --accent3: #f59e0b;
  --text: #e2e8f0;
  --muted: #64748b;
  --success: #10b981;
  --error: #ef4444;
}

* { box-sizing: border-box; }

html, body, .stApp {
  background-color: var(--bg) !important;
  color: var(--text) !important;
  font-family: 'Syne', sans-serif;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem !important; max-width: 1200px !important; }

/* Hero header */
.hero-header {
  text-align: center;
  padding: 2rem 0 1rem;
  position: relative;
}
.hero-header h1 {
  font-family: 'Syne', sans-serif;
  font-weight: 800;
  font-size: 3.2rem;
  background: linear-gradient(135deg, #7c3aed, #06b6d4, #f59e0b);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin: 0;
  letter-spacing: -2px;
}
.hero-sub {
  color: var(--muted);
  font-family: 'Space Mono', monospace;
  font-size: 0.8rem;
  margin-top: 0.5rem;
  letter-spacing: 3px;
  text-transform: uppercase;
}

/* Cards */
.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.5rem;
  margin-bottom: 1rem;
  position: relative;
  overflow: hidden;
}
.card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, var(--accent), var(--accent2));
}
.card-label {
  font-family: 'Space Mono', monospace;
  font-size: 0.65rem;
  color: var(--muted);
  letter-spacing: 3px;
  text-transform: uppercase;
  margin-bottom: 0.75rem;
}
.card-content {
  font-size: 1rem;
  color: var(--text);
  line-height: 1.6;
}

/* Intent badge */
.intent-badge {
  display: inline-block;
  padding: 0.3rem 1rem;
  border-radius: 999px;
  font-family: 'Space Mono', monospace;
  font-size: 0.75rem;
  font-weight: 700;
  letter-spacing: 1px;
  text-transform: uppercase;
}
.intent-create_file    { background: rgba(124,58,237,0.2); border: 1px solid #7c3aed; color: #a78bfa; }
.intent-write_code     { background: rgba(6,182,212,0.2);  border: 1px solid #06b6d4; color: #67e8f9; }
.intent-summarize_text { background: rgba(245,158,11,0.2); border: 1px solid #f59e0b; color: #fcd34d; }
.intent-general_chat   { background: rgba(16,185,129,0.2); border: 1px solid #10b981; color: #6ee7b7; }

/* Status */
.status-ok  { color: var(--success); font-family: 'Space Mono', monospace; font-size: 0.8rem; }
.status-err { color: var(--error);   font-family: 'Space Mono', monospace; font-size: 0.8rem; }

/* Timeline step */
.step {
  display: flex;
  align-items: flex-start;
  gap: 1rem;
  padding: 0.75rem 0;
  border-bottom: 1px solid var(--border);
}
.step:last-child { border-bottom: none; }
.step-num {
  width: 28px; height: 28px;
  border-radius: 50%;
  background: linear-gradient(135deg, var(--accent), var(--accent2));
  display: flex; align-items: center; justify-content: center;
  font-family: 'Space Mono', monospace;
  font-size: 0.7rem;
  font-weight: 700;
  flex-shrink: 0;
  color: white;
}
.step-done { background: var(--success) !important; }
.step-body { flex: 1; }
.step-title { font-weight: 600; font-size: 0.9rem; color: var(--text); }
.step-detail { font-family: 'Space Mono', monospace; font-size: 0.75rem; color: var(--muted); margin-top: 0.2rem; }

/* History item */
.hist-item {
  padding: 0.75rem;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  margin-bottom: 0.5rem;
  cursor: pointer;
  transition: border-color 0.2s;
}
.hist-item:hover { border-color: var(--accent); }
.hist-time { font-family: 'Space Mono', monospace; font-size: 0.65rem; color: var(--muted); }
.hist-text { font-size: 0.85rem; margin-top: 0.3rem; }

/* Streamlit overrides */
.stButton > button {
  background: linear-gradient(135deg, var(--accent), #6d28d9) !important;
  color: white !important;
  border: none !important;
  border-radius: 8px !important;
  font-family: 'Space Mono', monospace !important;
  font-size: 0.8rem !important;
  letter-spacing: 1px !important;
  padding: 0.6rem 1.5rem !important;
  width: 100%;
  transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }
.stFileUploader, .stAudioInput {
  background: var(--surface) !important;
  border: 1px dashed var(--border) !important;
  border-radius: 10px !important;
  padding: 1rem !important;
}
.stTextArea textarea {
  background: var(--surface) !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  font-family: 'Space Mono', monospace !important;
  font-size: 0.85rem !important;
}
.stSelectbox select, .stSelectbox > div > div {
  background: var(--surface) !important;
  color: var(--text) !important;
  border-color: var(--border) !important;
}
div[data-testid="stExpander"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
}

/* Divider */
.divider {
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--border), transparent);
  margin: 1.5rem 0;
}

/* Code block */
pre {
  background: #0d0d14 !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  padding: 1rem !important;
  font-family: 'Space Mono', monospace !important;
  font-size: 0.8rem !important;
  overflow-x: auto;
  color: #e2e8f0 !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Session State ──────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "pending_confirmation" not in st.session_state:
    st.session_state.pending_confirmation = None
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "action_notice" not in st.session_state:
  st.session_state.action_notice = None
if "audio_bytes" not in st.session_state:
  st.session_state.audio_bytes = None
if "audio_suffix" not in st.session_state:
  st.session_state.audio_suffix = ".wav"
if "memory_panel_items" not in st.session_state:
    st.session_state.memory_panel_items = []
if "summarize_btn" not in st.session_state:
    st.session_state.summarize_btn = False
if "context_text" not in st.session_state:
    st.session_state.context_text = ""

# ─── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
  <h1>🎙️ VoiceAgent</h1>
  <p class="hero-sub">Speak → Transcribe → Understand → Execute</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

if st.session_state.action_notice:
  notice = st.session_state.action_notice
  if notice.get("type") == "success":
    st.success(notice.get("text", "Action completed."))
  elif notice.get("type") == "error":
    st.error(notice.get("text", "Action failed."))
  else:
    st.info(notice.get("text", "Working..."))

with st.sidebar:
  st.markdown("### 🧠 Memory")
  panel_items = get_learned_facts(limit=8)
  if panel_items:
    st.session_state.memory_panel_items = panel_items

  if st.session_state.memory_panel_items:
    for item in st.session_state.memory_panel_items:
      st.markdown(f"**{item.get('created_at','')}**")
      st.markdown(f"- {item.get('memory','')}")
  else:
    st.caption("No learned preferences yet. Run a few commands to build memory.")

# ─── Layout ─────────────────────────────────────────────────────────────────
left, right = st.columns([3, 2], gap="large")

with left:
    st.markdown("### 🎤 Input")

    tab1, tab2, tab3 = st.tabs(["Upload Audio", "Live Record", "Text Demo"])

    audio_bytes = None
    demo_text = None

    with tab1:
      uploaded = st.file_uploader(
        "Drop a .wav or .mp3 file",
        type=["wav", "mp3", "m4a", "ogg"],
        label_visibility="collapsed"
      )
      if uploaded:
        st.session_state.audio_bytes = uploaded.read()
        suffix = Path(uploaded.name).suffix.lower() if getattr(uploaded, "name", None) else ".wav"
        st.session_state.audio_suffix = suffix if suffix in {".wav", ".mp3", ".m4a", ".ogg"} else ".wav"
      if st.session_state.audio_bytes:
        audio_bytes = st.session_state.audio_bytes
        st.audio(audio_bytes)

    with tab2:
      st.markdown('<p style="color:var(--muted);font-size:0.8rem;">Record directly from your microphone in the browser.</p>', unsafe_allow_html=True)
      recorded = st.audio_input("Tap to record", label_visibility="collapsed")
      if recorded:
        st.session_state.audio_bytes = recorded.getvalue()
        st.session_state.audio_suffix = ".wav"
      if st.session_state.audio_bytes:
        audio_bytes = st.session_state.audio_bytes
        st.audio(audio_bytes)

    with tab3:
        st.markdown('<p style="color:var(--muted);font-size:0.8rem;">No mic on this machine? Type a command to simulate:</p>', unsafe_allow_html=True)
        demo_text = st.text_area(
            "Type command",
            placeholder='e.g. "Create a Python file with a bubble sort function"',
            height=100,
            label_visibility="collapsed"
        )

    if demo_text:
        demo_text = demo_text.strip()

    col_clear, _ = st.columns([1, 2])
    with col_clear:
      if st.button(" Clear audio buffer", use_container_width=False):
        st.session_state.audio_bytes = None
        st.session_state.audio_suffix = ".wav"
        audio_bytes = None
        st.rerun()

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Context for summarize intent
    with st.expander("📄 Provide text to summarize"):
        context_input = st.text_area(
            "Text to summarize",
            placeholder="Paste long text here if your command involves summarization...",
            height=120,
            label_visibility="collapsed"
        )
        st.session_state.context_text = context_input
    
    # Filename override
    with st.expander("⚙️ Advanced options"):
        custom_filename = st.text_input("Custom filename (optional)", placeholder="e.g. my_script.py")
        model_choice = st.selectbox("LLM Backend", ["ollama (llama3.2)", "ollama (mistral)", "ollama (codellama)"])

    col_run, col_summarize = st.columns([2, 1])
    with col_run:
      run_btn = st.button("⚡ RUN AGENT", use_container_width=True)
    with col_summarize:
      summarize_btn = st.button("📝 SUMMARIZE", use_container_width=True)

    # ── Human-in-the-loop confirmation ──────────────────────────────────────
    if st.session_state.pending_confirmation:
        p = st.session_state.pending_confirmation
        st.markdown(f"""
        <div class="card" style="border-color:#f59e0b;">
          <div class="card-label">⚠️ Confirmation Required</div>
          <div class="card-content">
            About to <b>{p['action']}</b><br>
            <code style="color:#fcd34d;">{p['detail']}</code>
          </div>
        </div>
        """, unsafe_allow_html=True)
        col_yes, col_no = st.columns(2)
        with col_yes:
          if st.button("✅ Confirm", use_container_width=True):
            st.session_state.action_notice = {"type": "info", "text": f"Executing {p['action']}..."}
            with st.spinner(f"Running {p['action']}..."):
              result = execute_intent(
                p["intent_data"],
                p["context"],
                confirmed=True,
                custom_filename=custom_filename or None,
                model=p.get("llm_model")
              )
            save_ok, _ = save_interaction(
              p["transcription"],
              p["intent_data"]["primary_intent"],
              (result.get("output") or result.get("error") or "")[:200],
            )
            if save_ok:
              st.session_state.memory_panel_items = get_learned_facts(limit=8)
            st.session_state.last_result = result
            st.session_state.history.insert(0, {
              "time": time.strftime("%H:%M:%S"),
              "text": p["transcription"],
              "intent": p["intent_data"]["primary_intent"],
              "result": result,
            })
            if result.get("success"):
                st.session_state.action_notice = {"type": "success", "text": result.get("action_taken", "Action completed.")}
            else:
                st.session_state.action_notice = {"type": "error", "text": result.get("error", "Action failed.")}
            st.session_state.pending_confirmation = None
            st.rerun()
        with col_no:
            if st.button("❌ Cancel", use_container_width=True):
                st.session_state.pending_confirmation = None
                st.rerun()

with right:
    st.markdown("### 📋 Session History")
    if not st.session_state.history:
        st.markdown('<p style="color:var(--muted);font-size:0.85rem;">No commands yet. Run your first command →</p>', unsafe_allow_html=True)
    for i, item in enumerate(st.session_state.history[:10]):
        intent_cls = f"intent-{item['intent']}"
        st.markdown(f"""
        <div class="hist-item">
          <div class="hist-time">{item['time']}</div>
          <div class="hist-text">{item['text'][:80]}{"..." if len(item['text'])>80 else ""}</div>
          <span class="intent-badge {intent_cls}" style="margin-top:0.5rem;display:inline-block;">{item['intent'].replace('_',' ')}</span>
        </div>
        """, unsafe_allow_html=True)

# ─── Quick Summarize Action ────────────────────────────────────────────────────
if summarize_btn:
    text_to_summarize = st.session_state.context_text.strip()
    if not text_to_summarize:
        st.warning("Please provide text to summarize in the 📄 Summarize section.")
    else:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("### ⚙️ Pipeline Execution")

        steps_placeholder = st.empty()
        result_placeholder = st.empty()

        steps = [
            {"title": "Prepare Text",       "detail": "Ready to summarize...",     "done": False},
            {"title": "Summarization",      "detail": "Generating summary...",     "done": False},
            {"title": "Output Ready",       "detail": "Formatting results...",     "done": False},
        ]

        def render_steps_summarize(steps: List[Dict[str, Any]]) -> None:
            """Render summarize pipeline steps.

            Args:
                steps: Ordered list of step dictionaries with title/detail/done keys.
            """
            html = '<div class="card">'
            for i, s in enumerate(steps):
                cls = "step-done" if s["done"] else ""
                icon = "✓" if s["done"] else str(i+1)
                html += f"""
                <div class="step">
                  <div class="step-num {cls}">{icon}</div>
                  <div class="step-body">
                    <div class="step-title">{s['title']}</div>
                    <div class="step-detail">{s['detail']}</div>
                  </div>
                </div>"""
            html += '</div>'
            steps_placeholder.markdown(html, unsafe_allow_html=True)

        render_steps_summarize(steps)

        # Prepare text
        steps[0]["detail"] = f"✓ {len(text_to_summarize)} characters"
        steps[0]["done"] = True
        render_steps_summarize(steps)

        # Execute summarization
        model_name = model_choice.split("(")[1].rstrip(")")
        intent_data = {
            "primary_intent": "summarize_text",
            "sub_intents": [],
            "confidence": "direct",
            "raw_text": f"Summarize this: {text_to_summarize[:100]}...",
            "description": "Direct summarization request"
        }
        
        result = execute_intent(
            intent_data,
            text_to_summarize,
            confirmed=True,
            custom_filename=custom_filename or None,
            model=model_name,
        )
        
        save_ok, _ = save_interaction(
            f"Summarize: {text_to_summarize[:80]}",
            "summarize_text",
            (result.get("output") or result.get("error") or "")[:200],
        )
        if save_ok:
            st.session_state.memory_panel_items = get_learned_facts(limit=8)
        
        steps[1]["detail"] = f'✓ {result.get("action_taken","done")}'
        steps[1]["done"] = True
        steps[2]["done"] = True
        steps[2]["detail"] = "✓ Results ready below"
        render_steps_summarize(steps)
        
        st.session_state.last_result = result
        st.session_state.history.insert(0, {
            "time": time.strftime("%H:%M:%S"),
            "text": f"Summarize: {text_to_summarize[:60]}",
            "intent": "summarize_text",
            "result": result,
        })
        st.rerun()

# ─── Agent Execution ────────────────────────────────────────────────────────
if run_btn:
    text_input = (demo_text or "").strip()
    use_text_input = bool(text_input)
    selected_audio = audio_bytes if not use_text_input else None

    if not selected_audio and not use_text_input:
        st.warning("Please upload an audio file or type a command.")
    else:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("### ⚙️ Pipeline Execution")

        steps_placeholder = st.empty()
        result_placeholder = st.empty()

        steps = [
            {"title": "Audio → Text",       "detail": "Transcribing audio...",     "done": False},
            {"title": "Intent Analysis",    "detail": "Classifying intent...",     "done": False},
            {"title": "Tool Execution",     "detail": "Running action...",         "done": False},
            {"title": "Output Ready",       "detail": "Formatting results...",     "done": False},
        ]

        def render_steps(steps: List[Dict[str, Any]]) -> None:
            """Render the main execution pipeline steps.

            Args:
                steps: Ordered list of step dictionaries with title/detail/done keys.
            """
            html = '<div class="card">'
            for i, s in enumerate(steps):
                cls = "step-done" if s["done"] else ""
                icon = "✓" if s["done"] else str(i+1)
                html += f"""
                <div class="step">
                  <div class="step-num {cls}">{icon}</div>
                  <div class="step-body">
                    <div class="step-title">{s['title']}</div>
                    <div class="step-detail">{s['detail']}</div>
                  </div>
                </div>"""
            html += '</div>'
            steps_placeholder.markdown(html, unsafe_allow_html=True)

        render_steps(steps)

        # Step 1: Transcription
        with st.spinner(""):
          if use_text_input:
            transcription = text_input
            steps[0]["detail"] = f"(demo mode) → \"{transcription[:60]}...\""
          else:
            audio_suffix = st.session_state.get("audio_suffix", ".wav") or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=audio_suffix) as f:
              f.write(selected_audio)
              tmp_path = f.name
            transcription, stt_error = transcribe_audio(tmp_path)
            os.unlink(tmp_path)
            if stt_error:
              st.error(f"STT Error: {stt_error}")
              st.stop()
            steps[0]["detail"] = f'✓ "{transcription[:80]}"'
          steps[0]["done"] = True
          render_steps(steps)

        # Step 2: Intent classification
        model_name = model_choice.split("(")[1].rstrip(")")
        memory_context = get_relevant_context(transcription)
        intent_data, intent_error = classify_intent(transcription, model_name, memory_context=memory_context)
        if intent_error:
            st.error(f"Intent Error: {intent_error}")
            st.stop()
        steps[1]["detail"] = f'✓ {intent_data["primary_intent"]} (conf: {intent_data.get("confidence","?")}) | sub-intents: {", ".join(intent_data.get("sub_intents",[]))}'
        steps[1]["done"] = True
        render_steps(steps)

        # Step 3: Human-in-the-loop for file ops
        context = st.session_state.context_text
        if memory_context:
          context = f"{memory_context}\n\n{context}".strip()
        file_ops = {"create_file", "write_code"}
        if intent_data["primary_intent"] in file_ops:
            filename = custom_filename or intent_data.get("suggested_filename", "output_file.txt")
            steps[2]["detail"] = f"⏸ Awaiting confirmation for: {filename}"
            render_steps(steps)
            st.session_state.pending_confirmation = {
                "intent_data": intent_data,
                "context": context,
                "action": intent_data["primary_intent"],
                "detail": f"output/{filename}",
                "transcription": transcription,
              "llm_model": model_name,
            }
            st.rerun()
        else:
            result = execute_intent(
              intent_data,
              context,
              confirmed=True,
              custom_filename=custom_filename or None,
              model=model_name,
            )
            save_ok, _ = save_interaction(
                transcription,
                intent_data["primary_intent"],
                (result.get("output") or result.get("error") or "")[:200],
            )
            if save_ok:
                st.session_state.memory_panel_items = get_learned_facts(limit=8)
            steps[2]["detail"] = f'✓ {result.get("action_taken","done")}'
            steps[2]["done"] = True
            steps[3]["done"] = True
            steps[3]["detail"] = "✓ Results ready below"
            render_steps(steps)
            st.session_state.last_result = result
            st.session_state.history.insert(0, {
                "time": time.strftime("%H:%M:%S"),
                "text": transcription,
                "intent": intent_data["primary_intent"],
                "result": result,
            })
            st.rerun()

# ─── Result Display ──────────────────────────────────────────────────────────
if st.session_state.last_result:
    r = st.session_state.last_result
    it = r.get("intent","")
    intent_cls = f"intent-{it}"

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("### 📊 Results")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="card">
          <div class="card-label">Transcription</div>
          <div class="card-content">"{r.get('transcription', r.get('input',''))}"</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="card">
          <div class="card-label">Detected Intent</div>
          <div class="card-content">
            <span class="intent-badge {intent_cls}">{it.replace('_',' ')}</span>
            <br><br>
            <span style="color:var(--muted);font-size:0.8rem;font-family:'Space Mono',monospace;">
              Confidence: {r.get('confidence','N/A')} | Sub-intents: {", ".join(r.get('sub_intents',[]))}
            </span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    if r.get("success"):
        st.markdown(f'<p class="status-ok">✓ {r.get("action_taken","")}</p>', unsafe_allow_html=True)
    else:
        st.markdown(f'<p class="status-err">✗ {r.get("error","Unknown error")}</p>', unsafe_allow_html=True)

    if r.get("output"):
        st.markdown(f"""
        <div class="card">
          <div class="card-label">Output</div>
          <div class="card-content"><pre>{r['output'][:3000]}</pre></div>
        </div>
        """, unsafe_allow_html=True)

    if r.get("file_path"):
        st.markdown(f"""
        <div class="card">
          <div class="card-label">File Created</div>
          <div class="card-content" style="font-family:'Space Mono',monospace;color:#a78bfa;">
            📁 {r['file_path']}
          </div>
        </div>
        """, unsafe_allow_html=True)
