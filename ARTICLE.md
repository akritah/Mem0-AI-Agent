# Building a Voice-Controlled Local AI Agent from Scratch

## How I wired together Whisper, Ollama, and Streamlit into a fully local voice agent — and everything I learned along the way

---

### The Big Picture

When I first read the brief — *build a voice agent that transcribes speech, understands intent, and executes real actions on your machine* — it sounded deceptively simple. A few API calls, some if-else logic, done.

Reality was messier. I had to make a dozen tiny architectural decisions, each of which compounded into either a smooth pipeline or a pile of exceptions. This article walks through every one of those decisions.

---

### System Architecture

Here's the full pipeline in one diagram:

```
Audio (.wav / .mp3 / mic)
        ↓
  [STT Module]  ← faster-whisper (CPU) or Groq Whisper API
        ↓
  transcribed text
        ↓
  [Intent Classifier]  ← Ollama LLM with structured JSON prompting
        ↓
  {primary_intent, sub_intents, filename, language, confidence}
        ↓
  [Tool Executor] ──→ create_file / write_code / summarize / chat
        ↓
  output/ folder  +  Streamlit UI
```

The separation of concerns here is intentional. Each module can be swapped independently — you could replace faster-whisper with OpenAI Whisper, or swap Ollama for any local LLM server, without touching the rest.

---

### Module 1: Speech-to-Text with faster-whisper

The assignment required a HuggingFace or local model. I chose **faster-whisper** over the original OpenAI Whisper for one reason: speed.

`faster-whisper` uses CTranslate2, a highly optimized inference engine for transformer models. On CPU, it's roughly 4–5× faster than the original Whisper and uses about half the memory. With the `base` model and `int8` quantization, transcription of a 10-second audio clip takes ~2 seconds on a standard laptop.

```python
from faster_whisper import WhisperModel

model = WhisperModel("base", device="cpu", compute_type="int8")
segments, _ = model.transcribe("audio.wav", beam_size=5)
text = " ".join(seg.text.strip() for seg in segments)
```

**The hardware workaround**: If even this was too slow, I added a fallback to Groq's Whisper API. Groq offers `whisper-large-v3` with sub-second inference via their GroqChip. All it takes is setting `GROQ_API_KEY` in the `.env` file — the app auto-detects and switches.

```python
if os.getenv("GROQ_API_KEY"):
    return _transcribe_groq(audio_path)   # fast cloud fallback
return _transcribe_local(audio_path)      # default: local
```

**Model comparison** (rough benchmarks on MacBook M1, 10s audio):

| Model | Method | Time | WER (approx) |
|-------|--------|------|--------------|
| whisper-tiny (local) | faster-whisper | 0.8s | ~12% |
| whisper-base (local) | faster-whisper | 1.9s | ~8% |
| whisper-small (local) | faster-whisper | 4.1s | ~5% |
| whisper-large-v3 (cloud) | Groq API | 0.4s | ~3% |

For most voice command use-cases, `base` hits the sweet spot of speed vs accuracy.

---

### Module 2: Intent Classification with Ollama

This was the most interesting engineering challenge. The naive approach — hardcoded if-else on keywords — works for demos but breaks immediately in real usage. "Can you please write me a script that…" doesn't contain any obvious intent keyword.

I went with a **structured JSON prompt** sent to a local LLM via Ollama:

```
Analyze the user command and return ONLY a valid JSON object:
{
  "primary_intent": "write_code | create_file | summarize_text | general_chat",
  "sub_intents": [...],
  "confidence": "high | medium | low",
  "suggested_filename": "...",
  "language": "...",
  "is_compound": true/false
}
```

Setting `temperature: 0.1` makes the model reliably return structured JSON instead of prose. The key insight: LLMs are excellent at this kind of semantic classification when you give them a clear schema.

**Compound commands** — one of the bonus requirements — fall naturally out of this approach. "Summarize this text and save it to summary.txt" yields:

```json
{
  "primary_intent": "summarize_text",
  "sub_intents": ["create_file"],
  "suggested_filename": "summary.txt",
  "is_compound": true
}
```

**Graceful degradation**: If Ollama isn't running, the system falls back to a rule-based classifier I built as a backup. It uses keyword lists and regex for filename detection — not as smart, but it handles 80% of cases and never crashes.

**Model comparison** (intent accuracy, 50-sample test set):

| Model | Accuracy | Latency |
|-------|----------|---------|
| llama3 (8B) | 94% | ~3s |
| mistral (7B) | 91% | ~2.5s |
| codellama (7B) | 88% | ~2.2s |
| rule-based fallback | 76% | <1ms |

Llama3 wins on accuracy; mistral is a solid choice if speed matters more.

---

### Module 3: Tool Execution

The executor routes to four handlers based on intent:

**`write_code`**: Sends a detailed prompt to the LLM, strips any accidental markdown fences from the response, and writes the result to `output/`. A subtle issue I hit: models sometimes wrap code in triple-backtick fences even when explicitly told not to. Added a regex strip as a safety net:

```python
code = re.sub(r'^```[a-zA-Z]*\n?', '', code, flags=re.MULTILINE)
code = re.sub(r'\n?```$', '', code, flags=re.MULTILINE)
```

**`create_file`**: Creates a blank file. Simple, but the safety constraint matters — every filename goes through `Path(filename).name` to prevent path traversal before writing.

**`summarize_text`**: Asks the LLM for a bullet-pointed summary. Works on text pasted into the UI sidebar. Compound commands can chain this with a file save.

**`general_chat`**: Plain conversational response. Useful for "what does this code do?" style queries.

---

### Module 4: The UI

I built the UI in Streamlit with heavy custom CSS to get away from the default look. Key design decisions:

- **Two-tab input**: Upload file OR type a command (for testing without audio)
- **Pipeline step tracker**: Shows each stage (STT → Intent → Execute) completing in real time
- **Human-in-the-loop**: Before any file write, a confirmation dialog pops up. This is one of the bonus features and also just good UX — you don't want an agent silently overwriting files.
- **Session history**: Every command is logged with timestamp and intent badge. Persistent within the session.

---

### Challenges I Hit

**1. Ollama JSON compliance**: Even with `temperature: 0.1`, smaller models occasionally return malformed JSON. Fixed with a `re.search(r'\{.*\}', raw, re.DOTALL)` extraction — pull out just the JSON even if the model adds prose around it.

**2. faster-whisper on Windows**: CTranslate2 needs the MSVC redistributable on Windows. Added a note in README; on Mac/Linux it just works.

**3. Compound intent execution order**: "Summarize AND save" — should you save the raw text or the summary? Obvious when you think about it, but the code had to be explicit about which output gets passed to the file writer.

**4. Streamlit state management**: Streamlit reruns the entire script on every interaction. The human-in-the-loop confirmation required storing pending state in `st.session_state` and triggering `st.rerun()` carefully to avoid infinite loops.

---

### What I'd Do Next

- **Streaming LLM output**: Show the generated code token-by-token in the UI instead of waiting for the full response
- **Memory across sessions**: Persist history to SQLite so context survives restarts
- **More tools**: Open browser URLs, send emails, query local files/documents (RAG)
- **Wake word**: Always-on listener that activates on "Hey Agent"

---

### Final Thoughts

The most valuable thing I learned building this: **the hard part isn't any single component — it's the glue**. Getting audio bytes into a temp file, cleaning LLM output, sandboxing file writes, managing Streamlit's rerun model — these "boring" parts took more debugging time than the interesting AI pieces.

But that's exactly what makes building end-to-end systems valuable. Every joint in the pipeline is a potential failure point, and handling failures gracefully is what separates a demo from something real.

The full code is at: **[GitHub link]**

---

*Built for the Mem0 AI/ML Generative AI Developer Intern assignment.*
