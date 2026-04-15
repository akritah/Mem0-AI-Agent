"""Intent executor for code, file, summary, and chat tool actions."""

import os
import re
import json
import requests
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

# ── Safe output directory ────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Prompts ──────────────────────────────────────────────────────────────────
CODE_PROMPT = """You are an expert software engineer. Write clean, well-commented, production-quality code.

Task: {description}
Language: {language}
Filename: {filename}

Return ONLY the code, no markdown fences, no explanation.
"""

SUMMARIZE_PROMPT = """You are a skilled technical writer. Summarize the following text concisely.

Text to summarize:
{text}

Provide a clear, structured summary in 3-5 bullet points followed by a 2-sentence conclusion.
"""

CHAT_PROMPT = """You are VoiceAgent, a helpful and concise AI assistant. Answer the user's question directly.

User: {text}
"""


def execute_intent(
    intent_data: Dict[str, Any],
    context: str = "",
    confirmed: bool = False,
    custom_filename: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute an action based on the classifier output.

    Args:
        intent_data: Intent payload from the classifier.
        context: Optional external context (for summarization and memory).
        confirmed: Whether a file-operation action was confirmed by the user.
        custom_filename: Optional filename override from the UI.
        model: Optional model override for Ollama calls.

    Returns:
        Normalized execution result payload for UI rendering.
    """
    primary = intent_data.get("primary_intent", "general_chat")
    sub_intents = intent_data.get("sub_intents", [])
    text = intent_data.get("raw_text", "")
    lang = intent_data.get("language") or "python"
    desc = intent_data.get("description", text)
    filename = custom_filename or intent_data.get("suggested_filename")
    model = model or os.getenv("OLLAMA_MODEL", "llama3.2")

    base_result = {
        "success": False,
        "intent": primary,
        "sub_intents": sub_intents,
        "confidence": intent_data.get("confidence", "?"),
        "transcription": text,
        "action_taken": "",
        "output": "",
        "file_path": None,
        "error": None,
    }

    try:
        if primary == "write_code":
            result = _handle_write_code(text, desc, lang, filename, model)

        elif primary == "create_file":
            result = _handle_create_file(filename or "new_file.txt")

        elif primary == "summarize_text":
            result = _handle_summarize(context or text, model)
            # Handle compound: summarize AND save
            if "create_file" in sub_intents and result["success"]:
                save_result = _save_text(
                    result["output"],
                    filename or "summary.txt"
                )
                result["file_path"] = save_result.get("file_path")
                result["action_taken"] += f" + saved to {result['file_path']}"

        elif primary == "general_chat":
            result = _handle_chat(text, model)

        else:
            result = {"success": False, "error": f"Unknown intent: {primary}"}

    except Exception as e:
        result = {"success": False, "error": str(e)}

    return {**base_result, **result}


# ── Handlers ─────────────────────────────────────────────────────────────────

def _handle_write_code(
    text: str,
    desc: str,
    lang: str,
    filename: Optional[str],
    model: str,
) -> Dict[str, Any]:
    """Generate source code with the LLM and write it into the sandbox.

    Args:
        text: Original transcription text.
        desc: Parsed task description.
        lang: Programming language hint.
        filename: Optional target filename.
        model: Ollama model name.

    Returns:
        Execution payload with generated code and output file path.
    """
    # Map language to extension
    ext_map = {
        "python": "py", "javascript": "js", "typescript": "ts",
        "java": "java", "cpp": "cpp", "c++": "cpp", "go": "go",
        "rust": "rs", "bash": "sh", "shell": "sh", "html": "html",
        "css": "css", "sql": "sql", "r": "r",
    }
    ext = ext_map.get(lang.lower(), "py")

    if not filename:
        # Auto-derive filename from description
        words = re.findall(r'[a-z]+', desc.lower())
        fn_words = [w for w in words if w not in ('a','an','the','with','and','for','to','in','create','write','make','file','code')]
        filename = "_".join(fn_words[:3]) + f".{ext}" if fn_words else f"output.{ext}"

    # Sanitize filename
    filename = re.sub(r'[^\w.\-]', '_', filename)
    if '.' not in filename:
        filename += f".{ext}"

    prompt = CODE_PROMPT.format(description=desc or text, language=lang, filename=filename)
    code = _call_ollama(prompt, model)

    if not code:
        return {"success": False, "error": "LLM returned empty code"}

    # Strip accidental markdown fences
    code = re.sub(r'^```[a-zA-Z]*\n?', '', code, flags=re.MULTILINE)
    code = re.sub(r'\n?```$', '', code, flags=re.MULTILINE)
    code = code.strip()

    file_path = _safe_write(filename, code)
    return {
        "success": True,
        "action_taken": f"Generated {lang} code → saved to {file_path}",
        "output": code,
        "file_path": str(file_path),
    }


def _handle_create_file(filename: str) -> Dict[str, Any]:
    """Create an empty file in the output sandbox.

    Args:
        filename: Requested filename.

    Returns:
        Execution payload describing the created file.
    """
    filename = re.sub(r'[^\w.\-]', '_', filename)
    file_path = _safe_write(filename, "")
    return {
        "success": True,
        "action_taken": f"Created blank file: {file_path}",
        "output": f"Empty file created at output/{filename}",
        "file_path": str(file_path),
    }


def _handle_summarize(text: str, model: str) -> Dict[str, Any]:
    """Summarize provided text using the configured LLM.

    Args:
        text: Source text to summarize.
        model: Ollama model name.

    Returns:
        Execution payload containing the summary output.
    """
    if not text.strip():
        return {"success": False, "error": "No text provided to summarize. Paste text in the 'Provide text to summarize' box."}

    prompt = SUMMARIZE_PROMPT.format(text=text[:8000])  # cap tokens
    summary = _call_ollama(prompt, model)

    if not summary:
        return {"success": False, "error": "LLM returned empty summary"}

    return {
        "success": True,
        "action_taken": "Summarized text",
        "output": summary,
    }


def _handle_chat(text: str, model: str) -> Dict[str, Any]:
    """Generate a general chat response.

    Args:
        text: User prompt.
        model: Ollama model name.

    Returns:
        Execution payload containing chat output.
    """
    prompt = CHAT_PROMPT.format(text=text)
    response = _call_ollama(prompt, model)

    if not response:
        return {"success": False, "error": "LLM returned empty response"}

    return {
        "success": True,
        "action_taken": "Responded to chat",
        "output": response,
    }


def _save_text(text: str, filename: str) -> Dict[str, Any]:
    """Save text content into the output sandbox.

    Args:
        text: Content to write.
        filename: Target output filename.

    Returns:
        Dictionary with success metadata and path.
    """
    filename = re.sub(r'[^\w.\-]', '_', filename)
    file_path = _safe_write(filename, text)
    return {"success": True, "file_path": str(file_path)}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _safe_write(filename: str, content: str) -> Path:
    """Write content into the sandboxed output directory.

    Args:
        filename: Requested filename.
        content: Text content to persist.

    Returns:
        Filesystem path to the written file.
    """
    # Prevent path traversal
    safe_name = Path(filename).name
    file_path = OUTPUT_DIR / safe_name
    file_path.write_text(content, encoding="utf-8")
    print(f"[Executor] Wrote {len(content)} chars to {file_path}")
    return file_path


def _call_ollama(prompt: str, model: str, timeout: int = 60) -> str:
    """Call Ollama for text generation.

    Args:
        prompt: Prompt sent to the model.
        model: Ollama model name.
        timeout: Default timeout seconds when env override is absent.

    Returns:
        Model response text.
    """
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    timeout = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", str(timeout)))
    retries = int(os.getenv("OLLAMA_MAX_RETRIES", "2"))
    last_error = None

    try:
        for api_mode, endpoint in _ollama_endpoints(ollama_url):
            payload = {
                "model": model,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 1500},
            }
            if api_mode == "generate":
                payload["prompt"] = prompt
            else:
                payload["messages"] = [{"role": "user", "content": prompt}]

            response = None
            for attempt in range(1, retries + 1):
                try:
                    response = requests.post(endpoint, json=payload, timeout=timeout)
                    break
                except requests.exceptions.ReadTimeout:
                    last_error = f"read timeout after {timeout}s (attempt {attempt}/{retries})"
                    if attempt == retries:
                        raise RuntimeError(
                            f"Ollama read timed out after {timeout}s. "
                            f"Try a smaller model, lower generation size, or increase OLLAMA_TIMEOUT_SECONDS in .env."
                        )

            if response is None:
                continue

            if response.status_code == 404:
                body = ""
                try:
                    body = response.json().get("error", "")
                except Exception:
                    body = response.text[:200]

                # Ollama returns 404 when the requested model does not exist.
                if "model" in body.lower() and "not found" in body.lower():
                    raise RuntimeError(
                        f"Model '{model}' not found in Ollama. Run: ollama pull {model} or choose an installed model (run: ollama list)."
                    )

                last_error = f"404 at {endpoint}: {body}"
                continue

            response.raise_for_status()
            data = response.json()
            if api_mode == "generate":
                content = data.get("response", "").strip()
            else:
                content = data.get("message", {}).get("content", "").strip()

            if content:
                return content

        raise RuntimeError(f"Ollama request failed for all endpoints. Last error: {last_error or 'unknown'}")
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Cannot connect to Ollama at {ollama_url}. Make sure Ollama is running: ollama serve"
        )
    except Exception as e:
        raise RuntimeError(f"Ollama error: {e}")


def _ollama_endpoints(ollama_url: str) -> list[tuple[str, str]]:
    """Return ordered candidate Ollama endpoints.

    Args:
        ollama_url: Base URL or explicit endpoint URL.

    Returns:
        Ordered ``(mode, endpoint)`` tuples.
    """
    base = (ollama_url or "http://localhost:11434").strip().rstrip("/")
    if base.endswith("/api/generate"):
        root = base[:-len("/api/generate")]
        return [("generate", base), ("chat", f"{root}/api/chat")]
    if base.endswith("/api/chat"):
        root = base[:-len("/api/chat")]
        return [("chat", base), ("generate", f"{root}/api/generate")]
    return [("generate", f"{base}/api/generate"), ("chat", f"{base}/api/chat")]
