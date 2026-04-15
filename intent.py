"""Intent classification for voice commands with Ollama and rule fallback."""

import json
import re
import os
from typing import Any, Dict, List, Optional, Tuple


INTENT_PROMPT = """You are an intent classifier for a voice-controlled AI agent.

Analyze the user command and return ONLY a valid JSON object (no markdown, no explanation).

JSON schema:
{{
    "primary_intent": "<one of: create_file | write_code | summarize_text | general_chat>",
    "sub_intents": ["<additional intents if compound command, else empty list>"],
    "confidence": "<high | medium | low>",
    "suggested_filename": "<filename with extension if file operation, else null>",
    "language": "<programming language if write_code, else null>",
    "description": "<brief description of what to do>",
    "is_compound": <true | false>
}}

Rules:
- If user says "create a python file" → write_code (not create_file)
- If user says "create a blank/empty file" → create_file
- If user says "summarize and save" → primary=summarize_text, sub_intents=["create_file"], is_compound=true
- Always suggest a filename when file operations are involved
- Confidence = high if intent is very clear, medium if somewhat ambiguous, low if unclear

User command: {command}

{memory_context}
"""


def classify_intent(text: str, model: str = "llama3.2", memory_context: str = "") -> Tuple[Dict[str, Any], Optional[str]]:
    """Classify user text into an intent payload.

    Args:
        text: User utterance or typed command.
        model: Ollama model name used for classification.
        memory_context: Optional semantic memory injected into the prompt.

    Returns:
        A tuple of ``(intent_dict, error_message)``. The error value is ``None``
        on successful classification.
    """
    # Try Ollama first
    try:
        return _classify_ollama(text, model, memory_context)
    except Exception as e:
        print(f"[Intent] Ollama failed ({e}), falling back to rules...")

    # Graceful degradation: rule-based fallback
    return _classify_rules(text), None


def _classify_ollama(text: str, model: str, memory_context: str = "") -> Tuple[Dict[str, Any], Optional[str]]:
    """Call Ollama endpoints to classify intent.

    Args:
        text: User utterance to classify.
        model: Ollama model name.
        memory_context: Optional retrieved memory context.

    Returns:
        A tuple of ``(intent_dict, error_message)``.
    """
    import requests

    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    timeout = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "60"))
    retries = int(os.getenv("OLLAMA_MAX_RETRIES", "2"))
    prompt = INTENT_PROMPT.format(command=text, memory_context=memory_context or "")

    raw = ""
    last_error = None
    for api_mode, endpoint in _ollama_endpoints(ollama_url):
        try:
            payload = {
                "model": model,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 300},
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
                            f"Increase OLLAMA_TIMEOUT_SECONDS in .env or use a smaller model."
                        )

            if response is None:
                continue

            if response.status_code == 404:
                body = ""
                try:
                    body = response.json().get("error", "")
                except Exception:
                    body = response.text[:200]

                if "model" in body.lower() and "not found" in body.lower():
                    raise RuntimeError(
                        f"Model '{model}' not found in Ollama. Run: ollama pull {model} or choose an installed model."
                    )

                # Some Ollama-compatible servers only expose one of these endpoints.
                last_error = f"404 at {endpoint}: {body}"
                continue

            response.raise_for_status()
            data = response.json()
            if api_mode == "generate":
                raw = data.get("response", "")
            else:
                raw = data.get("message", {}).get("content", "")

            if raw:
                break
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Cannot connect to Ollama at {ollama_url}. Start it with: ollama serve"
            )
        except Exception as e:
            last_error = str(e)

    if not raw:
        raise RuntimeError(
            f"Ollama request failed for all endpoints. Last error: {last_error or 'unknown'}"
        )

    # Extract JSON from response (sometimes model wraps in markdown)
    json_match = re.search(r'\{.*\}', raw, re.DOTALL)
    if not json_match:
        raise ValueError(f"No JSON in response: {raw[:200]}")

    intent_data = json.loads(json_match.group())
    intent_data["raw_text"] = text
    print(f"[Intent-Ollama] {intent_data}")
    return intent_data, None


def _ollama_endpoints(ollama_url: str) -> List[Tuple[str, str]]:
    """Build ordered endpoint candidates for Ollama-compatible APIs.

    Args:
        ollama_url: Base URL or explicit API endpoint.

    Returns:
        Ordered list of ``(mode, endpoint_url)`` tuples.
    """
    base = (ollama_url or "http://localhost:11434").strip().rstrip("/")
    if base.endswith("/api/generate"):
        root = base[:-len("/api/generate")]
        return [("generate", base), ("chat", f"{root}/api/chat")]
    if base.endswith("/api/chat"):
        root = base[:-len("/api/chat")]
        return [("chat", base), ("generate", f"{root}/api/generate")]
    return [("generate", f"{base}/api/generate"), ("chat", f"{base}/api/chat")]


def _classify_rules(text: str) -> Dict[str, Any]:
    """Classify intent with deterministic keyword rules.

    Args:
        text: User command text.

    Returns:
        Intent payload matching the LLM classifier schema.
    """
    t = text.lower()

    # Keyword lists
    code_kw   = ["code", "function", "script", "program", "class", "write", "python", "javascript", "java", "cpp", "generate"]
    file_kw   = ["create file", "make file", "new file", "create a file", "empty file", "blank file", "mkdir", "folder"]
    summ_kw   = ["summarize", "summarise", "summary", "brief", "tldr", "condense", "shorten"]
    compound  = ["and save", "and store", "and write", "then save"]

    sub_intents = []
    is_compound = any(kw in t for kw in compound)

    if any(kw in t for kw in summ_kw):
        primary = "summarize_text"
        if is_compound:
            sub_intents = ["create_file"]
    elif any(kw in t for kw in code_kw):
        primary = "write_code"
    elif any(kw in t for kw in file_kw):
        primary = "create_file"
    else:
        primary = "general_chat"

    # Detect language
    lang = None
    for l in ["python", "javascript", "typescript", "java", "cpp", "c++", "go", "rust", "bash", "shell"]:
        if l in t:
            lang = l
            break

    # Suggest filename
    filename = None
    fn_match = re.search(r'(?:called?|named?|file\s+)([a-zA-Z0-9_\-]+\.?[a-zA-Z0-9]*)', t)
    if fn_match:
        filename = fn_match.group(1)
        if '.' not in filename:
            ext = {"python":"py","javascript":"js","typescript":"ts","java":"java","cpp":"cpp","go":"go","rust":"rs","bash":"sh"}.get(lang or "", "txt")
            filename += f".{ext}"
    elif primary == "write_code" and lang:
        filename = f"output.{lang[:2]}" if lang in ['python'] else f"output.{lang}"
        if lang == "python": filename = "output.py"

    return {
        "primary_intent": primary,
        "sub_intents": sub_intents,
        "confidence": "medium",
        "suggested_filename": filename,
        "language": lang,
        "description": f"Rule-based classification of: {text[:80]}",
        "is_compound": is_compound,
        "raw_text": text,
        "fallback": True,
    }
