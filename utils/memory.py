"""Mem0 integration helpers for cross-session memory."""

import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

try:
    from mem0 import MemoryClient
except Exception:  # pragma: no cover - import failure handled gracefully
    MemoryClient = None

USER_ID = os.getenv("MEM0_USER_ID", "voice-agent-user")


def _get_client() -> Optional[Any]:
    """Create a Mem0 client only when configuration is valid.

    Returns:
        Initialized Mem0 client instance when available, otherwise ``None``.
    """
    api_key = os.getenv("MEM0_API_KEY")
    if not api_key or MemoryClient is None:
        return None
    try:
        return MemoryClient(api_key=api_key)
    except Exception:
        return None


def _coerce_results(payload: Any) -> List[Dict[str, Any]]:
    """Normalize Mem0 responses into a list.

    Args:
        payload: Raw response payload from Mem0 SDK methods.

    Returns:
        A list of memory dictionaries.
    """
    if payload is None:
        return []
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        results = payload.get("results")
        if isinstance(results, list):
            return results
    return []


def save_interaction(command: str, intent: str, result: str) -> Tuple[bool, str]:
    """Persist an interaction for future contextual recall.

    Args:
        command: Original user command.
        intent: Classified primary intent.
        result: Short assistant result summary.

    Returns:
        Tuple of ``(success, message)``.
    """
    client = _get_client()
    if client is None:
        return False, "Mem0 disabled (missing MEM0_API_KEY or mem0ai package)."

    try:
        client.add(
            [
                {"role": "user", "content": command},
                {"role": "assistant", "content": f"Intent: {intent}. Result: {result}"},
            ],
            user_id=USER_ID,
        )
        return True, "saved"
    except Exception as e:
        return False, str(e)


def get_relevant_context(command: str, limit: int = 5) -> str:
    """Fetch relevant memory facts for a command.

    Args:
        command: Current command text.
        limit: Maximum number of memories to retrieve.

    Returns:
        Prompt-ready memory context string.
    """
    client = _get_client()
    if client is None:
        return ""

    try:
        memories = _coerce_results(client.search(command, user_id=USER_ID, limit=limit))
        if not memories:
            return ""
        # Keep context compact to avoid overloading the downstream LLM prompt.
        facts = "\n".join(f"- {m.get('memory', '').strip()}" for m in memories if m.get("memory"))
        return f"Relevant past context:\n{facts}" if facts else ""
    except Exception:
        return ""


def get_learned_facts(limit: int = 8) -> List[Dict[str, str]]:
    """Fetch recent facts for the sidebar memory panel.

    Args:
        limit: Maximum number of facts to return.

    Returns:
        List of dictionaries with ``memory`` and ``created_at`` keys.
    """
    client = _get_client()
    if client is None:
        return []

    try:
        query = "coding preferences, style, project habits, language and file naming choices"
        memories = _coerce_results(client.search(query, user_id=USER_ID, limit=limit))
        facts = []
        for m in memories:
            text = (m.get("memory") or "").strip()
            if not text:
                continue
            created = m.get("created_at") or m.get("updated_at") or ""
            if not created:
                created = datetime.now().strftime("%Y-%m-%d %H:%M")
            facts.append({"memory": text, "created_at": str(created)})
        return facts
    except Exception:
        return []
