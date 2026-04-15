"""Benchmark intent classification latency across Ollama and rule fallback."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

from intent import _classify_ollama, _classify_rules

TEST_COMMANDS: List[str] = [
    "Create a blank file called notes.txt",
    "Write a Python function for binary search and save it as search.py",
    "Summarize this article in five bullet points",
    "Summarize this text and save it to summary.txt",
    "Explain recursion in simple terms",
]


def _run_ollama(command: str, model: str) -> Tuple[Dict[str, Any], str]:
    """Run Ollama classifier path and return payload with optional error text.

    Args:
        command: Command text to classify.
        model: Ollama model name.

    Returns:
        Tuple of classification payload and optional error string.
    """
    try:
        payload, _ = _classify_ollama(command, model, memory_context="")
        return payload, ""
    except (RuntimeError, ValueError) as exc:
        return {
            "primary_intent": "error",
            "confidence": "n/a",
            "description": "ollama_error",
        }, str(exc)


def _run_rules(command: str) -> Tuple[Dict[str, Any], str]:
    """Run rule-based classifier path.

    Args:
        command: Command text to classify.

    Returns:
        Tuple of classification payload and optional error string.
    """
    return _classify_rules(command), ""


def _format_ms(seconds: float) -> str:
    """Convert seconds to millisecond string.

    Args:
        seconds: Latency in seconds.

    Returns:
        Millisecond string with two decimal precision.
    """
    return f"{seconds * 1000:.2f}"


def _print_table(rows: List[Dict[str, str]]) -> None:
    """Print benchmark rows as a fixed-width table.

    Args:
        rows: Normalized table rows to print.
    """
    headers = ["backend", "command", "intent", "confidence", "latency_ms", "error"]
    widths = {h: len(h) for h in headers}

    for row in rows:
        for header in headers:
            widths[header] = max(widths[header], len(row.get(header, "")))

    sep = " | "
    header_line = sep.join(header.ljust(widths[header]) for header in headers)
    divider_line = "-+-".join("-" * widths[header] for header in headers)

    print(header_line)
    print(divider_line)
    for row in rows:
        print(sep.join(row.get(header, "").ljust(widths[header]) for header in headers))


def main() -> None:
    """Run benchmark scenarios against both classifier backends."""
    ollama_model = "llama3"
    rows: List[Dict[str, str]] = []

    for command in TEST_COMMANDS:
        for backend in ("ollama", "rules"):
            start = time.perf_counter()
            if backend == "ollama":
                payload, error_text = _run_ollama(command, ollama_model)
            else:
                payload, error_text = _run_rules(command)
            elapsed = time.perf_counter() - start

            rows.append(
                {
                    "backend": backend,
                    "command": command,
                    "intent": str(payload.get("primary_intent", "")),
                    "confidence": str(payload.get("confidence", "")),
                    "latency_ms": _format_ms(elapsed),
                    "error": error_text[:80],
                }
            )

    _print_table(rows)


if __name__ == "__main__":
    main()
