# Contributing

Thanks for contributing to VoiceAgent.

## Development Setup

1. Create and activate a virtual environment.

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Create local environment config.

```bash
copy .env.example .env
```

4. Run the app.

```bash
.venv\Scripts\python.exe -m streamlit run app.py
```

## Coding Guidelines

- Use Python type hints on all function signatures.
- Use Google-style docstrings for functions and modules.
- Keep file writes sandboxed to `output/`.
- Prefer explicit exception handling over generic catch-all behavior.
- Add concise inline comments only for non-obvious logic.

## Pull Request Checklist

- [ ] Code compiles: `python -m py_compile app.py intent.py executor.py stt.py`
- [ ] New or changed functions include type hints and docstrings
- [ ] No bare `except:` blocks
- [ ] README is updated for user-facing behavior changes
- [ ] Generated artifacts are not accidentally committed unless intentional

## Commit Style

Use clear commit messages with an action verb:

- `feat: add benchmark script for intent latency`
- `fix: preserve summarize context in session state`
- `docs: rewrite README architecture section`

## Reporting Issues

When filing issues, include:

- OS and Python version
- Model name used in Ollama
- Relevant `.env` settings (redact secrets)
- Reproduction steps and observed logs
