# Index - Browser AI Agent

## Overview

Index is a state-of-the-art open-source browser agent that autonomously executes complex web tasks using vision-capable LLMs. It can navigate websites, interact with elements, and extract structured data.

## Architecture

- **Language**: Python 3.10+
- **Package Manager**: uv (with virtual environment at `.venv/`)
- **Build System**: Hatchling
- **Browser Automation**: Playwright (Chromium)
- **CLI Framework**: Typer + Textual (rich terminal UI)
- **LLM Providers**: Gemini, Anthropic (Claude), OpenAI, Groq
- **Observability**: Laminar (lmnr)

## Project Structure

```
index/
  agent/       - Agent logic, prompts, message management, models
  browser/     - Playwright browser lifecycle and element detection
  controller/  - Action execution (click, type, scroll, etc.)
  llm/         - LLM provider abstraction (OpenAI, Anthropic, Gemini, Groq)
  cli.py       - CLI entry point
evals/         - Evaluation scripts
tests/         - Unit tests (pytest)
static/        - Static assets
```

## Setup & Running

The project uses a Python virtual environment at `.venv/`. Dependencies are installed via uv.

### Workflow
- **Start application**: `source .venv/bin/activate && python webapp/server.py`
- Serves a web UI at port 5000
- FastAPI backend with WebSocket for real-time agent streaming
- Beautiful dark-themed UI with live browser screenshots

### API Keys Required (set via Secrets)
- `GEMINI_API_KEY` - For Gemini 2.5 Pro/Flash models
- `ANTHROPIC_API_KEY` - For Claude 3.7 Sonnet
- `OPENAI_API_KEY` - For OpenAI o4-mini
- `LMNR_PROJECT_API_KEY` - Optional, for Laminar observability/tracing

## Important Notes

### Compatibility Fixes Applied
1. **lmnr version**: Pinned to `<0.7` (0.6.x) because newer versions removed `use_span` which is used in `index/agent/agent.py`
2. **Playwright StorageState**: Removed `typing.StorageState` import from `index/agent/models.py` and changed `storage_state` field type to `Optional[Any]` to fix Pydantic compatibility on Python < 3.12

### Playwright
- Chromium is installed at `.cache/ms-playwright/`
- Runs headless by default
- Supports connecting to local Chrome via `--local-chrome` flag

## Usage

```bash
# Interactive CLI mode
index run

# Interactive CLI with initial prompt
index run --prompt "Go to example.com and summarize the page"

# GUI mode (Textual TUI)
index ui

# Connect to local Chrome
index run --local-chrome
```
