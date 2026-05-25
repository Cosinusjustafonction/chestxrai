"""
Central LLM configuration.

Set these environment variables (or add them to a .env file) to switch providers
without touching any code:

  AGENT_LLM_PROVIDER   = ollama | openai | anthropic | groq | gemini  (default: ollama)
  AGENT_LLM_MODEL      = mistral:7b | gpt-4o-mini | gemini-2.5-flash  (default: mistral:7b)

  MANAGER_LLM_PROVIDER = ollama | openai | anthropic | groq | gemini  (default: ollama)
  MANAGER_LLM_MODEL    = qwen2.5:14b | gpt-4o | gemini-2.5-pro        (default: qwen2.5:14b)

  VLM_PROVIDER         = ollama | openai | anthropic | gemini          (default: ollama)
  VLM_MODEL            = llava:7b | gpt-4o | gemini-2.5-flash          (default: llava:7b)

  OLLAMA_BASE_URL      = http://localhost:11434                        (default)
  OPENAI_API_KEY       = sk-...
  ANTHROPIC_API_KEY    = sk-ant-...
  GROQ_API_KEY         = gsk_...
  GEMINI_API_KEY       = AIza...
  OPENROUTER_API_KEY   = sk-or-v1-...  (also accepts OPENAI_API_KEY as fallback)
"""

import os
from crewai import LLM

# ── Shared base settings ───────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def build_llm(provider: str | None = None, model: str | None = None, **kwargs) -> LLM:
    """
    Build a CrewAI LLM from a provider name and model string.

    Supported providers:
      ollama      — local Ollama server
      openai      — OpenAI API
      anthropic   — Anthropic API
      groq        — Groq API (OpenAI-compatible, very fast)
      gemini      — Google Gemini API
      openrouter  — OpenRouter (access any model via one API key)
    """
    provider = (provider or "ollama").lower().strip()
    model    = model or "mistral:7b"

    if provider == "ollama":
        return LLM(model=f"ollama/{model}", base_url=OLLAMA_BASE_URL, **kwargs)

    if provider == "openai":
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise EnvironmentError("OPENAI_API_KEY is not set")
        return LLM(model=model, api_key=key, **kwargs)

    if provider == "anthropic":
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise EnvironmentError("ANTHROPIC_API_KEY is not set")
        return LLM(model=f"anthropic/{model}", api_key=key, **kwargs)

    if provider == "groq":
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise EnvironmentError("GROQ_API_KEY is not set")
        return LLM(model=f"groq/{model}", api_key=key, **kwargs)

    if provider == "gemini":
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise EnvironmentError("GEMINI_API_KEY is not set")
        return LLM(model=f"gemini/{model}", api_key=key, **kwargs)

    if provider == "openrouter":
        key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not key:
            raise EnvironmentError("OPENROUTER_API_KEY (or OPENAI_API_KEY) is not set")
        return LLM(model=f"openrouter/{model}", api_key=key, **kwargs)

    raise ValueError(
        f"Unknown LLM provider '{provider}'. "
        "Supported: ollama, openai, anthropic, groq, gemini, openrouter"
    )


# ── Pre-built instances used by crew.py ───────────────────────

# Specialist agents — tasks are narrow (call one tool, format output)
agent_llm = build_llm(
    provider=os.getenv("AGENT_LLM_PROVIDER", "ollama"),
    model=os.getenv("AGENT_LLM_MODEL",    "mistral:7b"),
)

# Hierarchical orchestrator — needs strong instruction-following
manager_llm = build_llm(
    provider=os.getenv("MANAGER_LLM_PROVIDER", "ollama"),
    model=os.getenv("MANAGER_LLM_MODEL",    "qwen2.5:14b"),
)
