"""Provider-agnostic LLM client.

Groq, Google Gemini, OpenAI, Anthropic (Claude) and Ollama all expose an
OpenAI-compatible Chat Completions endpoint, so one client covers them all.
Pick a provider with the ``LLM_PROVIDER`` environment variable and override
the model with ``LLM_MODEL`` (model names change often - check the provider's
docs if a default stops working).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

PROVIDERS: dict[str, dict[str, str | None]] = {
    # free tier, very fast
    "groq": {"base_url": "https://api.groq.com/openai/v1", "key_env": "GROQ_API_KEY",
             "model": "llama-3.3-70b-versatile"},
    # free tier
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
               "key_env": "GEMINI_API_KEY", "model": "gemini-2.5-flash"},
    "openai": {"base_url": None, "key_env": "OPENAI_API_KEY", "model": "gpt-4o-mini"},
    "anthropic": {"base_url": "https://api.anthropic.com/v1/", "key_env": "ANTHROPIC_API_KEY",
                  "model": "claude-haiku-5-5"},
    # fully local, no key: install Ollama and run `ollama pull llama3.1`
    "ollama": {"base_url": "http://localhost:11434/v1", "key_env": None, "model": "llama3.1"},
}


@dataclass
class LLMClient:
    provider: str
    model: str
    client: object  # openai.OpenAI
    temperature: float = 0.2

    def chat(self, messages: list[dict], tools: list[dict] | None = None):
        """Send one chat request and return the assistant message object."""
        kwargs = {"model": self.model, "messages": messages, "temperature": self.temperature}
        if tools:
            kwargs.update(tools=tools, tool_choice="auto")
        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message


def detect_provider() -> str | None:
    """Use LLM_PROVIDER if set, otherwise the first provider whose API key is present."""
    explicit = os.environ.get("LLM_PROVIDER")
    if explicit:
        return explicit.lower()
    for name, cfg in PROVIDERS.items():
        if cfg["key_env"] and os.environ.get(cfg["key_env"]):
            return name
    return None


def make_client(provider: str | None = None, model: str | None = None,
                temperature: float = 0.2) -> LLMClient | None:
    """Create a client, or return ``None`` when no provider is configured.

    Returning ``None`` (instead of raising) lets the agent fall back to a
    retrieval-only mode, so the notebook and tests still run without a key.
    """
    provider = (provider or detect_provider() or "").lower()
    if provider not in PROVIDERS:
        return None
    cfg = PROVIDERS[provider]
    api_key = os.environ.get(cfg["key_env"]) if cfg["key_env"] else "ollama"
    if not api_key:
        return None

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=cfg["base_url"]) if cfg["base_url"] else OpenAI(api_key=api_key)
    return LLMClient(provider=provider, model=model or os.environ.get("LLM_MODEL") or cfg["model"],
                     client=client, temperature=temperature)
