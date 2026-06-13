"""Perplexity provider profile.

Perplexity's sonar models have real-time web search built in and return
citations alongside every response.  This profile:

  1. Cleans message history to Perplexity's required alternating
     user/assistant format — tool_calls and tool-result messages are
     stripped because Perplexity does not support function calling.
  2. Enables citations by default and exposes search configuration via
     three optional env vars:

       PPLX_SEARCH_RECENCY   month|week|day|hour   (default: unset)
       PPLX_SEARCH_DOMAINS   comma-separated list   (default: unset)
       PPLX_SEARCH_CONTEXT   low|medium|high        (default: unset)
"""

from __future__ import annotations

import os
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class PerplexityProfile(ProviderProfile):
    """Perplexity sonar models — web search + citations."""

    def prepare_messages(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Strip tool-related content and enforce alternating turns.

        Perplexity only accepts system/user/assistant messages with plain
        text content.  Tool calls in assistant turns and tool-result turns
        are removed; consecutive messages of the same role are merged so
        the alternating constraint is always satisfied.
        """
        cleaned: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "")
            if role == "tool":
                continue
            if role not in ("system", "user", "assistant"):
                continue

            content = msg.get("content", "")
            if isinstance(content, list):
                # Flatten multi-part content — keep only text parts.
                parts = [
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                ]
                content = "\n".join(parts)
            elif not isinstance(content, str):
                content = str(content) if content else ""

            # Skip assistant turns that were pure tool-call stubs (no text).
            if role == "assistant" and not content.strip():
                continue

            # Merge consecutive same-role messages.
            if cleaned and cleaned[-1]["role"] == role:
                cleaned[-1]["content"] += "\n" + content
            else:
                cleaned.append({"role": role, "content": content})

        return cleaned

    def build_extra_body(
        self, *, session_id: str | None = None, **context: Any
    ) -> dict[str, Any]:
        """Enable citations and apply optional search configuration.

        The three search knobs are read from env vars so the user can tune
        them without touching code:

          PPLX_SEARCH_RECENCY — restrict results to the last N days
          PPLX_SEARCH_DOMAINS — restrict results to specific domains
          PPLX_SEARCH_CONTEXT — control breadth of the search pass
        """
        body: dict[str, Any] = {"return_citations": True}

        recency = os.getenv("PPLX_SEARCH_RECENCY", "").strip().lower()
        if recency in ("month", "week", "day", "hour"):
            body["search_recency_filter"] = recency

        domains_raw = os.getenv("PPLX_SEARCH_DOMAINS", "").strip()
        if domains_raw:
            domains = [d.strip() for d in domains_raw.split(",") if d.strip()]
            if domains:
                body["search_domain_filter"] = domains

        ctx_size = os.getenv("PPLX_SEARCH_CONTEXT", "").strip().lower()
        if ctx_size in ("low", "medium", "high"):
            body["web_search_options"] = {"search_context_size": ctx_size}

        return body


perplexity = PerplexityProfile(
    name="perplexity",
    aliases=("pplx",),
    env_vars=("PPLX_API_KEY",),
    display_name="Perplexity",
    description="Perplexity — AI search models with real-time web access",
    signup_url="https://www.perplexity.ai/settings/api",
    base_url="https://api.perplexity.ai",
    auth_type="api_key",
    supports_vision=True,
    supports_tools=False,
    fallback_models=(
        "sonar-pro",
        "sonar",
        "sonar-reasoning-pro",
        "sonar-reasoning",
        "sonar-deep-research",
    ),
    default_aux_model="sonar",
)

register_provider(perplexity)
