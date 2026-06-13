"""Tests for the Perplexity provider profile."""

import pytest

from providers import get_provider_profile


@pytest.fixture()
def profile():
    return get_provider_profile("perplexity")


# =============================================================================
# Registry
# =============================================================================


class TestPerplexityRegistry:
    def test_discovered(self, profile):
        assert profile is not None

    def test_name(self, profile):
        assert profile.name == "perplexity"

    def test_alias_pplx(self):
        assert get_provider_profile("pplx").name == "perplexity"

    def test_env_var(self, profile):
        assert "PPLX_API_KEY" in profile.env_vars

    def test_base_url(self, profile):
        assert profile.base_url == "https://api.perplexity.ai"

    def test_auth_type(self, profile):
        assert profile.auth_type == "api_key"

    def test_supports_vision(self, profile):
        assert profile.supports_vision is True

    def test_fallback_models_present(self, profile):
        assert "sonar-pro" in profile.fallback_models
        assert "sonar" in profile.fallback_models
        assert "sonar-reasoning-pro" in profile.fallback_models
        assert "sonar-deep-research" in profile.fallback_models

    def test_default_aux_model(self, profile):
        assert profile.default_aux_model == "sonar"

    def test_hostname_derived(self, profile):
        assert "perplexity.ai" in profile.get_hostname()


# =============================================================================
# build_extra_body — citation default
# =============================================================================


class TestBuildExtraBody:
    def test_citations_always_enabled(self, profile):
        body = profile.build_extra_body()
        assert body["return_citations"] is True

    def test_no_recency_by_default(self, profile):
        body = profile.build_extra_body()
        assert "search_recency_filter" not in body

    def test_no_domains_by_default(self, profile):
        body = profile.build_extra_body()
        assert "search_domain_filter" not in body

    def test_no_web_options_by_default(self, profile):
        body = profile.build_extra_body()
        assert "web_search_options" not in body

    def test_recency_month(self, profile, monkeypatch):
        monkeypatch.setenv("PPLX_SEARCH_RECENCY", "month")
        body = profile.build_extra_body()
        assert body["search_recency_filter"] == "month"

    def test_recency_hour(self, profile, monkeypatch):
        monkeypatch.setenv("PPLX_SEARCH_RECENCY", "hour")
        body = profile.build_extra_body()
        assert body["search_recency_filter"] == "hour"

    def test_recency_invalid_ignored(self, profile, monkeypatch):
        monkeypatch.setenv("PPLX_SEARCH_RECENCY", "yesterday")
        body = profile.build_extra_body()
        assert "search_recency_filter" not in body

    def test_domains_single(self, profile, monkeypatch):
        monkeypatch.setenv("PPLX_SEARCH_DOMAINS", "arxiv.org")
        body = profile.build_extra_body()
        assert body["search_domain_filter"] == ["arxiv.org"]

    def test_domains_multiple(self, profile, monkeypatch):
        monkeypatch.setenv("PPLX_SEARCH_DOMAINS", "reddit.com, arxiv.org, github.com")
        body = profile.build_extra_body()
        assert body["search_domain_filter"] == ["reddit.com", "arxiv.org", "github.com"]

    def test_domains_empty_string_ignored(self, profile, monkeypatch):
        monkeypatch.setenv("PPLX_SEARCH_DOMAINS", "")
        body = profile.build_extra_body()
        assert "search_domain_filter" not in body

    def test_context_size_high(self, profile, monkeypatch):
        monkeypatch.setenv("PPLX_SEARCH_CONTEXT", "high")
        body = profile.build_extra_body()
        assert body["web_search_options"] == {"search_context_size": "high"}

    def test_context_size_low(self, profile, monkeypatch):
        monkeypatch.setenv("PPLX_SEARCH_CONTEXT", "low")
        body = profile.build_extra_body()
        assert body["web_search_options"] == {"search_context_size": "low"}

    def test_context_size_invalid_ignored(self, profile, monkeypatch):
        monkeypatch.setenv("PPLX_SEARCH_CONTEXT", "ultra")
        body = profile.build_extra_body()
        assert "web_search_options" not in body

    def test_all_options_together(self, profile, monkeypatch):
        monkeypatch.setenv("PPLX_SEARCH_RECENCY", "week")
        monkeypatch.setenv("PPLX_SEARCH_DOMAINS", "arxiv.org")
        monkeypatch.setenv("PPLX_SEARCH_CONTEXT", "medium")
        body = profile.build_extra_body()
        assert body["return_citations"] is True
        assert body["search_recency_filter"] == "week"
        assert body["search_domain_filter"] == ["arxiv.org"]
        assert body["web_search_options"] == {"search_context_size": "medium"}

    def test_session_id_ignored(self, profile):
        """session_id is accepted but not forwarded (Perplexity has no use for it)."""
        body = profile.build_extra_body(session_id="test-session")
        assert "session_id" not in body
        assert body["return_citations"] is True


# =============================================================================
# prepare_messages — tool stripping + alternation enforcement
# =============================================================================


class TestPrepareMessages:
    def test_passthrough_plain_conversation(self, profile):
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]
        result = profile.prepare_messages(msgs)
        assert len(result) == 4
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[2]["role"] == "assistant"
        assert result[3]["role"] == "user"

    def test_tool_result_messages_stripped(self, profile):
        msgs = [
            {"role": "user", "content": "Run ls"},
            {"role": "assistant", "content": ""},  # pure tool-call stub
            {"role": "tool", "content": "file1.txt\nfile2.txt", "tool_call_id": "x"},
            {"role": "user", "content": "Thanks"},
        ]
        result = profile.prepare_messages(msgs)
        roles = [m["role"] for m in result]
        assert "tool" not in roles

    def test_empty_assistant_tool_stub_stripped(self, profile):
        """Assistant turns with no text content (pure tool_call stubs) are dropped."""
        msgs = [
            {"role": "user", "content": "do something"},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "follow up"},
        ]
        result = profile.prepare_messages(msgs)
        roles = [m["role"] for m in result]
        # The empty assistant message should be gone
        assert roles.count("assistant") == 0

    def test_consecutive_user_messages_merged(self, profile):
        msgs = [
            {"role": "user", "content": "part one"},
            {"role": "user", "content": "part two"},
        ]
        result = profile.prepare_messages(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert "part one" in result[0]["content"]
        assert "part two" in result[0]["content"]

    def test_list_content_flattened_to_text(self, profile):
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
                ],
            }
        ]
        result = profile.prepare_messages(msgs)
        assert len(result) == 1
        assert result[0]["content"] == "What is this?"

    def test_list_content_multiple_text_parts_joined(self, profile):
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "first"},
                    {"type": "text", "text": "second"},
                ],
            }
        ]
        result = profile.prepare_messages(msgs)
        assert result[0]["content"] == "first\nsecond"

    def test_unknown_roles_dropped(self, profile):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "function", "content": "some result"},
        ]
        result = profile.prepare_messages(msgs)
        roles = [m["role"] for m in result]
        assert "function" not in roles

    def test_system_message_preserved(self, profile):
        msgs = [{"role": "system", "content": "You are a search assistant."}]
        result = profile.prepare_messages(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are a search assistant."

    def test_empty_messages_list(self, profile):
        assert profile.prepare_messages([]) == []

    def test_tool_messages_between_turns_cleaned(self, profile):
        """Realistic agentic history: tool calls between real user turns."""
        msgs = [
            {"role": "user", "content": "Search for papers on LLMs"},
            {"role": "assistant", "content": ""},  # tool stub
            {"role": "tool", "content": "result 1"},
            {"role": "tool", "content": "result 2"},
            {"role": "assistant", "content": "Here is what I found…"},
            {"role": "user", "content": "Summarise"},
        ]
        result = profile.prepare_messages(msgs)
        roles = [m["role"] for m in result]
        assert "tool" not in roles
        # Should end with a proper alternating sequence
        assert roles[-1] == "user"
        assert roles[-2] == "assistant"
