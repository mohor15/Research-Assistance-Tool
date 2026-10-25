"""Phase 0 tests: configuration loads, validates, and fails clearly when malformed.

No retrieval, agent, or integration behaviour is tested here — that belongs
to later phases.
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest

from research_assistant.config.loader import (
    ConfigError,
    load_config,
    load_settings,
    load_sources,
    load_topics,
)
from research_assistant.utils.paths import SETTINGS_PATH, SOURCES_PATH, TOPICS_PATH


def test_settings_yaml_loads_and_validates():
    settings = load_settings()
    assert settings.system.llm.provider == "ollama"
    assert settings.system.llm.model == "qwen2.5:3b"
    assert settings.system.timezone == "Asia/Kolkata"


def test_sources_yaml_loads_and_preserves_extra_sections():
    sources = load_sources()
    assert len(sources.peers) >= 8
    assert len(sources.ai_providers) >= 8
    assert len(sources.analysts) >= 5
    assert len(sources.secondary_sources) >= 8
    # These two sections were added on top of the original spec and must survive.
    assert sources.startups.funding_events, "funding_events must be preserved"


def test_every_secondary_source_now_has_a_discoverable_url():
    """Regression: secondary sources previously had no `urls:` at all, so
    discover_candidates() could never traverse them (an empty-list loop is
    a silent no-op). Each of the 11 secondary sources must now have at
    least one real, non-empty URL."""
    sources = load_sources()
    assert len(sources.secondary_sources) == 11
    for entry in sources.secondary_sources:
        assert entry.urls, f"{entry.name} has no configured urls"
        assert all(u.startswith("https://") for u in entry.urls)


def test_secondary_source_types_and_priorities_are_unchanged():
    sources = load_sources()
    by_name = {s.name: s for s in sources.secondary_sources}
    expected = {
        "Reuters": ("secondary", "high"),
        "CNBC": ("secondary", "high"),
        "Bloomberg": ("secondary", "high"),
        "Financial Times": ("secondary", "high"),
        "Barron's": ("secondary", "medium"),
        "Wall Street Journal": ("secondary", "high"),
        "TechCrunch": ("secondary", "medium"),
        "The Information": ("secondary", "medium"),
        "PR Newswire": ("press_distribution", "medium"),
        "Yahoo Finance": ("secondary", "high"),
        "TradingView": ("secondary", "high"),
    }
    assert set(by_name) == set(expected)
    for name, (source_type, priority) in expected.items():
        assert by_name[name].source_type.value == source_type
        assert by_name[name].priority.value == priority


def test_pr_newswire_remains_press_distribution_not_independent_journalism():
    sources = load_sources()
    pr_newswire = next(s for s in sources.secondary_sources if s.name == "PR Newswire")
    assert pr_newswire.source_type.value == "press_distribution"


def test_primary_source_registries_are_unchanged_by_the_secondary_url_fix():
    sources = load_sources()
    assert len(sources.peers) == 9
    assert len(sources.ai_providers) == 8
    assert len(sources.analysts) == 5
    expected_peer_names = {
        "JPMorgan Chase", "Morgan Stanley", "Bank of America", "Citi",
        "Wells Fargo", "HSBC", "Lloyds Banking Group", "Deutsche Bank", "NatWest Group",
    }
    assert {s.name for s in sources.peers} == expected_peer_names
    # Every primary-registry entry must still have exactly the url count it had before.
    for entry in sources.peers + sources.ai_providers + sources.analysts:
        assert len(entry.urls) == 1
    assert sources.retrieval_policy.blocked_behavior, "retrieval_policy must be preserved"
    assert sources.retrieval_policy.blocked_behavior["paywall"] == "do_not_bypass"


def test_topics_yaml_loads_and_validates():
    topics = load_topics()
    assert "artificial intelligence" in topics.ai_topics.core
    assert "fintech" in topics.financial_services.core
    assert "funding" in topics.event_types
    assert len(topics.strategic_themes) > 0


def test_full_config_loads_together():
    config = load_config()
    assert config.settings.research.morning_digest_hour == 8
    assert config.settings.research.evening_digest_hour == 20
    assert config.sources.peers
    assert config.topics.ai_topics.core


def test_relevance_weights_sum_to_one():
    settings = load_settings()
    weights = settings.ranking.relevance.model_dump()
    assert abs(sum(weights.values()) - 1.0) < 1e-6


def test_confidence_weights_sum_to_one():
    settings = load_settings()
    weights = settings.ranking.confidence.model_dump()
    assert abs(sum(weights.values()) - 1.0) < 1e-6


def test_env_overrides_take_precedence(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:3b")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://example-ollama:11434")
    settings = load_settings()
    assert settings.system.llm.model == "qwen2.5:3b"
    assert settings.system.llm.base_url == "http://example-ollama:11434"


def test_missing_config_file_raises_config_error(tmp_path: Path):
    with pytest.raises(ConfigError, match="not found"):
        load_settings(tmp_path / "does_not_exist.yaml")


def test_malformed_yaml_raises_config_error(tmp_path: Path):
    bad_file = tmp_path / "settings.yaml"
    bad_file.write_text("system: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="Invalid YAML"):
        load_settings(bad_file)


def test_invalid_ranking_weights_fail_validation(tmp_path: Path):
    bad_settings = tmp_path / "settings.yaml"
    bad_settings.write_text(
        textwrap.dedent(
            """
            system:
              timezone: Asia/Kolkata
              llm:
                provider: ollama
                model: qwen3:8b
                base_url: http://localhost:11434
            research:
              lookback_hours: 24
              top_k_per_agent: 5
              morning_digest_hour: 8
              evening_digest_hour: 20
              use_primary_sources_first: true
              use_secondary_sources: true
            retrieval:
              request_timeout_seconds: 20
              max_retries: 3
              retry_delay_seconds: 2
              max_articles_per_source: 30
              max_candidates_per_source: 50
              browser_fallback: true
              respect_robots_txt: true
              respect_rate_limits: true
              cache_pages: true
            ranking:
              relevance:
                ai_relevance: 0.5
                financial_services_relevance: 0.5
                strategic_significance: 0.5
                recency: 0.5
                novelty: 0.5
              confidence:
                primary_source_confirmation: 0.35
                multiple_source_corroboration: 0.25
                publication_date_certainty: 0.15
                entity_extraction_confidence: 0.10
                event_classification_confidence: 0.10
                content_completeness: 0.05
            deduplication:
              enabled: true
              similarity_threshold: 0.85
              cluster_related_events: true
            agents:
              peer: {enabled: true, top_k: 5}
              ai_product: {enabled: true, top_k: 5}
              startup: {enabled: true, top_k: 5}
              analyst: {enabled: true, top_k: 5}
              report: {enabled: true}
            report:
              include_relevance_score: true
              include_confidence_score: true
              include_source_evidence: true
              require_source_for_claim: true
              include_nomura_opportunity: true
            storage:
              database: sqlite
              database_path: data/research.db
            """
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="validation failed"):
        load_settings(bad_settings)


def test_unsupported_llm_provider_fails_validation(tmp_path: Path):
    bad_settings_data = load_settings()  # sanity: real file loads first
    assert bad_settings_data.system.llm.provider == "ollama"

    bad_file = tmp_path / "settings.yaml"
    bad_file.write_text(
        SETTINGS_PATH.read_text(encoding="utf-8").replace("provider: ollama", "provider: openai"),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="validation failed"):
        load_settings(bad_file)


def test_config_files_exist_at_expected_paths():
    assert SETTINGS_PATH.exists()
    assert SOURCES_PATH.exists()
    assert TOPICS_PATH.exists()
