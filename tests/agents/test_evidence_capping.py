"""Phase 12 EOD hardening: evidence content sent to the LLM is capped at a
fixed character limit, because it is the only unbounded, variable-size part
of the prompt and the dominant driver of CPU-only inference latency
(measured: an ~11k-char article pushed a single qwen2.5:3b call to ~90s in
isolation, and real same-sized articles exceeded the production timeout).
Only the LLM-facing prompt is capped -- `Evidence.content` in storage is
never touched, and title/date/publisher/URL are always sent in full.
"""

from __future__ import annotations

from research_assistant.agents.base import MAX_EVIDENCE_CONTENT_CHARS, _capped_evidence_content
from research_assistant.agents.peer import PeerAgent
from research_assistant.evidence.conversion import article_to_evidence
from research_assistant.evidence.normalizer import normalize_article
from research_assistant.retrieval.models import ExtractedArticle

from tests.agents.conftest import FakeLLMClient
from tests.pipeline.conftest import jpmorgan_source
from tests.pipeline.test_graph import _relevant_assessment


def test_short_content_is_never_truncated():
    content = "Short article body." * 5
    assert _capped_evidence_content(content) == content


def test_long_content_is_capped_with_an_explicit_marker():
    content = "x" * (MAX_EVIDENCE_CONTENT_CHARS + 5000)
    result = _capped_evidence_content(content)

    assert len(result) > MAX_EVIDENCE_CONTENT_CHARS  # marker text adds a bit on top
    assert result.startswith("x" * MAX_EVIDENCE_CONTENT_CHARS)
    assert "truncated" in result
    assert str(MAX_EVIDENCE_CONTENT_CHARS) in result


def test_capping_keeps_the_beginning_of_the_article_not_the_end():
    content = "BEGINNING-MARKER " + ("filler " * 2000) + " END-MARKER"
    result = _capped_evidence_content(content)

    assert "BEGINNING-MARKER" in result
    assert "END-MARKER" not in result


def test_prompt_sent_to_the_llm_is_capped_for_a_very_long_real_article(config, peer_source):
    long_text = "JPMorgan Chase today announced a new generative AI initiative. " * 500  # well over the cap
    extracted = ExtractedArticle(url="https://jpmorganchase.com/long", title="JPMorgan AI initiative", text=long_text)
    article = normalize_article(extracted, source=peer_source)
    evidence = article_to_evidence(article)
    assert len(evidence.content) > MAX_EVIDENCE_CONTENT_CHARS  # sanity: the uncapped evidence really is long

    llm = FakeLLMClient([_relevant_assessment()])
    agent = PeerAgent(llm, config)
    agent.analyze(article, evidence)

    assert len(llm.calls) == 1
    _system, prompt, _schema = llm.calls[0]
    assert "truncated" in prompt
    # The full long text must never reach the LLM uncapped.
    assert prompt.count("JPMorgan Chase today announced a new generative AI initiative.") < 500


def test_evidence_storage_is_never_truncated_even_though_the_prompt_is(config, peer_source):
    long_text = "JPMorgan Chase today announced a new AI initiative. " * 500
    extracted = ExtractedArticle(url="https://jpmorganchase.com/long2", title="JPMorgan AI initiative", text=long_text)
    article = normalize_article(extracted, source=peer_source)
    evidence = article_to_evidence(article)

    assert len(evidence.content) > MAX_EVIDENCE_CONTENT_CHARS * 2  # full text preserved in storage
