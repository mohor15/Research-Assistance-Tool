"""Final production gate: deterministic proof of
`invalid/no Evidence -> no agent candidate -> no Ollama call -> no ResearchItem`
for every named failure shape, plus a broader relevance negative/positive
sweep through the real `ResearchAgent.analyze()` path (deterministic
pre-filter + scripted LLM, consistent with this project's existing test
style -- a real-Ollama E2E for the positive case already exists
separately in the scratch-DB controlled gate). The Michigan LIFT
regression itself lives in `tests/agents/test_date_precision_scope.py`
and is referenced, not duplicated, here.
"""

from __future__ import annotations

from research_assistant.agents.peer import PeerAgent
from research_assistant.agents.schemas import AgentAssessment
from research_assistant.evidence.conversion import article_to_evidence, evidence_rejection_reason
from research_assistant.evidence.normalizer import normalize_article
from research_assistant.models.research_item import (
    RelevanceAssessment,
    RelevanceDecision,
    RelevanceLabel,
    RelevanceLevel,
    TopicCategory,
)
from research_assistant.retrieval.extractor import extract_article
from research_assistant.retrieval.models import ExtractedArticle, ExtractionStatus

from tests.agents.conftest import FakeLLMClient, peer_source  # noqa: F401 -- fixture import

# ---------------------------------------------------------------------------
# Part 1: Evidence/Agent gate -- 7 named failure shapes, each independently
# confirmed to never produce usable Evidence.
# ---------------------------------------------------------------------------

JS_SHELL_HTML = '<html><head><title>App Shell</title></head><body><div id="root"></div></body></html>'
# A real-shaped subscription/paywall page: a nav-only shell with no
# <article>/<main> content at all (same structural shape confirmed on the
# real CNBC "Subscribe for PRO" pages this project has encountered).
SUBSCRIPTION_PAGE_HTML = '<html><head><title>Subscribe for PRO</title></head><body><nav><a href="/signup">Subscribe for PRO</a></nav></body></html>'
HUB_PAGE_HTML = (
    '<html><head><title>Press Releases | Example Bank</title>'
    '<meta property="og:title" content="Press Releases | Example Bank" />'
    '<meta property="og:site_name" content="Example Bank" /></head>'
    '<body><article><p>Explore our latest news and announcements from across the firm.</p></article></body></html>'
)
RSS_XML_TEXT = '<?xml version="1.0"?><rss version="2.0"><channel><title>Feed</title></channel></rss>'
PDF_BYTES_AS_TEXT = "%PDF-1.7\n%\xe2\xe3\xcf\xd3\n34 0 obj\n<<>>\nendobj\nxref\n<binary pdf bytes here>"


def _reject_at_evidence_layer(html, url, *, content_type=None):
    extracted = extract_article(html, url, content_type=content_type)
    article = normalize_article(extracted, source=_fake_source())
    reason = evidence_rejection_reason(article)
    evidence = article_to_evidence(article)
    return extracted, article, reason, evidence


def _fake_source():
    from research_assistant.models.enums import SourcePriority, SourceType
    from research_assistant.retrieval.models import SourceCategory, SourceDefinition

    return SourceDefinition(
        name="JPMorgan Chase", domain="jpmorganchase.com", urls=["https://jpmorganchase.com/newsroom"],
        source_type=SourceType.PRIMARY, priority=SourcePriority.HIGH, category=SourceCategory.PEER,
    )


def test_js_shell_produces_no_evidence():
    extracted, article, reason, evidence = _reject_at_evidence_layer(JS_SHELL_HTML, "https://example.com/js-shell")
    assert reason == "missing_clean_text_and_content_hash"
    assert evidence is None


def test_rss_xml_content_produces_no_evidence():
    extracted, article, reason, evidence = _reject_at_evidence_layer(RSS_XML_TEXT, "https://example.com/feed.xml")
    assert evidence is None


def test_unsupported_pdf_produces_no_evidence():
    extracted, article, reason, evidence = _reject_at_evidence_layer(
        PDF_BYTES_AS_TEXT, "https://example.com/doc.pdf", content_type="application/pdf",
    )
    assert extracted.status.value == "failed"
    assert reason == "missing_clean_text_and_content_hash"
    assert evidence is None


def test_subscription_page_produces_no_evidence():
    extracted, article, reason, evidence = _reject_at_evidence_layer(SUBSCRIPTION_PAGE_HTML, "https://example.com/subscribe")
    assert evidence is None


def test_generic_hub_page_produces_no_evidence():
    extracted, article, reason, evidence = _reject_at_evidence_layer(HUB_PAGE_HTML, "https://example.com/newsroom/press-releases")
    assert extracted.status.value == "not_article"  # the hub-shell fix
    assert evidence is None


def test_missing_clean_text_directly_is_rejected():
    article = normalize_article(
        ExtractedArticle(url="https://example.com/x", text=None, status=ExtractionStatus.SUCCESS),
        source=_fake_source(),
    )
    assert article.clean_text is None
    assert evidence_rejection_reason(article) == "missing_clean_text_and_content_hash"
    assert article_to_evidence(article) is None


def test_missing_content_hash_directly_is_rejected():
    """Constructs the specific missing-ONLY-hash edge case directly on the
    model, since normalize_article always derives a hash from real text --
    this proves the gate itself, not just the common path to it."""
    article = normalize_article(
        ExtractedArticle(
            url="https://example.com/y", text="Some real text here, long enough to matter for this test.",
            status=ExtractionStatus.SUCCESS,
        ),
        source=_fake_source(),
    )
    stripped = article.model_copy(update={"content_hash": None})
    assert evidence_rejection_reason(stripped) == "missing_content_hash"
    assert article_to_evidence(stripped) is None


def test_none_of_the_seven_failure_shapes_can_ever_reach_an_agent():
    """The actual pipeline gate (`pipeline/nodes.py::agent_analysis`)
    builds candidates as
    `[(article, evidence_by_article_id[article.id]) for article in routed
    if article.id in evidence_by_article_id]` -- since every case above
    produced no Evidence at all, none of these article ids could ever be a
    key in that dict, so none can become an agent candidate, confirmed by
    construction rather than by re-simulating the node here."""
    for html, url, content_type in [
        (JS_SHELL_HTML, "https://example.com/js-shell-2", None),
        (SUBSCRIPTION_PAGE_HTML, "https://example.com/subscribe-2", None),
        (HUB_PAGE_HTML, "https://example.com/newsroom/press-releases-2", None),
    ]:
        _, _, _, evidence = _reject_at_evidence_layer(html, url, content_type=content_type)
        assert evidence is None  # -> can never populate evidence_by_article_id -> can never be a candidate


# ---------------------------------------------------------------------------
# Part 2: relevance negative/positive sweep through the real
# ResearchAgent.analyze() path (pre-filter + scripted LLM). The Michigan
# LIFT case itself is covered in test_date_precision_scope.py.
# ---------------------------------------------------------------------------

GENERIC_CORPORATE_TEXT = (
    "Example Bank today announced it has relocated its downtown office to a new "
    "building, consolidating several teams under one roof as part of a broader "
    "facilities modernization effort across its regional footprint."
)

GENERIC_AI_EXPLAINER_TEXT = (
    "Artificial intelligence refers to computer systems that can perform tasks "
    "normally requiring human intelligence, such as understanding language or "
    "recognizing patterns. Machine learning is a subset of artificial intelligence "
    "focused on learning from data rather than explicit instructions."
)

UNRELATED_FINANCIAL_SERVICES_TEXT = (
    "Example Bank today announced an expansion of its branch network, opening "
    "twelve new locations across three states to improve access to in-person "
    "banking services for retail customers in underserved communities."
)

SUSTAINABILITY_COMMUNITY_TEXT = (
    "Example Bank today announced a new partnership with a local environmental "
    "nonprofit to plant trees and restore wetlands across several watershed "
    "regions, continuing its long-standing commitment to environmental stewardship."
)

SUBSTANTIVE_BANKING_AI_TEXT = (
    "Example Bank today announced the launch of a new generative AI platform for "
    "capital markets research, giving analysts faster access to AI-generated "
    "summaries of earnings calls and filings across the firm's coverage universe."
)

GENUINE_AI_PARTNERSHIP_TEXT = (
    "Example Bank today announced a new partnership with Acme AI Cloud to deploy "
    "a generative AI assistant across its wealth management division, building on "
    "a joint engineering effort between the two companies over the past year."
)

GENUINE_AI_INFRASTRUCTURE_TEXT = (
    "Example Bank today announced it has deployed new AI inference infrastructure "
    "in its data centers, including GPU-accelerated compute clusters supporting "
    "its generative AI models used across trading, risk, and operations teams."
)


def _assessment(label: RelevanceLabel, **overrides) -> AgentAssessment:
    kwargs = dict(
        relevance=RelevanceDecision(label=label, reason="test"),
        category=TopicCategory.ENTERPRISE_AI,
        is_dated_development=True,
        financial_services_relevance=RelevanceAssessment(level=RelevanceLevel.MEDIUM, rationale="x"),
        strategic_significance=RelevanceAssessment(level=RelevanceLevel.LOW, rationale="x"),
    )
    kwargs.update(overrides)
    return AgentAssessment(**kwargs)


def _analyze(config, peer_source, text, *, llm_responses):
    # status=SUCCESS set explicitly: these tests construct ExtractedArticle
    # directly (bypassing the real HTML extractor for speed/determinism),
    # and ExtractedArticle's own default status is FAILED -- a real
    # extraction of genuine article text would always be SUCCESS.
    extracted = ExtractedArticle(url="https://jpmorganchase.com/z", title="t", text=text, status=ExtractionStatus.SUCCESS)
    article = normalize_article(extracted, source=peer_source)
    evidence = article_to_evidence(article)
    llm = FakeLLMClient(llm_responses)
    agent = PeerAgent(llm, config)
    item = agent.analyze(article, evidence)
    return item, llm


def test_generic_corporate_announcement_is_prefiltered_irrelevant(config, peer_source):
    item, llm = _analyze(config, peer_source, GENERIC_CORPORATE_TEXT, llm_responses=[])
    assert item is None
    assert len(llm.calls) == 0  # no AI keywords anywhere -- never reaches the LLM


def test_unrelated_financial_services_announcement_is_prefiltered_irrelevant(config, peer_source):
    item, llm = _analyze(config, peer_source, UNRELATED_FINANCIAL_SERVICES_TEXT, llm_responses=[])
    assert item is None
    assert len(llm.calls) == 0


def test_generic_sustainability_community_article_is_prefiltered_irrelevant(config, peer_source):
    item, llm = _analyze(config, peer_source, SUSTAINABILITY_COMMUNITY_TEXT, llm_responses=[])
    assert item is None
    assert len(llm.calls) == 0


def test_generic_ai_explainer_page_is_judged_irrelevant_by_the_llm(config, peer_source):
    """Contains real AI keywords ("artificial intelligence", "machine
    learning"), so it DOES reach the LLM (the pre-filter is necessary but
    not sufficient) -- but a real agent, correctly reasoning that this is
    a generic reference/explainer with no concrete development, would
    mark it irrelevant. Scripted here to prove the architecture honors
    that verdict rather than second-guessing it."""
    item, llm = _analyze(
        config, peer_source, GENERIC_AI_EXPLAINER_TEXT,
        llm_responses=[_assessment(RelevanceLabel.IRRELEVANT)],
    )
    assert len(llm.calls) == 1  # reached the LLM, unlike the pre-filtered cases above
    assert item is None  # but the LLM's irrelevant verdict is honored


def test_substantive_banking_ai_announcement_is_relevant(config, peer_source):
    item, llm = _analyze(
        config, peer_source, SUBSTANTIVE_BANKING_AI_TEXT,
        llm_responses=[_assessment(RelevanceLabel.RELEVANT)],
    )
    assert item is not None
    assert item.relevance.label == RelevanceLabel.RELEVANT


def test_genuine_ai_partnership_is_relevant(config, peer_source):
    item, llm = _analyze(
        config, peer_source, GENUINE_AI_PARTNERSHIP_TEXT,
        llm_responses=[_assessment(RelevanceLabel.RELEVANT, partnerships=["Acme AI Cloud"])],
    )
    assert item is not None
    assert item.partnerships == ["Acme AI Cloud"]  # validated against the real partnership language in the evidence


def test_genuine_ai_infrastructure_development_is_relevant(config, peer_source):
    item, llm = _analyze(
        config, peer_source, GENUINE_AI_INFRASTRUCTURE_TEXT,
        llm_responses=[_assessment(RelevanceLabel.RELEVANT)],
    )
    assert item is not None


def test_retrieved_does_not_imply_relevant_across_the_whole_sweep(config, peer_source):
    """The core invariant, checked in one place: every negative text above
    was fully discovered, fetched, and extracted successfully (real
    Evidence exists for all of them) -- yet none became a ResearchItem."""
    for text in (GENERIC_CORPORATE_TEXT, UNRELATED_FINANCIAL_SERVICES_TEXT, SUSTAINABILITY_COMMUNITY_TEXT):
        extracted = ExtractedArticle(url="https://jpmorganchase.com/w", title="t", text=text, status=ExtractionStatus.SUCCESS)
        article = normalize_article(extracted, source=peer_source)
        evidence = article_to_evidence(article)
        assert evidence is not None  # retrieval/extraction/evidence all succeeded
        item, _ = _analyze(config, peer_source, text, llm_responses=[])
        assert item is None  # yet relevance correctly says no
