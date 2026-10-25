"""OllamaClient: structured output validation, invalid output, unavailable backend,
config-driven model selection. No live Ollama server is used anywhere here."""

from __future__ import annotations

import json

import requests
from pydantic import BaseModel

from research_assistant.agents.schemas import AnalystAssessment
from research_assistant.config import get_config
from research_assistant.llm.client import OllamaClient, create_ollama_client
from research_assistant.llm.errors import LLMResponseError, LLMUnavailableError


class _Schema(BaseModel):
    answer: str
    confidence: int


class _FakeResponse:
    def __init__(self, status_code: int, json_body=None, raise_on_json=False):
        self.status_code = status_code
        self._json_body = json_body
        self._raise_on_json = raise_on_json

    def json(self):
        if self._raise_on_json:
            raise ValueError("not json")
        return self._json_body


class _FakeSession:
    def __init__(self, response_or_exception):
        self._item = response_or_exception
        self.last_call = None

    def post(self, url, json=None, timeout=None):
        self.last_call = {"url": url, "json": json, "timeout": timeout}
        if isinstance(self._item, Exception):
            raise self._item
        return self._item


def _client(session) -> OllamaClient:
    return OllamaClient(base_url="http://localhost:11434", model="qwen3:8b", session=session)


def test_valid_structured_output_is_parsed_and_validated():
    session = _FakeSession(_FakeResponse(200, {"response": '{"answer": "yes", "confidence": 9}'}))
    client = _client(session)

    result = client.generate_structured(system="sys", prompt="p", schema=_Schema)

    assert result.answer == "yes"
    assert result.confidence == 9


def test_non_json_model_output_raises_llm_response_error():
    session = _FakeSession(_FakeResponse(200, {"response": "not actually json"}))
    client = _client(session)

    try:
        client.generate_structured(system="sys", prompt="p", schema=_Schema)
        assert False, "expected LLMResponseError"
    except LLMResponseError:
        pass


def test_model_output_missing_required_fields_raises_llm_response_error():
    session = _FakeSession(_FakeResponse(200, {"response": '{"answer": "yes"}'}))  # missing "confidence"
    client = _client(session)

    try:
        client.generate_structured(system="sys", prompt="p", schema=_Schema)
        assert False, "expected LLMResponseError"
    except LLMResponseError:
        pass


def test_connection_error_raises_llm_unavailable_error():
    session = _FakeSession(requests.ConnectionError("no route to host"))
    client = _client(session)

    try:
        client.generate_structured(system="sys", prompt="p", schema=_Schema)
        assert False, "expected LLMUnavailableError"
    except LLMUnavailableError:
        pass


def test_non_200_status_raises_llm_unavailable_error():
    session = _FakeSession(_FakeResponse(500, {}))
    client = _client(session)

    try:
        client.generate_structured(system="sys", prompt="p", schema=_Schema)
        assert False, "expected LLMUnavailableError"
    except LLMUnavailableError:
        pass


def test_malformed_envelope_raises_llm_response_error():
    session = _FakeSession(_FakeResponse(200, raise_on_json=True))
    client = _client(session)

    try:
        client.generate_structured(system="sys", prompt="p", schema=_Schema)
        assert False, "expected LLMResponseError"
    except LLMResponseError:
        pass


def test_envelope_missing_response_field_raises_llm_response_error():
    session = _FakeSession(_FakeResponse(200, {"not_response": "x"}))
    client = _client(session)

    try:
        client.generate_structured(system="sys", prompt="p", schema=_Schema)
        assert False, "expected LLMResponseError"
    except LLMResponseError:
        pass


def test_request_payload_uses_configured_model_and_schema_derived_format():
    session = _FakeSession(_FakeResponse(200, {"response": '{"answer": "yes", "confidence": 1}'}))
    client = OllamaClient(base_url="http://localhost:11434", model="qwen3:8b", session=session)

    client.generate_structured(system="sys", prompt="p", schema=_Schema)

    assert session.last_call["json"]["model"] == "qwen3:8b"
    assert session.last_call["json"]["format"] == _Schema.model_json_schema()
    assert session.last_call["json"]["stream"] is False


def test_generate_raw_without_a_schema_falls_back_to_bare_json_format():
    """The lower-level `_generate_raw()` still supports the plain `"json"`
    mode for any caller with no Pydantic schema on hand -- `generate_structured`
    itself always has one and never exercises this fallback in practice."""
    session = _FakeSession(_FakeResponse(200, {"response": "{}"}))
    client = OllamaClient(base_url="http://localhost:11434", model="qwen3:8b", session=session)

    client._generate_raw(system="sys", prompt="p")

    assert session.last_call["json"]["format"] == "json"


def test_create_ollama_client_reads_model_and_base_url_from_config():
    config = get_config()
    client = create_ollama_client(config)

    assert client.model == config.settings.system.llm.model == "qwen2.5:3b"
    assert client.base_url == config.settings.system.llm.base_url.rstrip("/")


def test_create_ollama_client_reads_timeout_from_config():
    config = get_config()
    client = create_ollama_client(config)

    assert client.timeout_seconds == config.settings.system.llm.timeout_seconds > 0


def test_create_ollama_client_reflects_env_override(monkeypatch):
    """The env override mechanism itself still works regardless of which
    model config/settings.yaml currently defaults to -- override to a
    different model than the configured default to prove that."""
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3:8b")
    from research_assistant.config.loader import load_config

    config = load_config()
    client = create_ollama_client(config)

    assert client.model == "qwen3:8b"


# ---------------------------------------------------------------------------
# Schema-constrained generation (the qwen2.5:3b schema-adherence fix):
# Ollama's `format` now receives the actual derived JSON Schema for the
# real AnalystAssessment model, never a hand-duplicated second schema.
# ---------------------------------------------------------------------------


def test_the_schema_sent_to_ollama_is_derived_from_the_real_analyst_assessment_model():
    session = _FakeSession(_FakeResponse(200, {"response": "{}"}))
    client = OllamaClient(base_url="http://localhost:11434", model="qwen2.5:3b", session=session)

    try:
        client.generate_structured(system="sys", prompt="p", schema=AnalystAssessment)
    except LLMResponseError:
        pass  # "{}" fails validation -- irrelevant here, we only inspect the outgoing payload

    sent_format = session.last_call["json"]["format"]
    assert sent_format == AnalystAssessment.model_json_schema()
    # Confirms this is the REAL, complete schema (base AgentAssessment fields
    # plus the analyst-specific nested structure), not a hand-duplicated stand-in.
    assert "relevance" in sent_format["properties"]
    assert "category" in sent_format["properties"]
    assert "financial_services_relevance" in sent_format["properties"]
    assert "strategic_significance" in sent_format["properties"]
    assert "analyst_findings" in sent_format["properties"]
    # `nomura_relevance` is deliberately NOT part of the schema sent to the
    # model (see agents/schemas.py's comment) -- it is derived deterministically
    # in agents/base.py instead, to cut one of three expensive free-text
    # rationale paragraphs from every CPU-only qwen2.5:3b deep-analysis call.
    assert "nomura_relevance" not in sent_format["properties"]
    assert set(sent_format["required"]) >= {
        "relevance", "category", "financial_services_relevance",
        "strategic_significance", "analyst_findings",
    }


def test_a_complete_analyst_response_produces_the_expected_typed_object():
    """A fake Ollama response containing the FULL AnalystAssessment shape
    (base fields + nested analyst_findings) round-trips into a correctly
    typed object -- this is what a schema-constrained qwen2.5:3b response
    is expected to look like."""
    complete_response = {
        "relevance": {"label": "relevant", "reason": "Clear, evidence-backed finding on AI in capital markets."},
        "category": "financial_services",
        "event_type": "research",
        "summary": "BCG found banks using generative AI copilots cut research drafting time.",
        "is_dated_development": True,
        "entities": [{"name": "BCG", "type": "company"}],
        "people": [],
        "products": [],
        "deal_value": None,
        "partnerships": [],
        "technologies": ["generative AI", "AI copilots"],
        "geography": ["North America"],
        "financial_services_relevance": {"level": "high", "rationale": "Directly about bank research workflows."},
        "strategic_significance": {"level": "medium", "rationale": "Signals a broader adoption trend."},
        "analyst_findings": {
            "thesis": "Generative AI copilots materially speed up equity research drafting.",
            "key_findings": ["30% reduction in drafting time for early adopters."],
            "evidence_points": [{"statement": "30% reduction in time spent drafting client research notes.", "is_quantitative": True}],
            "industry_pattern": "Broader shift toward AI-assisted research workflows.",
            "financial_implication": "Banks slow to adopt risk falling behind in client advisory.",
            "strategic_themes": ["AI adoption", "AI productivity"],
        },
    }
    session = _FakeSession(_FakeResponse(200, {"response": json.dumps(complete_response)}))
    client = OllamaClient(base_url="http://localhost:11434", model="qwen2.5:3b", session=session)

    result = client.generate_structured(system="sys", prompt="p", schema=AnalystAssessment)

    assert isinstance(result, AnalystAssessment)
    assert result.relevance.label.value == "relevant"
    assert result.category.value == "financial_services"
    assert result.financial_services_relevance.level.value == "high"
    assert result.strategic_significance.level.value == "medium"
    assert result.analyst_findings.thesis == "Generative AI copilots materially speed up equity research drafting."
    assert result.analyst_findings.evidence_points[0].is_quantitative is True
