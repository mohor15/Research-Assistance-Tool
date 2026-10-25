"""Regression fixtures for the CURRENT run `7aab105954d38497` forensic
investigation: two real PR Newswire releases whose publication timestamps
must normalize correctly and fall inside the run's stated CURRENT period
(`2026-10-08 10:04 IST` .. `2026-10-09 10:04 IST`).

Both fixtures mirror the exact real-page pattern (confirmed via live
inspection of the two actual article URLs): PR Newswire's own CMS exposes a
standard `<meta name='date' content="...">` tag carrying an explicit
timezone offset (US Eastern, `-04:00` during EDT) -- already a name this
project's extractor treats as a standard publication-date tag (no new
fallback was needed; this is confirmed, not assumed, by the tests below).
No live network -- fixtures only."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from research_assistant.retrieval.extractor import extract_article
from research_assistant.retrieval.models import ExtractionStatus
from research_assistant.scheduling.period import compute_current_period

_BITS_IN_GLASS_URL = (
    "https://www.prnewswire.com/news-releases/"
    "bits-in-glass-achieves-select-partner-status-in-the-claude-partner-network-services-track-302903136.html"
)
_IQUEST_Q1_URL = (
    "https://www.prnewswire.com/news-releases/"
    "iquest-q1-draws-early-developer-attention-across-coding-and-agentic-workflows-302903211.html"
)

# Real listing timestamp: "Oct 08, 2026, 20:40 ET" -> the page's own meta
# tag expresses this as -04:00 (EDT, confirmed live -- the US is still on
# daylight saving in October).
_BITS_IN_GLASS_HTML = """
<html><head>
<meta name='date' content="2026-10-08T20:40:00-04:00"/>
<title>Bits In Glass achieves Select Partner Status | PR Newswire</title>
</head><body><article>
<h1>Bits In Glass achieves Select Partner Status in the Claude Partner Network Services Track</h1>
<p>DENVER, Oct. 8, 2026 /PRNewswire/ -- Bits In Glass (BIG) has achieved Select partner status in the
Claude Partner Network Services Track. The Services Track recognizes partners for building a certified
delivery team, delivering Claude use cases for clients, and proving the work through public customer
references.</p>
<p>BIG brings over two decades of business process and data automation to applied AI, helping enterprises
move past pilots by putting agentic AI to work in the middle and back office.</p>
</article></body></html>
"""

# Real listing timestamp: "2026-10-09T02:52:00+00:00" -- the page's own
# meta tag expresses the same instant in its local -04:00 offset.
_IQUEST_Q1_HTML = """
<html><head>
<meta name='date' content="2026-10-08T22:52:00-04:00"/>
<title>IQuest-Q1 Draws Early Developer Attention | PR Newswire</title>
</head><body><article>
<h1>IQuest-Q1 Draws Early Developer Attention Across Coding and Agentic Workflows</h1>
<p>BEIJING, Oct. 8, 2026 /PRNewswire/ -- IQuest-Q1 has drawn early praise from developers and industry
watchers since launch, particularly for its performance on coding, software engineering, interactive
application generation, and long-horizon agentic workloads.</p>
<p>IQuest-Q1 was developed by IQuest Research. Developers and research teams interested in participating
in upcoming early-access testing programs can apply for trial access.</p>
</article></body></html>
"""

# A PR Newswire TOPIC HUB page -- a listing of many individually dated
# release cards, never itself a single dated release. Its own page carries
# no article-level `date` meta tag; a neighbouring story card's timestamp
# must never be misattributed to the hub page itself.
_AI_TOPIC_HUB_HTML = """
<html><head>
<title>Artificial Intelligence News | PR Newswire</title>
</head><body>
<div class="card"><h3>Bits In Glass achieves Select Partner Status</h3><span class="date">Oct 08, 2026, 20:40 ET</span></div>
<div class="card"><h3>IQuest-Q1 Draws Early Developer Attention</h3><span class="date">Oct 08, 2026, 22:52 ET</span></div>
<div class="card"><h3>Another AI company announces something</h3><span class="date">Oct 07, 2026, 09:00 ET</span></div>
""" + ("<p>Browse more artificial intelligence press releases.</p>" * 20) + """
</body></html>
"""

# Period from the forensic brief: 2026-10-08 10:04 IST .. 2026-10-09 10:04 IST
_PERIOD_START = datetime(2026, 10, 8, 4, 34, tzinfo=timezone.utc)   # 10:04 IST
_PERIOD_END = datetime(2026, 10, 9, 4, 34, tzinfo=timezone.utc)     # 10:04 IST


def test_bits_in_glass_publication_timestamp_normalizes_correctly():
    result = extract_article(_BITS_IN_GLASS_HTML, _BITS_IN_GLASS_URL)
    assert result.status == ExtractionStatus.SUCCESS
    assert result.published_at is not None
    assert result.published_at == datetime(2026, 10, 9, 0, 40, tzinfo=timezone.utc)


def test_bits_in_glass_publication_timestamp_falls_inside_the_current_period():
    result = extract_article(_BITS_IN_GLASS_HTML, _BITS_IN_GLASS_URL)
    assert _PERIOD_START <= result.published_at < _PERIOD_END


def test_iquest_q1_publication_timestamp_normalizes_correctly():
    result = extract_article(_IQUEST_Q1_HTML, _IQUEST_Q1_URL)
    assert result.status == ExtractionStatus.SUCCESS
    assert result.published_at is not None
    assert result.published_at == datetime(2026, 10, 9, 2, 52, tzinfo=timezone.utc)


def test_iquest_q1_publication_timestamp_falls_inside_the_current_period():
    result = extract_article(_IQUEST_Q1_HTML, _IQUEST_Q1_URL)
    assert _PERIOD_START <= result.published_at < _PERIOD_END


def test_topic_hub_page_does_not_acquire_a_false_date_from_an_embedded_story_card():
    """The hub page itself has no article-level date metadata -- a card's
    visible listing timestamp for a DIFFERENT release must never be
    attributed to the hub page."""
    result = extract_article(
        _AI_TOPIC_HUB_HTML, "https://www.prnewswire.com/news-releases/latest-news-topics/artificial-intelligence/",
    )
    assert result.published_at is None


def test_run_period_boundaries_match_the_forensic_brief_exactly():
    """Confirms the fixture period constants above are the exact
    [period_start, period_end) the real run used -- a pure rolling 24h
    window ending at 2026-10-09 10:04 IST, computed the same way the
    pipeline itself did AT THE TIME of that run (`window_hours=24` pinned
    explicitly here since this documents a specific historical run, not
    CURRENT's present-day default -- which has since widened to 48h, see
    `config.settings.research.current_window_hours`)."""
    now_utc = _PERIOD_END  # 2026-10-09 10:04 IST expressed in UTC
    period_start, period_end = compute_current_period(now_utc, window_hours=24)
    assert period_end - period_start == timedelta(hours=24)
    assert period_start == _PERIOD_START
    assert period_end == _PERIOD_END
    assert period_start <= datetime(2026, 10, 9, 0, 40, tzinfo=timezone.utc) < period_end  # Bits In Glass
    assert period_start <= datetime(2026, 10, 9, 2, 52, tzinfo=timezone.utc) < period_end  # IQuest-Q1
