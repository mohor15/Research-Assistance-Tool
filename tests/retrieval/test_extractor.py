"""Article extraction: normal article, malformed HTML, missing metadata, thin/empty content.

Trafilatura is optional; the trafilatura-specific test auto-skips when it
isn't installed and this sandbox always exercises the BeautifulSoup fallback.
"""

from __future__ import annotations

import pytest

from research_assistant.retrieval.extractor import extract_article, is_trafilatura_available
from research_assistant.retrieval.models import ExtractionMethod, ExtractionStatus

GOOD_ARTICLE_HTML = """
<html>
<head>
  <title>Fallback Title (should not be used)</title>
  <meta property="og:title" content="Bank Launches New AI Copilot for Advisors" />
  <meta name="author" content="Jane Reporter" />
  <meta property="article:published_time" content="2025-09-01T09:00:00Z" />
  <meta property="og:site_name" content="Example Financial News" />
</head>
<body>
  <nav><a href="/">Home</a><a href="/about">About</a></nav>
  <article>
    <p>A major bank today announced the launch of a new AI-powered copilot for
    its financial advisors, aiming to speed up research and client reporting.</p>
    <p>The tool draws on internal research notes and public market data to
    generate draft summaries, which advisors review before sending to clients.
    Executives said the rollout would expand to all regions by early next year,
    with training sessions already underway for pilot teams.</p>
    <p>Analysts said the move reflects a broader industry trend toward
    embedding generative AI directly into daily advisor workflows rather than
    treating it as a standalone tool.</p>
  </article>
  <footer>Copyright 2025</footer>
</body>
</html>
"""

MALFORMED_HTML = "<html><body><article><p>Unclosed paragraph and tag<article></body>"

NO_DATE_NO_AUTHOR_HTML = """
<html><head><title>Some Announcement</title></head>
<body><article>
<p>This announcement has no author or publication date metadata anywhere in
the document, but it does contain a reasonably long body of article text so
that extraction should still be considered a success based on content length
alone, since Phase 1 only requires graceful handling of missing fields.</p>
</article></body></html>
"""

EMPTY_CONTENT_HTML = "<html><head><title>Just Nav</title></head><body><nav><a href='/'>Home</a></nav></body></html>"

JSON_LD_HTML = """
<html><head>
<script type="application/ld+json">
{"@type": "NewsArticle", "datePublished": "2025-08-15T00:00:00Z", "dateModified": "2025-08-16T00:00:00Z"}
</script>
</head><body><article>
<p>Article text sourced primarily from JSON-LD metadata for its publication
and modification dates, since no article:published_time meta tag is present
here, exercising the JSON-LD fallback path in the extractor directly.</p>
</article></body></html>
"""


def test_normal_article_is_extracted_successfully():
    result = extract_article(GOOD_ARTICLE_HTML, "https://example.com/news/ai-copilot")

    assert result.status == ExtractionStatus.SUCCESS
    assert result.title == "Bank Launches New AI Copilot for Advisors"
    assert result.author == "Jane Reporter"
    assert result.publisher == "Example Financial News"
    assert result.published_at is not None
    assert result.published_at.year == 2025
    assert "AI-powered copilot" in result.text
    assert "Home" not in result.text  # nav stripped
    assert "Copyright" not in result.text  # footer stripped


def test_malformed_html_does_not_raise_and_degrades_to_failed_or_partial():
    result = extract_article(MALFORMED_HTML, "https://example.com/broken")
    assert result.status in (ExtractionStatus.PARTIAL, ExtractionStatus.NOT_ARTICLE, ExtractionStatus.FAILED)


def test_missing_publication_date_and_author_do_not_crash_and_text_still_extracted():
    result = extract_article(NO_DATE_NO_AUTHOR_HTML, "https://example.com/announcement")

    assert result.published_at is None
    assert result.author is None
    assert result.status == ExtractionStatus.SUCCESS
    assert result.text is not None


def test_page_with_no_real_content_is_marked_not_article():
    result = extract_article(EMPTY_CONTENT_HTML, "https://example.com/empty")
    assert result.status == ExtractionStatus.NOT_ARTICLE
    assert not result.text


def test_empty_html_is_marked_failed_without_raising():
    result = extract_article("", "https://example.com/nothing")
    assert result.status == ExtractionStatus.FAILED
    assert result.extraction_method == ExtractionMethod.NONE


def test_bs4_fallback_is_used_when_trafilatura_unavailable_or_empty():
    result = extract_article(GOOD_ARTICLE_HTML, "https://example.com/news/ai-copilot")
    if not is_trafilatura_available():
        assert result.extraction_method == ExtractionMethod.BS4_FALLBACK


def test_json_ld_dates_are_used_as_a_fallback():
    result = extract_article(JSON_LD_HTML, "https://example.com/jsonld")
    assert result.published_at is not None
    assert result.published_at.month == 8
    assert result.modified_at is not None


@pytest.mark.skipif(not is_trafilatura_available(), reason="trafilatura not installed in this environment")
def test_trafilatura_backend_extracts_the_same_article():
    result = extract_article(GOOD_ARTICLE_HTML, "https://example.com/news/ai-copilot")
    assert result.extraction_method == ExtractionMethod.TRAFILATURA
    assert result.status == ExtractionStatus.SUCCESS


# ---------------------------------------------------------------------------
# Binary-content guard (Phase 10 hardening fix): a non-HTML asset (PDF,
# image, arbitrary binary) must never reach the bs4 HTML fallback and be
# mistaken for article text.
# ---------------------------------------------------------------------------

# Bytes of a real PDF, decoded as text the way `requests`/the retrieval
# client would hand it to the extractor -- reproduces the exact NVIDIA
# gallery-file bug: binary content masquerading as an HTML string.
PDF_BYTES_AS_TEXT = "%PDF-1.7\n%\xe2\xe3\xcf\xd3\n34 0 obj\n<<>>\nendobj\nxref\n<binary pdf bytes here>"


def test_pdf_response_is_rejected_before_reaching_bs4_fallback():
    result = extract_article(
        PDF_BYTES_AS_TEXT, "https://nvidianews.nvidia.com/_gallery/get_file/?file_id=abc&file_ext=.pdf",
        content_type="application/pdf",
    )

    assert result.status == ExtractionStatus.FAILED
    assert result.extraction_method == ExtractionMethod.NONE
    assert result.metadata["reason"] == "unsupported_content_type"
    assert result.text is None


@pytest.mark.parametrize(
    "content_type",
    ["application/pdf", "image/png", "image/jpeg", "video/mp4", "audio/mpeg", "application/octet-stream"],
)
def test_unsupported_binary_content_types_are_rejected_cleanly(content_type):
    result = extract_article(
        PDF_BYTES_AS_TEXT, "https://example.com/asset", content_type=content_type,
    )

    assert result.status == ExtractionStatus.FAILED
    assert result.extraction_method == ExtractionMethod.NONE
    assert result.metadata["reason"] == "unsupported_content_type"


def test_normal_html_extraction_is_unaffected_by_the_content_type_guard():
    result = extract_article(
        GOOD_ARTICLE_HTML, "https://example.com/news/ai-copilot", content_type="text/html; charset=utf-8",
    )

    assert result.status == ExtractionStatus.SUCCESS
    assert "AI-powered copilot" in result.text


def test_missing_content_type_still_extracts_normal_html_pages_unchanged():
    """No Content-Type header at all (matches every pre-existing test and
    real responses that omit the header) must not change behavior for a
    normal HTML page -- only a recognizably binary URL extension does."""
    result = extract_article(GOOD_ARTICLE_HTML, "https://example.com/news/ai-copilot")

    assert result.status == ExtractionStatus.SUCCESS
    assert "AI-powered copilot" in result.text


def test_missing_content_type_but_binary_url_extension_is_rejected():
    result = extract_article(
        PDF_BYTES_AS_TEXT, "https://nvidianews.nvidia.com/_gallery/get_file.pdf",
    )

    assert result.status == ExtractionStatus.FAILED
    assert result.metadata["reason"] == "unsupported_content_type"
    assert "url_extension=pdf" in result.metadata["detail"]


# ---------------------------------------------------------------------------
# Phase 14 forensic fix 1: a date must be corroborated by SOME tag-level
# value on the page (any <meta>, <time>, or JSON-LD date -- not a fixed
# named list) before being trusted. Reproduces the exact real-world
# regression (an Anthropic report with zero date metadata of any kind,
# where trafilatura mined "a February 2023 investigation" out of the
# BODY PROSE of a report that is actually about 2026 events) and the exact
# real-world success case it must not break (a JPMorganChase page with no
# standard date tag at all, but a real, non-standard
# `alg-search-date` epoch field that DOES corroborate the correct date).
# ---------------------------------------------------------------------------

UNCORROBORATED_DATE_IN_BODY_PROSE_HTML = """
<html><head>
<title>Countering misuse of AI: September 2026</title>
<meta property="og:title" content="Countering misuse of AI: September 2026" />
<meta property="og:description" content="Case studies from threat actors disrupted between December 2025 and August 2026." />
</head>
<body>
<article>
<p>This report documents case studies from threat actors we disrupted between December 2025 and August 2026 across seven areas of harm, including coordinated inauthentic behavior and fraud.</p>
<p>Our findings independently corroborate a February 2023 investigation by a separate journalism network, which first identified the actor behind an earlier scheme.</p>
<p>An individual associated with a foreign intelligence service was active in late 2023, according to open-source reporting reviewed as part of this analysis, well before the campaigns described above began.</p>
<p>We are publishing this report today as part of our ongoing commitment to transparency about misuse of our models and the steps we take to counter it across multiple platforms and regions.</p>
</article>
</body></html>
"""

CORROBORATED_DATE_VIA_NONSTANDARD_META_TAG_HTML = """
<html><head>
<title>Example Corp Launches New Initiative | ExampleCorp</title>
<meta name="alg-search-date" content="1790697480" />
</head>
<body>
<article>
<p>NEW YORK, September 29, 2026 -- Example Corp today announced the launch of a
new initiative to accelerate industrial innovation and strengthen domestic
manufacturing competitiveness, the company said in a statement.</p>
<p>The initiative will bring together manufacturers, researchers and policy
experts to identify opportunities for scaling advanced production techniques
across multiple regions over the coming years.</p>
<p>"This is an important step for the industry," said a company spokesperson,
noting that the program builds on several years of prior investment in similar
efforts across the sector.</p>
</article>
</body></html>
"""


@pytest.mark.skipif(not is_trafilatura_available(), reason="trafilatura not installed in this environment")
def test_an_uncorroborated_date_mined_from_body_prose_is_discarded():
    """Reproduces the exact real Anthropic regression: zero date metadata
    of any kind on the page, and trafilatura's own heuristic mines a date
    mentioned in the article's own historical narrative. Must result in
    published_at=None, never the wrong guessed date."""
    result = extract_article(UNCORROBORATED_DATE_IN_BODY_PROSE_HTML, "https://example.com/threat-report-2026")
    assert result.published_at is None
    assert result.status == ExtractionStatus.SUCCESS  # the article content itself is still extracted fine


@pytest.mark.skipif(not is_trafilatura_available(), reason="trafilatura not installed in this environment")
def test_a_date_corroborated_by_a_nonstandard_meta_tag_is_kept():
    """Reproduces the exact real JPMorganChase success case: no standard
    date tag exists, but a real, non-standard, epoch-valued meta field
    (never named/hardcoded in production code) corroborates the date."""
    result = extract_article(CORROBORATED_DATE_VIA_NONSTANDARD_META_TAG_HTML, "https://example.com/press-releases/example-launch")
    assert result.published_at is not None
    assert result.published_at.date().isoformat() == "2026-09-29"


def test_a_meta_tag_on_a_different_day_does_not_corroborate():
    """Direct unit test of the corroboration helper itself: a tag value
    existing somewhere on the page is not enough on its own -- it must
    agree with the candidate date on the actual calendar day. (Exercised
    directly rather than through the full trafilatura pipeline: real-world
    behavior shows trafilatura's own metadata search already adopts any
    date-shaped meta tag value as its primary candidate before this gate
    ever runs, so a page that has ANY matching-format date tag never
    reaches this helper with a mismatched candidate in practice -- this
    test verifies the guard clause itself stays correct in isolation.)"""
    from bs4 import BeautifulSoup

    from research_assistant.retrieval.extractor import _corroborate_against_page_metadata
    from research_assistant.retrieval.dates import parse_date

    soup = BeautifulSoup(
        '<html><head><meta name="completely-unrelated-field" content="2019-01-01T00:00:00Z" /></head><body></body></html>',
        "lxml",
    )
    candidate = parse_date("2026-09-29")
    assert _corroborate_against_page_metadata(candidate, soup) is None


def test_article_published_time_meta_tag_still_corroborates_as_before():
    """The existing, already-working named-tag path (article:published_time)
    must remain correct after this change."""
    result = extract_article(GOOD_ARTICLE_HTML, "https://example.com/news/ai-copilot")
    assert result.published_at is not None
    assert result.published_at.year == 2025


# ---------------------------------------------------------------------------
# Phase 14 forensic fix 2: a "Site | Section" pipe-style title combined
# with a thin body (< 460 chars, the real observed gap between 10 confirmed
# JPMorganChase hub-shell pages at 361-445 chars and the shortest confirmed
# genuine pipe-titled evergreen page at 496 chars) is reclassified as
# NOT_ARTICLE.
# ---------------------------------------------------------------------------

HUB_SHELL_HTML = """
<html><head>
<title>Press Releases | JPMorganChase</title>
<meta property="og:title" content="Press Releases | JPMorganChase" />
<meta property="og:site_name" content="JPMorganChase" />
</head>
<body><article>
<p>Explore our latest news and announcements from across the firm.</p>
</article></body></html>
"""

LONG_PIPE_TITLED_ARTICLE_HTML = """
<html><head>
<title>Bank Launches New AI Copilot | Example Bank Newsroom</title>
<meta property="og:title" content="Bank Launches New AI Copilot | Example Bank Newsroom" />
<meta property="og:site_name" content="Example Bank Newsroom" />
</head>
<body><article>
<p>A major bank today announced the launch of a new AI-powered copilot for
its financial advisors, aiming to speed up research and client reporting.
The tool draws on internal research notes and public market data to
generate draft summaries, which advisors review before sending to clients.</p>
<p>Executives said the rollout would expand to all regions by early next
year, with training sessions already underway for pilot teams across the
firm's major hubs. Analysts said the move reflects a broader industry trend
toward embedding generative AI directly into daily advisor workflows rather
than treating it as a standalone tool bolted onto existing systems.</p>
<p>The bank said it expects the copilot to materially reduce the time
advisors spend on routine research tasks, freeing them to focus on client
relationships and more complex portfolio decisions over the coming year.</p>
</article></body></html>
"""

SHORT_NO_PIPE_TITLE_HTML = """
<html><head><title>Quick update</title></head>
<body><article><p>Explore our latest news and announcements.</p></article></body></html>
"""


def test_a_thin_pipe_titled_page_is_reclassified_as_not_article():
    result = extract_article(HUB_SHELL_HTML, "https://example.com/newsroom/press-releases")
    assert result.status == ExtractionStatus.NOT_ARTICLE


def test_a_substantive_pipe_titled_article_is_not_reclassified():
    """A pipe in the title alone must never reject a genuine long article."""
    result = extract_article(LONG_PIPE_TITLED_ARTICLE_HTML, "https://example.com/newsroom/ai-copilot-launch")
    assert result.status == ExtractionStatus.SUCCESS


def test_a_thin_page_without_a_pipe_title_is_unaffected_by_this_rule():
    """The thin-body-without-pipe case is governed entirely by the
    pre-existing MIN_SUCCESS_CHARS/MIN_PARTIAL_CHARS thresholds -- this new
    rule must not change its outcome."""
    with_pipe = extract_article(HUB_SHELL_HTML, "https://example.com/newsroom/press-releases")
    without_pipe = extract_article(SHORT_NO_PIPE_TITLE_HTML, "https://example.com/some-other-page")
    # Both are thin; both should already be NOT_ARTICLE via the pre-existing
    # character thresholds -- same ultimate verdict, but the no-pipe one
    # must reach it via the OLD path, not the new pipe-title rule.
    assert with_pipe.status == ExtractionStatus.NOT_ARTICLE
    assert without_pipe.status == ExtractionStatus.NOT_ARTICLE


def test_an_ir_path_article_is_not_rejected_merely_for_its_url(tmp_path=None):
    """Directly addresses the explicit requirement: a legitimate article
    under /ir/ must survive even though several real /ir/ hub pages were
    demonstrated to be hub shells -- the rule keys off title+body shape,
    never the URL path."""
    result = extract_article(LONG_PIPE_TITLED_ARTICLE_HTML, "https://example.com/ir/annual-report/2025/ar-ceo-letters")
    assert result.status == ExtractionStatus.SUCCESS


# ---------------------------------------------------------------------------
# Phase 14 date-recovery fix A: generic JSON-LD `@graph` support. Reproduces
# the exact real HSBC structure (WebSite + Article nodes bundled in one
# `@graph`) and the precedence/edge cases the task requires.
# ---------------------------------------------------------------------------

JSONLD_FLAT_HTML = """
<html><head>
<script type="application/ld+json">
{"@type": "Article", "datePublished": "2026-06-17T00:01:00Z"}
</script>
</head><body><article>
<p>A flat (non-graph) JSON-LD object -- the original, pre-existing, already-
working behavior that must remain unchanged by this fix. Enough body text
to clear the minimum extraction threshold for a successful article.</p>
</article></body></html>
"""

JSONLD_TOPLEVEL_LIST_HTML = """
<html><head>
<script type="application/ld+json">
[{"@type": "Article", "datePublished": "2026-06-17T00:01:00Z"}]
</script>
</head><body><article>
<p>A top-level JSON-LD list -- the original, pre-existing behavior for a
bare list of entities, which must also remain unchanged by this fix. Enough
body text to clear the minimum extraction threshold here too.</p>
</article></body></html>
"""

JSONLD_GRAPH_WEBSITE_THEN_ARTICLE_HTML = """
<html><head>
<script type="application/ld+json">
{"@context": "http://schema.org", "@graph": [
  {"@type": "WebSite", "name": "Example Bank"},
  {"@type": "Article", "headline": "Example Bank and Cloud Partner Co announce AI banking partnership",
   "dateCreated": "2026-04-07T11:51:33Z", "dateModified": "2026-06-17T04:59:19Z",
   "datePublished": "2026-06-17T00:01:00Z"}
]}
</script>
</head><body><article>
<p>Reproduces the exact real HSBC structure: a WebSite node appears before
the Article node in the same @graph array. The Article's own datePublished
must still be recovered. This body text is long enough to be a genuine
successful extraction on its own.</p>
</article></body></html>
"""

JSONLD_GRAPH_UNRELATED_DATE_NOT_SELECTED_HTML = """
<html><head>
<script type="application/ld+json">
{"@graph": [
  {"@type": "Organization", "datePublished": "1999-01-01T00:00:00Z"},
  {"@type": "Article", "datePublished": "2026-06-17T00:01:00Z"}
]}
</script>
</head><body><article>
<p>An unrelated, non-Article graph node ALSO happens to carry a same-named
date key with a clearly wrong value -- the Article node's own date must win
regardless of ordering, never the unrelated node's. Body text long enough
for a successful extraction.</p>
</article></body></html>
"""

JSONLD_GRAPH_NO_DATE_BEARING_ARTICLE_HTML = """
<html><head>
<script type="application/ld+json">
{"@graph": [
  {"@type": "WebSite", "name": "Example Bank"},
  {"@type": "Article", "headline": "An article with no date at all"}
]}
</script>
</head><body><article>
<p>The @graph exists and has an Article node, but neither node carries any
date-bearing key at all -- the result must be None, never a guess. Enough
body text for a successful extraction regardless of the missing date.</p>
</article></body></html>
"""

JSONLD_DATE_PUBLISHED_PREFERRED_OVER_MODIFIED_HTML = """
<html><head>
<script type="application/ld+json">
{"@type": "Article", "datePublished": "2026-06-17T00:01:00Z", "dateModified": "2026-09-30T09:00:00Z"}
</script>
</head><body><article>
<p>Both datePublished and dateModified are present on the same node --
publication date must come from datePublished, never silently replaced by
the newer dateModified value. Body text long enough for success.</p>
</article></body></html>
"""

JSONLD_MALFORMED_HTML = """
<html><head>
<script type="application/ld+json">
{not valid json at all
</script>
</head><body><article>
<p>Malformed JSON-LD must never raise -- it is simply treated as if no
JSON-LD date exists, and extraction of the real body text proceeds
normally regardless. This body is long enough for a successful result.</p>
</article></body></html>
"""

JSONLD_INVALID_DATE_VALUE_HTML = """
<html><head>
<script type="application/ld+json">
{"@type": "Article", "datePublished": "not-a-real-date"}
</script>
</head><body><article>
<p>A JSON-LD date key is present but its value is not a parseable date at
all -- the result must be None, never a crash, and the real article text
is still extracted successfully regardless of the bad date value.</p>
</article></body></html>
"""


def test_1_flat_article_date_published_is_recovered():
    result = extract_article(JSONLD_FLAT_HTML, "https://example.com/a")
    assert result.published_at is not None
    assert result.published_at.date().isoformat() == "2026-06-17"


def test_2_nested_graph_article_date_published_is_recovered():
    result = extract_article(JSONLD_GRAPH_WEBSITE_THEN_ARTICLE_HTML, "https://example.com/b")
    assert result.published_at is not None
    assert result.published_at.date().isoformat() == "2026-06-17"


def test_3_graph_with_website_before_article_still_recovers_the_article_date():
    # Same fixture as test 2 -- explicitly named per the task's own
    # numbered requirement (WebSite node ordered before the Article node).
    result = extract_article(JSONLD_GRAPH_WEBSITE_THEN_ARTICLE_HTML, "https://example.com/b2")
    assert result.published_at.date().isoformat() == "2026-06-17"


def test_4_an_unrelated_graph_node_date_is_not_incorrectly_selected():
    result = extract_article(JSONLD_GRAPH_UNRELATED_DATE_NOT_SELECTED_HTML, "https://example.com/c")
    assert result.published_at is not None
    assert result.published_at.year == 2026  # never the unrelated node's 1999


def test_5_graph_with_no_date_bearing_article_returns_none():
    result = extract_article(JSONLD_GRAPH_NO_DATE_BEARING_ARTICLE_HTML, "https://example.com/d")
    assert result.published_at is None


def test_6_existing_flat_jsonld_behavior_is_unchanged():
    result = extract_article(JSONLD_FLAT_HTML, "https://example.com/a2")
    assert result.status == ExtractionStatus.SUCCESS
    assert result.published_at is not None


def test_7_date_published_is_preferred_over_date_modified():
    result = extract_article(JSONLD_DATE_PUBLISHED_PREFERRED_OVER_MODIFIED_HTML, "https://example.com/e")
    assert result.published_at.date().isoformat() == "2026-06-17"  # not 2026-09-30


def test_8_invalid_date_value_returns_none_without_raising():
    result = extract_article(JSONLD_INVALID_DATE_VALUE_HTML, "https://example.com/f")
    assert result.published_at is None
    assert result.status == ExtractionStatus.SUCCESS  # real article text still extracted


def test_9_toplevel_jsonld_list_behavior_is_intact():
    result = extract_article(JSONLD_TOPLEVEL_LIST_HTML, "https://example.com/g")
    assert result.published_at is not None
    assert result.published_at.date().isoformat() == "2026-06-17"


def test_malformed_jsonld_does_not_raise():
    result = extract_article(JSONLD_MALFORMED_HTML, "https://example.com/h")
    assert result.published_at is None
    assert result.status == ExtractionStatus.SUCCESS


# ---------------------------------------------------------------------------
# Phase 14 date-recovery fix B: narrow, source-aware Bank of America URL
# fallback -- month precision only, never a fabricated day, never applied
# outside the exact BofA press-release URL shape.
# ---------------------------------------------------------------------------

NO_DATE_METADATA_AT_ALL_HTML = """
<html><head><title>Example Corp Announcement</title></head>
<body><article>
<p>This page carries no date metadata of any kind -- no meta tags, no time
tags, no JSON-LD -- reproducing the real, confirmed structure of Bank of
America's press-release pages. The body text itself is long enough to be a
genuine successful extraction on its own merits.</p>
</article></body></html>
"""

CORROBORATED_EXACT_DATE_HTML = """
<html><head>
<meta property="article:published_time" content="2026-09-28T09:00:00Z" />
</head><body><article>
<p>This page DOES carry a real, exact, standard date tag -- the BofA
fallback must never override a genuinely reliable exact date, even if the
URL ALSO happens to look like a BofA press-release URL. Body text long
enough for a successful extraction.</p>
</article></body></html>
"""


def test_10_valid_bofa_press_release_url_recovers_month_level_date():
    result = extract_article(
        NO_DATE_METADATA_AT_ALL_HTML,
        "https://newsroom.bankofamerica.com/content/newsroom/press-releases/2026/09/some-article.html",
    )
    assert result.published_at is not None
    assert result.published_at.date().isoformat() == "2026-09-01"  # day is a placeholder, see precision
    assert result.published_at_precision == "month"


def test_11_bofa_url_with_invalid_month_number_is_not_used():
    result = extract_article(
        NO_DATE_METADATA_AT_ALL_HTML,
        "https://newsroom.bankofamerica.com/content/newsroom/press-releases/2026/13/some-article.html",
    )
    assert result.published_at is None


def test_12_non_bofa_url_with_the_same_yyyy_mm_shape_gets_the_generic_url_fallback():
    """Superseded by the generic, source-agnostic Tier 3 URL-date fallback
    (`_generic_url_date_fallback`): the original narrow BofA-only rule no
    longer defines the ceiling of what this extractor can recover -- ANY
    domain's `/YYYY/MM/` URL shape now recovers a month-precision date
    generically, exactly as this project's publication-date extraction is
    required to work across all sources, not a fixed domain list. The
    BofA-specific fallback below this one in precedence is unaffected and
    now simply redundant for this exact shape."""
    result = extract_article(
        NO_DATE_METADATA_AT_ALL_HTML,
        "https://example.com/content/newsroom/press-releases/2026/09/some-article.html",
    )
    assert result.published_at is not None
    assert result.published_at.date().isoformat() == "2026-09-01"
    assert result.published_at_precision == "month"


def test_13_bofa_month_fallback_never_fabricates_a_specific_day():
    result = extract_article(
        NO_DATE_METADATA_AT_ALL_HTML,
        "https://newsroom.bankofamerica.com/content/newsroom/press-releases/2026/09/some-article.html",
    )
    assert result.published_at_precision == "month"  # never silently "exact"


def test_14_a_real_exact_metadata_date_overrides_the_bofa_url_fallback():
    result = extract_article(
        CORROBORATED_EXACT_DATE_HTML,
        "https://newsroom.bankofamerica.com/content/newsroom/press-releases/2026/09/some-article.html",
    )
    assert result.published_at.date().isoformat() == "2026-09-28"  # the real tag, not the URL's month
    # Date-precedence fix (CURRENT run `e818e08adc2e4dd1`): the trafilatura
    # backend now always reports a precision hint for any date it finds
    # (never left unset/None as before) -- trafilatura's own metadata
    # engine normalizes this tag's value to a bare calendar date
    # internally (losing the "T09:00:00Z" time this fixture's own
    # `<meta>` tag carries), and with no JSON-LD on this page to recover
    # the fuller timestamp from, "date_only" is the honest, evidence-
    # grounded precision for what was actually used -- never silently
    # "exact". The BofA month-level URL fallback still never fires here
    # either way (a real metadata date, whatever its precision, always
    # takes priority over it).
    assert result.published_at_precision == "date_only"


def test_a_bofa_stories_url_also_recovers_month_level_date():
    """Test A: BofA's 'stories' content section (narrative/commentary
    pieces, e.g. shareholder letters) uses the same <YYYY>/<MM>/ URL
    convention as press-releases and also carries zero date metadata --
    confirmed via the real article
    newsroom.bankofamerica.com/content/newsroom/stories/2026/03/
    a-letter-to-shareholders-from-chair-and-ceo--brian-moynihan.html."""
    result = extract_article(
        NO_DATE_METADATA_AT_ALL_HTML,
        "https://newsroom.bankofamerica.com/content/newsroom/stories/2026/03/"
        "a-letter-to-shareholders-from-chair-and-ceo--brian-moynihan.html",
    )
    assert result.published_at is not None
    assert result.published_at.date().isoformat() == "2026-03-01"
    assert result.published_at_precision == "month"


def test_b_bofa_press_releases_url_still_recovers_month_level_date():
    """Test B: the existing press-releases behavior is unchanged."""
    result = extract_article(
        NO_DATE_METADATA_AT_ALL_HTML,
        "https://newsroom.bankofamerica.com/content/newsroom/press-releases/2026/03/some-article.html",
    )
    assert result.published_at is not None
    assert result.published_at.date().isoformat() == "2026-03-01"
    assert result.published_at_precision == "month"


def test_c_unrelated_bofa_path_gets_no_fallback():
    """Test C: a BofA page outside both the press-releases and stories
    sections (e.g. the 'about' section) must never get a fabricated date."""
    result = extract_article(
        NO_DATE_METADATA_AT_ALL_HTML,
        "https://newsroom.bankofamerica.com/about/our-company/overview.html",
    )
    assert result.published_at is None


def test_d_bofa_stories_fallback_never_promotes_to_exact_day():
    """Test D: the recovered date stays MONTH precision -- the real
    'March 23, 2026' visible in the article body is never parsed or
    promoted to an exact day by this fallback."""
    result = extract_article(
        NO_DATE_METADATA_AT_ALL_HTML,
        "https://newsroom.bankofamerica.com/content/newsroom/stories/2026/03/"
        "a-letter-to-shareholders-from-chair-and-ceo--brian-moynihan.html",
    )
    assert result.published_at.date().isoformat() == "2026-03-01"  # month placeholder, not day 23
    assert result.published_at_precision == "month"


def test_bofa_narrow_fallback_requires_a_two_digit_month_segment_but_generic_url_tier_still_recovers_it():
    """The narrow, domain-gated BofA regex itself still requires a two-digit
    month segment (unchanged). But the new generic Tier 3 URL fallback
    (`_generic_url_date_fallback`, domain-independent, `\\d{1,2}` month)
    now recovers a month-precision date from this single-digit-month path
    regardless -- exactly the generic behavior the extraction system is
    required to provide across all sources, not a BofA-only special case."""
    result = extract_article(
        NO_DATE_METADATA_AT_ALL_HTML,
        "https://newsroom.bankofamerica.com/content/newsroom/press-releases/2026/9/some-article.html",
    )
    assert result.published_at is not None
    assert result.published_at.date().isoformat() == "2026-09-01"
    assert result.published_at_precision == "month"


# pwc.com's own CMS exposes a non-standard `pwcReleaseDate` meta tag --
# confirmed via live inspection of several pwc.com pages (2026-10-04) --
# distinct from its sibling `pwcLastModifiedDate`, a MODIFIED date that must
# never be read as a publication date.
PWC_RELEASE_DATE_HTML = """
<html><head><title>PwC and Example Corp: a transformation advisor</title>
<meta name="pwcReleaseDate" content="2025-10-15T01:00:00.000-03:00" />
<meta name="pwcLastModifiedDate" content="2026-09-22T19:05:27.444Z" />
</head><body><article>
<p>This page carries PwC's own non-standard pwcReleaseDate meta tag, not
any of the standard date meta-tag names any existing check recognizes.
Body text long enough for a successful extraction on its own merits
regardless of date metadata.</p>
</article></body></html>
"""

NO_PWC_META_AT_ALL_HTML = """
<html><head><title>PwC services page</title></head>
<body><article>
<p>This page carries no date metadata of any kind -- no pwcReleaseDate, no
meta tags, no time tags, no JSON-LD. Body text long enough for a genuine
successful extraction on its own merits.</p>
</article></body></html>
"""


def test_pwc_release_date_meta_tag_is_recovered_as_publication_date():
    result = extract_article(PWC_RELEASE_DATE_HTML, "https://www.pwc.com/us/en/some-page.html")
    assert result.published_at is not None
    assert result.published_at.date().isoformat() == "2025-10-15"


def test_pwc_last_modified_date_is_never_used_as_publication_date():
    """`pwcLastModifiedDate` on the SAME page as `pwcReleaseDate` must never
    leak into `published_at` -- only the release date is a publication
    date; the modified date is a separate concept."""
    result = extract_article(PWC_RELEASE_DATE_HTML, "https://www.pwc.com/us/en/some-page.html")
    assert result.published_at.date().isoformat() != "2026-09-22"


def test_a_pwc_page_with_no_release_date_meta_tag_gets_no_fallback():
    result = extract_article(NO_PWC_META_AT_ALL_HTML, "https://www.pwc.com/us/en/some-other-page.html")
    assert result.published_at is None


def test_pwc_release_date_fallback_is_not_applied_to_an_unrelated_source():
    """The fallback checks the HTML for the tag, not the domain -- but it
    must never fire merely because a page happens to be served from
    pwc.com; it fires only when the tag is actually present. This proves
    the converse never silently breaks: a non-pwc page with no such tag
    still gets published_at=None, same as before this fix existed."""
    result = extract_article(NO_PWC_META_AT_ALL_HTML, "https://example.com/some-other-page.html")
    assert result.published_at is None


def test_a_real_exact_metadata_date_overrides_the_pwc_release_date_fallback():
    html = CORROBORATED_EXACT_DATE_HTML.replace(
        "</head>", '<meta name="pwcReleaseDate" content="2020-01-01T00:00:00Z" /></head>'
    )
    result = extract_article(html, "https://www.pwc.com/us/en/some-page.html")
    assert result.published_at.date().isoformat() == "2026-09-28"  # the real tag, not pwcReleaseDate


def test_ambiguous_content_type_with_no_recognizable_extension_passes_through():
    """Anything not explicitly known to be binary (e.g. a generic or
    missing Content-Type on a normal-looking URL) must fall through to the
    existing extractors unchanged -- this guard is conservative, not a
    broad allowlist."""
    result = extract_article(GOOD_ARTICLE_HTML, "https://example.com/news/ai-copilot", content_type="text/plain")

    assert result.status == ExtractionStatus.SUCCESS
