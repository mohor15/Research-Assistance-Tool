"""Date-extraction precedence: JSON-LD, extended HTML metadata, <time>
elements, visible-article labelled text, embedded hydration/application
state, and publication-vs-modification-date precedence. Generic fallbacks
only -- no article/source-specific hacks. Confirmed real-production
regressions (2026-10-07 live CURRENT run `2a55bf02ff1b7d1f`): a live
Deloitte page's `datePublished` lived inside a non-JSON-LD `dataLayer.page`
tracking payload, and a live Anthropic page carried no trustworthy
publication date at all. Fixtures only -- no live network."""

from __future__ import annotations

from datetime import datetime, timezone

from research_assistant.retrieval.extractor import extract_article
from research_assistant.retrieval.models import ExtractionStatus

_URL = "https://example.test/content/article-1"
_BODY = "<article><p>" + ("Substantive article body text. " * 30) + "</p></article>"


def _page(head: str = "", body: str = _BODY) -> str:
    return f"<html><head>{head}</head><body>{body}</body></html>"


# ---------------------------------------------------------------------------
# A. JSON-LD datePublished, including nested/array structures
# ---------------------------------------------------------------------------


def test_jsonld_date_published_simple():
    html = _page(head="""
    <script type="application/ld+json">
    {"@type": "Article", "datePublished": "2026-10-05T09:00:00Z"}
    </script>
    """)
    result = extract_article(html, _URL)
    assert result.published_at is not None
    assert result.published_at.date().isoformat() == "2026-10-05"


def test_jsonld_date_published_nested_in_graph_array():
    html = _page(head="""
    <script type="application/ld+json">
    {"@graph": [
        {"@type": "WebSite", "name": "Example"},
        {"@type": "NewsArticle", "datePublished": "2026-10-05T09:00:00Z"}
    ]}
    </script>
    """)
    result = extract_article(html, _URL)
    assert result.published_at is not None
    assert result.published_at.date().isoformat() == "2026-10-05"


def test_jsonld_prefers_date_published_over_date_modified():
    html = _page(head="""
    <script type="application/ld+json">
    {"@type": "Article", "datePublished": "2026-10-05T09:00:00Z", "dateModified": "2026-10-06T10:00:00Z"}
    </script>
    """)
    result = extract_article(html, _URL)
    assert result.published_at is not None
    assert result.published_at.date().isoformat() == "2026-10-05"


def test_jsonld_modified_date_without_published_is_not_used_as_publication_date():
    html = _page(head="""
    <script type="application/ld+json">
    {"@type": "Article", "dateModified": "2026-10-06T10:00:00Z"}
    </script>
    """)
    result = extract_article(html, _URL)
    assert result.published_at is None


# ---------------------------------------------------------------------------
# B. Standard + extended HTML metadata tag names
# ---------------------------------------------------------------------------


def test_article_published_time_meta():
    html = _page(head='<meta property="article:published_time" content="2026-10-05T09:00:00Z" />')
    result = extract_article(html, _URL)
    assert result.published_at is not None
    assert result.published_at.date().isoformat() == "2026-10-05"


def test_extended_meta_date_published_name():
    html = _page(head='<meta name="datePublished" content="2026-10-05T09:00:00Z" />')
    result = extract_article(html, _URL)
    assert result.published_at is not None
    assert result.published_at.date().isoformat() == "2026-10-05"


def test_extended_meta_dc_date_issued():
    html = _page(head='<meta name="dc.date.issued" content="2026-10-05" />')
    result = extract_article(html, _URL)
    assert result.published_at is not None
    assert result.published_at.date().isoformat() == "2026-10-05"


def test_extended_meta_parsely_pub_date():
    html = _page(head='<meta name="parsely-pub-date" content="2026-10-05T09:00:00Z" />')
    result = extract_article(html, _URL)
    assert result.published_at is not None
    assert result.published_at.date().isoformat() == "2026-10-05"


# ---------------------------------------------------------------------------
# C. <time datetime=...> elements
# ---------------------------------------------------------------------------


def test_time_element_with_datetime_attribute():
    html = _page(body='<article><time datetime="2026-10-05T09:00:00Z"></time><p>' + "x" * 300 + "</p></article>")
    result = extract_article(html, _URL)
    assert result.published_at is not None
    assert result.published_at.date().isoformat() == "2026-10-05"


# ---------------------------------------------------------------------------
# D. Visible, labelled article-header publication date
# ---------------------------------------------------------------------------


def test_visible_labelled_published_date_is_recovered():
    html = _page(body=f"<article><p>Published: October 5, 2026</p><p>{'x' * 300}</p></article>")
    result = extract_article(html, _URL)
    assert result.published_at is not None
    assert result.published_at.date().isoformat() == "2026-10-05"


def test_unlabelled_date_in_body_prose_is_never_used():
    """A date mentioned in the article's own body prose (e.g. referencing a
    past event) with NO publication-label word nearby must never be
    mistaken for the publication date."""
    html = _page(body=(
        "<article><p>Back in March 15, 2019 the company first launched its platform. "
        + "x" * 300 + "</p></article>"
    ))
    assert extract_article(html, _URL).published_at is None


def test_copyright_footer_date_is_never_used_as_publication_date():
    html = _page(body=f"<article><p>{'x' * 300}</p></article><footer>&copy; 2015 Example Corp</footer>")
    assert extract_article(html, _URL).published_at is None


# ---------------------------------------------------------------------------
# E. OpenGraph / equivalent metadata (already-existing path, reconfirmed)
# ---------------------------------------------------------------------------


def test_opengraph_site_name_does_not_interfere_with_date_extraction():
    html = _page(head=(
        '<meta property="og:site_name" content="Example News" />'
        '<meta property="article:published_time" content="2026-10-05T09:00:00Z" />'
    ))
    result = extract_article(html, _URL)
    assert result.published_at is not None
    assert result.published_at.date().isoformat() == "2026-10-05"


# ---------------------------------------------------------------------------
# F. Embedded hydration/application-state publication date
# ---------------------------------------------------------------------------


def test_hydration_state_date_published_key():
    html = _page(body=(
        '<article><p>' + "x" * 300 + '</p></article>'
        '<script type="text/javascript">var dataLayer = {}; dataLayer.page = '
        '{"createDate": "14 Jan 2026", "dateModified": "23 Apr 2026", "datePublished": "20 Jan 2026"};</script>'
    ))
    result = extract_article(html, _URL)
    assert result.published_at is not None
    assert result.published_at.date().isoformat() == "2026-01-20"  # datePublished, never createDate/dateModified


def test_hydration_state_falls_back_to_create_date_only_if_no_published_key_exists():
    html = _page(body=(
        '<article><p>' + "x" * 300 + '</p></article>'
        '<script type="text/javascript">var dataLayer = {}; dataLayer.page = '
        '{"createDate": "14 Jan 2026", "dateModified": "23 Apr 2026"};</script>'
    ))
    result = extract_article(html, _URL)
    assert result.published_at is not None
    assert result.published_at.date().isoformat() == "2026-01-14"


def test_hydration_state_modified_only_key_never_used_alone():
    html = _page(body=(
        '<article><p>' + "x" * 300 + '</p></article>'
        '<script type="text/javascript">var dataLayer = {}; dataLayer.page = '
        '{"dateModified": "23 Apr 2026"};</script>'
    ))
    assert extract_article(html, _URL).published_at is None


# ---------------------------------------------------------------------------
# GA4 "PublishDate" custom-analytics-event call (generic, not NVIDIA-specific
# -- matched purely by the semantic event name "PublishDate")
# ---------------------------------------------------------------------------


def test_ga4_publish_date_single_quotes():
    html = _page(body=(
        '<article><p>' + "x" * 300 + '</p></article>'
        "<script>ga_four_event('PublishDate','View','September 28, 2026');</script>"
    ))
    result = extract_article(html, _URL)
    assert result.published_at is not None
    assert result.published_at.date().isoformat() == "2026-09-28"


def test_ga4_publish_date_double_quotes():
    html = _page(body=(
        '<article><p>' + "x" * 300 + '</p></article>'
        '<script>ga_four_event("PublishDate", "View", "September 28, 2026");</script>'
    ))
    result = extract_article(html, _URL)
    assert result.published_at is not None
    assert result.published_at.date().isoformat() == "2026-09-28"


def test_ga4_publish_date_extra_whitespace():
    html = _page(body=(
        '<article><p>' + "x" * 300 + '</p></article>'
        "<script>ga_four_event( 'PublishDate' , 'View' , 'September 28, 2026' );</script>"
    ))
    result = extract_article(html, _URL)
    assert result.published_at is not None
    assert result.published_at.date().isoformat() == "2026-09-28"


def test_ga4_other_event_category_is_never_mistaken_for_publish_date():
    """Only an event whose first argument is exactly "PublishDate"
    qualifies -- an unrelated GA4 event category (e.g. a page view) must
    never be read as a publication date."""
    html = _page(body=(
        '<article><p>' + "x" * 300 + '</p></article>'
        "<script>ga_four_event('PageView','Load','September 28, 2026');</script>"
    ))
    assert extract_article(html, _URL).published_at is None


def test_ga4_publish_date_loses_to_a_stronger_jsonld_date():
    """Precedence: JSON-LD `datePublished` is checked before the GA4
    fallback and must win if both are present."""
    html = _page(head="""
    <script type="application/ld+json">
    {"@type": "Article", "datePublished": "2026-10-05T09:00:00Z"}
    </script>
    """, body=(
        '<article><p>' + "x" * 300 + '</p></article>'
        "<script>ga_four_event('PublishDate','View','September 28, 2026');</script>"
    ))
    result = extract_article(html, _URL)
    assert result.published_at is not None
    assert result.published_at.date().isoformat() == "2026-10-05"


# ---------------------------------------------------------------------------
# D2. Enhanced Tier 2 content-date formats: labelled dates with a trailing
# time/timezone, slash-separated numeric dates, and an unlabelled but
# structurally-isolated header timestamp.
# ---------------------------------------------------------------------------


def test_labelled_date_with_time_and_timezone_abbreviation_is_recovered():
    html = _page(body=f"<article><p>Published on Oct 09, 2026, 10:00 ET</p><p>{'x' * 300}</p></article>")
    result = extract_article(html, _URL)
    assert result.published_at is not None
    assert result.published_at.date().isoformat() == "2026-10-09"
    assert result.published_at_precision == "exact"


def test_labelled_date_only_has_date_only_precision():
    html = _page(body=f"<article><p>Published: October 5, 2026</p><p>{'x' * 300}</p></article>")
    result = extract_article(html, _URL)
    assert result.published_at is not None
    assert result.published_at_precision == "date_only"


def test_labelled_slash_separated_numeric_date_is_recovered():
    html = _page(body=f"<article><p>Publication date: 2026/10/09</p><p>{'x' * 300}</p></article>")
    result = extract_article(html, _URL)
    assert result.published_at is not None
    assert result.published_at.date().isoformat() == "2026-10-09"


def test_unlabelled_isolated_header_timestamp_is_recovered():
    """A short, standalone <time> node sitting before the article's own
    body prose, with no label word, is recognized purely by its structural
    isolation/position."""
    html = _page(body=(
        "<article><h1>Example Co. Launches New Platform</h1>"
        "<time>October 9, 2026, 10:00 ET</time>"
        f"<p>{'x' * 300}</p></article>"
    ))
    result = extract_article(html, _URL)
    assert result.published_at is not None
    assert result.published_at.date().isoformat() == "2026-10-09"


def test_unlabelled_date_inside_a_long_paragraph_is_never_used_as_header():
    """The same bare date shape, but embedded inside a long paragraph of
    real prose (not an isolated short node) -- must never be mistaken for
    an unlabelled header timestamp."""
    html = _page(body=(
        "<article><p>"
        + "October 9, 2026 was a day the company will remember. " + ("Filler sentence. " * 30)
        + "</p></article>"
    ))
    assert extract_article(html, _URL).published_at is None


def test_unlabelled_header_date_never_fires_after_body_prose_has_started():
    """A date-only node appearing AFTER a substantial paragraph has already
    started the article body must not be read as a header timestamp --
    only a node preceding the body prose qualifies."""
    html = _page(body=(
        "<article><p>" + ("Real article body content. " * 20) + "</p>"
        "<p>October 9, 2026</p>"
        f"<p>{'x' * 300}</p></article>"
    ))
    assert extract_article(html, _URL).published_at is None


# ---------------------------------------------------------------------------
# D3. Tier 3: generic, source-agnostic URL-path date extraction
# ---------------------------------------------------------------------------


def test_url_full_year_month_day_path_is_recovered():
    html = _page()
    result = extract_article(html, "https://example.test/news/2026/10/09/example-article")
    assert result.published_at is not None
    assert result.published_at.date().isoformat() == "2026-10-09"
    assert result.published_at_precision == "date_only"


def test_url_iso_style_date_in_path_is_recovered():
    html = _page()
    result = extract_article(html, "https://example.test/news/2026-10-09/example-article")
    assert result.published_at is not None
    assert result.published_at.date().isoformat() == "2026-10-09"


def test_url_compact_date_path_is_recovered():
    html = _page()
    result = extract_article(html, "https://example.test/20261009/example-article")
    assert result.published_at is not None
    assert result.published_at.date().isoformat() == "2026-10-09"
    assert result.published_at_precision == "date_only"


def test_url_partial_year_month_path_is_month_precision():
    html = _page()
    result = extract_article(html, "https://example.test/2026/10/example-article")
    assert result.published_at is not None
    assert result.published_at.date().isoformat() == "2026-10-01"
    assert result.published_at_precision == "month"


def test_url_full_date_preferred_over_partial_year_month():
    html = _page()
    result = extract_article(html, "https://example.test/2026/10/09/example-article")
    assert result.published_at_precision == "date_only"
    assert result.published_at.date().isoformat() == "2026-10-09"


def test_url_ambiguous_numeric_article_id_is_not_treated_as_a_date():
    """A long, non-date-shaped numeric article id (as PR Newswire and
    similar publishers commonly append) must never be read as a date --
    it isn't bounded by path separators into a YYYY/MM/DD or YYYYMMDD
    shape."""
    html = _page()
    result = extract_article(
        html, "https://example.test/news-releases/example-article-302903455.html",
    )
    assert result.published_at is None


def test_url_query_and_tracking_parameters_are_never_read_as_a_date():
    html = _page()
    result = extract_article(
        html, "https://example.test/news/example-article?utm_date=20261009&ref=20261009",
    )
    assert result.published_at is None


def test_url_fallback_is_not_used_when_structured_metadata_already_found_a_date():
    """Tier 1/2 sources always win over the generic URL fallback."""
    html = _page(head='<meta property="article:published_time" content="2026-01-01T00:00:00Z" />')
    result = extract_article(html, "https://example.test/news/2026/10/09/example-article")
    assert result.published_at is not None
    assert result.published_at.date().isoformat() == "2026-01-01"


def test_url_derived_date_never_gains_an_invented_time():
    html = _page()
    result = extract_article(html, "https://example.test/news/2026/10/09/example-article")
    assert result.published_at == datetime(2026, 10, 9, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# D4. Real regression fixtures: KONST, PUFsecurity, Senao Networks
# (2026-10-12 forensic investigation) -- publication-date extraction must
# succeed independently of each article's `is_dated_development`
# classification (KONST/PUFsecurity are both undated developments per the
# separate guardrail; Senao is dated -- all three have real, extractable
# publication dates regardless).
# ---------------------------------------------------------------------------


def test_konst_series_b_release_publication_date_is_extracted():
    html = _page(head="""
    <script type="application/ld+json">
    {"@type": "NewsArticle", "datePublished": "2026-10-09T15:00:00Z"}
    </script>
    """, body=(
        "<article><h1>KONST Raises $30M Series B funding Led by ADATA Technology</h1>"
        "<p>TAIPEI, Oct. 9, 2026 /PRNewswire/ -- KONST, a Taiwan-based next-generation AI "
        "compute operator, announced that it has raised $30 million Series B funding led by "
        "ADATA Technology (TPEx: 3260). " + ("Additional detail about the round. " * 10)
        + "</p></article>"
    ))
    result = extract_article(html, "https://www.prnewswire.com/news-releases/konst-raises-30m-series-b-funding-302903455.html")
    assert result.published_at is not None
    assert result.published_at.date().isoformat() == "2026-10-09"


def test_pufsecurity_silicon_release_publication_date_is_extracted():
    html = _page(head="""
    <script type="application/ld+json">
    {"@type": "NewsArticle", "datePublished": "2026-10-09T14:00:00Z"}
    </script>
    """, body=(
        "<article><h1>Trusted AI Agents Must Be Anchored in Silicon</h1>"
        "<p>SAN JOSE, Calif., Oct. 9, 2026 /PRNewswire/ -- The first wave of generative AI "
        "answered questions and produced content. " + ("PUFsecurity explains further detail. " * 10)
        + "</p></article>"
    ))
    result = extract_article(html, "https://www.prnewswire.com/news-releases/trusted-ai-agents-must-be-anchored-in-silicon-302903324.html")
    # Publication date is extracted regardless of this article's (separately
    # investigated, correctly-undated) `is_dated_development` classification.
    assert result.published_at is not None
    assert result.published_at.date().isoformat() == "2026-10-09"


def test_senao_networks_launch_publication_date_is_extracted():
    html = _page(head="""
    <script type="application/ld+json">
    {"@type": "NewsArticle", "datePublished": "2026-10-09T16:00:00Z"}
    </script>
    """, body=(
        "<article><h1>Senao Networks Launches x86 Enterprise Server Series for Enterprise AI</h1>"
        "<p>TAIPEI, Oct. 9, 2026 /PRNewswire/ -- Senao Networks Inc. today announced the launch "
        "of its new x86 enterprise server portfolio. " + ("Additional product detail. " * 10)
        + "</p></article>"
    ))
    result = extract_article(html, "https://www.prnewswire.com/news-releases/senao-networks-launches-x86-enterprise-server-series-302902638.html")
    assert result.published_at is not None
    assert result.published_at.date().isoformat() == "2026-10-09"


def test_historical_date_in_content_differs_from_url_derived_publication_date():
    """An article whose body mentions a past, unrelated historical date
    must still recover its own publication date from the URL -- the
    historical date in the prose is never mistaken for it."""
    html = _page(body=(
        "<article><p>Back in March 15, 2019 the company first launched its original platform. "
        + ("Today the company is expanding that platform further. " * 15)
        + "</p></article>"
    ))
    result = extract_article(html, "https://example.test/news/2026/10/09/example-expansion-article")
    assert result.published_at is not None
    assert result.published_at.date().isoformat() == "2026-10-09"  # the URL's date, not the body's 2019 mention


def test_nvidia_regression_fixture_ga4_publish_date():
    """Regression fixture (2026-10-07 CURRENT run): the real NVIDIA
    newsroom page at https://nvidianews.nvidia.com/news/open-agent-safety-platform
    carries NO JSON-LD, meta-tag, or hydration-object date anywhere -- its
    only date signal is this exact GA4 analytics call. Matched generically
    by the "PublishDate" event name, never by this page's title/domain."""
    html = _page(body=(
        "<article><p>NVIDIA today announced NVIDIA Open Agent Safety Platform. "
        + "x" * 300 + "</p></article>"
        "<script>ga_four_event('PublishDate','View','September 28, 2026');</script>"
    ))
    result = extract_article(html, "https://nvidianews.nvidia.com/news/open-agent-safety-platform")
    assert result.published_at is not None
    assert result.published_at.date().isoformat() == "2026-09-28"


# ---------------------------------------------------------------------------
# Timezone / precision normalization
# ---------------------------------------------------------------------------


def test_timezone_offset_is_normalized_to_utc():
    html = _page(head='<meta property="article:published_time" content="2026-10-05T09:00:00-04:00" />')
    result = extract_article(html, _URL)
    assert result.published_at is not None
    assert result.published_at.tzinfo is not None  # normalized to a timezone-aware UTC datetime


def test_date_only_value_has_no_invented_time():
    html = _page(head='<meta property="article:published_time" content="2026-10-05" />')
    result = extract_article(html, _URL)
    assert result.published_at == datetime(2026, 10, 5, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Completely undated / malformed date
# ---------------------------------------------------------------------------


def test_completely_undated_page_yields_no_date():
    html = _page()
    result = extract_article(html, _URL)
    assert result.published_at is None
    assert result.status in (ExtractionStatus.SUCCESS, ExtractionStatus.PARTIAL, ExtractionStatus.NOT_ARTICLE)


def test_malformed_date_value_is_rejected_not_guessed():
    html = _page(head='<meta property="article:published_time" content="not-a-real-date" />')
    assert extract_article(html, _URL).published_at is None


def test_multiple_structured_dates_published_wins_over_modified_across_sources():
    """A page with both a (weaker) hydration-state date AND a (stronger)
    JSON-LD datePublished -- the stronger, higher-precedence source wins."""
    html = _page(head="""
    <script type="application/ld+json">
    {"@type": "Article", "datePublished": "2026-10-05T09:00:00Z"}
    </script>
    """, body=(
        '<article><p>' + "x" * 300 + '</p></article>'
        '<script type="text/javascript">var dataLayer = {}; dataLayer.page = '
        '{"datePublished": "2020-01-01"};</script>'
    ))
    result = extract_article(html, _URL)
    assert result.published_at is not None
    assert result.published_at.date().isoformat() == "2026-10-05"


# ---------------------------------------------------------------------------
# Precision/precedence fix (forensic finding, CURRENT run `e818e08adc2e4dd1`):
# trafilatura's own metadata date is frequently a bare calendar date with no
# time-of-day, while the SAME page's own JSON-LD `datePublished` often
# carries a full, timezone-aware timestamp that was previously always
# discarded once trafilatura found *anything* at all. Six real PR Newswire
# releases (HeyGen, Lean SuperIntelligence, TestMu, KONST, Pine AI, Driven
# Tech) lost their real publish time this way, collapsing to midnight UTC
# and failing CURRENT's zero-slack 24h wide-lookback filter before ever
# reaching the DB. Fixtures below are modeled on the real cached HTML these
# releases served (same shape as `tests/retrieval/test_pr_newswire_
# discovery.py`'s JumpCloud fixture, which exercises the exact same
# forensic pattern end-to-end) -- generic (driven only by the presence of a
# time component in each candidate STRING), never a publisher-specific hack.
# ---------------------------------------------------------------------------

_PR_NEWSWIRE_SHAPED_BODY = (
    "<article><h1>Example AI Co. Launches New Platform</h1>"
    "<p>SAN FRANCISCO, Oct. 9, 2026 /PRNewswire/ -- Example AI Co. today announced "
    "the launch of a new platform for enterprise customers. " + ("The platform brings together several new capabilities. " * 10)
    + "</p></article>"
)


def _pr_newswire_shaped_page(date_published: str, date_modified: str) -> str:
    return _page(head=f"""
    <script type="application/ld+json">
    {{"@context": "https://schema.org", "@type": "NewsArticle",
      "headline": "Example AI Co. Launches New Platform",
      "datePublished": "{date_published}", "dateModified": "{date_modified}",
      "publisher": {{"@type": "Organization", "name": "Cision PR Newswire"}}}}
    </script>
    """, body=_PR_NEWSWIRE_SHAPED_BODY)


def test_1_precise_jsonld_timestamp_wins_over_trafilaturas_date_only_value():
    """The core forensic regression: trafilatura's own metadata heuristic
    on a real PR-Newswire-shaped page normalizes to a bare calendar date,
    but the page's own JSON-LD carries the real time-of-day -- the precise
    value must be preserved, not silently collapsed to midnight UTC."""
    html = _pr_newswire_shaped_page("2026-10-09T13:20:00-04:00", "2026-10-09T13:20:37-04:00")
    result = extract_article(html, "https://www.prnewswire.com/news-releases/example-ai-co-launches-new-platform-302903718.html")

    assert result.published_at is not None
    assert result.published_at == datetime(2026, 10, 9, 17, 20, 0, tzinfo=timezone.utc)
    assert result.published_at_precision == "exact"


def test_2_non_utc_timezone_offset_converts_to_the_correct_instant():
    """A JSON-LD timestamp in a non-UTC timezone (here, +09:00, distinct
    from the -04:00 used elsewhere in this suite) must convert to the
    correct UTC instant, not be misread as already being in UTC."""
    html = _pr_newswire_shaped_page("2026-10-09T22:15:00+09:00", "2026-10-09T22:20:00+09:00")
    result = extract_article(html, "https://www.prnewswire.com/news-releases/example-ai-co-launches-new-platform-302903719.html")

    assert result.published_at is not None
    assert result.published_at == datetime(2026, 10, 9, 13, 15, 0, tzinfo=timezone.utc)
    assert result.published_at_precision == "exact"


def test_3_a_genuinely_date_only_jsonld_publication_remains_date_only():
    """When JSON-LD itself carries no time component (a bare calendar
    date), the result must stay DATE_ONLY -- this fix upgrades precision
    only when a real, more-precise timestamp is actually available, it
    never invents one."""
    html = _pr_newswire_shaped_page("2026-10-09", "2026-10-09")
    result = extract_article(html, "https://www.prnewswire.com/news-releases/example-ai-co-launches-new-platform-302903720.html")

    assert result.published_at is not None
    assert result.published_at.date().isoformat() == "2026-10-09"
    assert result.published_at_precision == "date_only"


def test_4_datemodified_never_overrides_a_valid_datepublished():
    """`dateModified` carries a different day AND a time component more
    precise-looking than `datePublished` -- it must never be read as, or
    override, the publication date."""
    html = _pr_newswire_shaped_page("2026-10-09T13:20:00-04:00", "2026-10-11T08:00:00-04:00")
    result = extract_article(html, "https://www.prnewswire.com/news-releases/example-ai-co-launches-new-platform-302903721.html")

    assert result.published_at is not None
    assert result.published_at.date().isoformat() == "2026-10-09"  # datePublished's day, never dateModified's
    assert result.modified_at is not None
    assert result.modified_at.date().isoformat() == "2026-10-11"


def test_5_invalid_jsonld_date_falls_back_to_existing_valid_extraction_behavior():
    """A JSON-LD `datePublished` that fails to parse must never crash or
    poison the result, and must never be "promoted" over anything (an
    invalid string has no time component by definition, so the new
    precedence check correctly never engages here) -- the extractor falls
    back exactly as it did before this fix, recovering the date from the
    page's own `dc.date.issued` meta tag."""
    html = _page(head="""
    <script type="application/ld+json">
    {"@type": "NewsArticle", "datePublished": "not-a-real-timestamp"}
    </script>
    <meta name="dc.date.issued" content="2026-10-09T13:20:00-04:00" />
    """)
    result = extract_article(html, _URL)
    assert result.published_at is not None
    assert result.published_at.date().isoformat() == "2026-10-09"


def test_6_six_pr_newswire_examples_retain_correct_precise_timestamps():
    """Representative fixtures matching the real cached HTML pattern of
    all six examples named in the `e818e08adc2e4dd1` forensic report --
    each must now retain its real, precise publish instant rather than
    collapsing to midnight UTC."""
    examples = [
        ("HeyGen Voice", "2026-10-09T13:20:00-04:00", datetime(2026, 10, 9, 17, 20, 0, tzinfo=timezone.utc)),
        ("Lean SuperIntelligence", "2026-10-09T13:39:00-04:00", datetime(2026, 10, 9, 17, 39, 0, tzinfo=timezone.utc)),
        ("TestMu AI", "2026-10-09T12:20:00-04:00", datetime(2026, 10, 9, 16, 20, 0, tzinfo=timezone.utc)),
        ("KONST", "2026-10-09T11:00:00-04:00", datetime(2026, 10, 9, 15, 0, 0, tzinfo=timezone.utc)),
        ("Pine AI", "2026-10-09T12:00:00-04:00", datetime(2026, 10, 9, 16, 0, 0, tzinfo=timezone.utc)),
        ("Driven Tech", "2026-10-09T10:15:00-04:00", datetime(2026, 10, 9, 14, 15, 0, tzinfo=timezone.utc)),
    ]
    for idx, (name, date_published, expected_utc) in enumerate(examples):
        modified = date_published  # mirrors real pages, where dateModified ~= datePublished at release time
        html = _pr_newswire_shaped_page(date_published, modified)
        url = f"https://www.prnewswire.com/news-releases/example-{idx}-302903{700 + idx}.html"
        result = extract_article(html, url)
        assert result.published_at == expected_utc, name
        assert result.published_at_precision == "exact", name


def test_7_no_regression_in_the_nvidia_ga4_publication_date_fallback():
    """This fix only changes behavior when JSON-LD offers a precise date
    AND trafilatura's own value lacks one -- the NVIDIA page has no JSON-LD
    at all, so its GA4 `PublishDate` fallback (`_apply_metadata_date_
    fallbacks`, exercised only when `published_at` is still None after the
    trafilatura backend) must be completely unaffected."""
    html = _page(body=(
        "<article><p>NVIDIA today announced NVIDIA Open Agent Safety Platform. "
        + "x" * 300 + "</p></article>"
        "<script>ga_four_event('PublishDate','View','September 28, 2026');</script>"
    ))
    result = extract_article(html, "https://nvidianews.nvidia.com/news/open-agent-safety-platform")
    assert result.published_at is not None
    assert result.published_at.date().isoformat() == "2026-09-28"
