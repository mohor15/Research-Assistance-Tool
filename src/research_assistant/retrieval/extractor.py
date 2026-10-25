"""Article extraction from retrieved HTML.

Trafilatura is preferred when installed; a BeautifulSoup-based extractor
(metadata tags + OpenGraph + JSON-LD + paragraph text) is always available
as a fallback so the engine still works without it. Extraction never
pretends to have succeeded: `ExtractionStatus` distinguishes a real article
from a thin/empty page, and callers (Phase 2+) decide what to do with a
`NOT_ARTICLE`/`FAILED` result.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import List, Optional, Tuple
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from research_assistant.retrieval.dates import date_string_has_time_component, parse_date
from research_assistant.retrieval.models import ExtractedArticle, ExtractionMethod, ExtractionStatus

logger = logging.getLogger(__name__)

try:
    import trafilatura  # type: ignore

    TRAFILATURA_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when trafilatura is absent
    TRAFILATURA_AVAILABLE = False

_BOILERPLATE_TAGS = ("script", "style", "nav", "header", "footer", "aside", "form", "noscript")
MIN_SUCCESS_CHARS = 200
MIN_PARTIAL_CHARS = 50

# Phase 14 forensic fix: a "Site | Section" pipe-style title (the same
# hub-page signal already used for RANKING demotion in
# `pipeline.nodes._HUB_TITLE_MARKERS`) combined with a body this thin is a
# listing/index-page shell, not a real article -- demonstrated on 10 real
# JPMorganChase pages (e.g. "Press Releases | JPMorganChase",
# "Events | JPMorganChase"), all of which extracted to 361-445 characters
# of boilerplate. The threshold is deliberately set in the real, observed
# gap between the longest confirmed hub-shell page (445 chars) and the
# shortest confirmed genuine pipe-titled evergreen page in the same audit
# (496 chars) -- not an arbitrary guess. This is a hard reclassification
# (SUCCESS/PARTIAL -> NOT_ARTICLE), never applied to a title without a
# pipe, so a genuinely short real article with no pipe in its title is
# completely unaffected.
_HUB_SHELL_PIPE_MARKER = "|"
_HUB_SHELL_MAX_CHARS = 460

# Phase 14 date-recovery fix (narrow, source-aware, NOT a general URL
# heuristic): Bank of America's newsroom pages carry no date metadata of
# any kind (confirmed by direct, live inspection -- zero <meta>, <time>,
# or JSON-LD date fields), but their own URL structure is a genuine,
# verified-across-multiple-real-articles convention:
# newsroom.bankofamerica.com/.../press-releases/<YYYY>/<MM>/<slug>.html.
# This gives MONTH precision only -- there is no day-level signal to
# recover, and none is invented. Deliberately gated on BOTH the exact
# domain AND a known BofA newsroom content-section path segment, so it can
# never fire for an unrelated bankofamerica.com page or any other source
# whose URL happens to contain a similarly-shaped "/YYYY/MM/" segment for
# an unrelated reason.
#
# Widened (still narrow) to also recognize `/stories/<YYYY>/<MM>/` --
# confirmed via a real, cached article (BofA's "A letter to shareholders
# from Chair and CEO Brian Moynihan", newsroom.bankofamerica.com/content/
# newsroom/stories/2026/03/...) that BofA's "stories" content section
# (narrative/commentary pieces, as opposed to formal "press-releases")
# uses the exact same <YYYY>/<MM>/ URL convention and also carries zero
# date metadata. Both path segments are BofA's own, pre-existing newsroom
# taxonomy -- this is not a generic catch-all, it's the same narrow
# per-source mechanism extended to the one other section it already
# covers by construction.
_BOFA_NEWSROOM_URL_RE = re.compile(
    r"^https?://newsroom\.bankofamerica\.com/.*?/(?:press-releases|stories)/(\d{4})/(\d{2})/", re.IGNORECASE,
)


def _bank_of_america_month_fallback(url: str) -> Optional[Tuple[datetime, str]]:
    """Returns `(first-of-month UTC datetime, "month")` if, and only if,
    `url` matches the exact, narrow Bank of America newsroom URL shape
    above (press-releases or stories); otherwise `None`. Never applied to
    any other source."""
    match = _BOFA_NEWSROOM_URL_RE.match(url)
    if not match:
        return None
    year, month = int(match.group(1)), int(match.group(2))
    if not (1 <= month <= 12) or not (2000 <= year <= 2100):
        return None
    return datetime(year, month, 1, tzinfo=timezone.utc), "month"


def _pwc_release_date_fallback(html: str) -> Optional[datetime]:
    """pwc.com's own CMS exposes a non-standard `<meta name="pwcReleaseDate"
    content="...">` tag -- confirmed via live inspection (2026-10-04) of
    several pwc.com pages -- distinct from its sibling `pwcLastModifiedDate`
    (a MODIFIED date, deliberately never read here: this function only ever
    returns the release/publication date). Neither name is one of the
    standard date meta-tag names (`article:published_time`, `date`,
    `itemprop=datePublished`, ...) any existing check recognizes, so a real,
    unambiguous, CMS-labeled publication date on these pages was otherwise
    always lost. Narrowly scoped to this one named tag -- never a generic
    "scan every meta tag for anything date-shaped" heuristic, which would
    risk picking up an unrelated date on sources this was never inspected
    against."""
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        return None
    tag = soup.find("meta", attrs={"name": "pwcReleaseDate"})
    if not tag or not tag.get("content"):
        return None
    return parse_date(tag["content"].strip())

# Content types that are known to be non-HTML binary assets (e.g. the PDF
# gallery files NVIDIA's newsroom links to). Trafilatura/bs4 are HTML
# parsers; handing them binary bytes decoded as "text" produces garbage
# that must never be treated as article content. V1 scope: reject cleanly,
# do not attempt to parse these formats.
_BINARY_CONTENT_TYPE_PREFIXES = ("image/", "video/", "audio/")
_BINARY_CONTENT_TYPES = ("application/pdf", "application/octet-stream")
_BINARY_URL_EXTENSIONS = (
    ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".bmp",
    ".mp4", ".mp3", ".wav", ".mov", ".zip",
)


def is_trafilatura_available() -> bool:
    return TRAFILATURA_AVAILABLE


def _unsupported_content_reason(content_type: Optional[str], url: str) -> Optional[str]:
    """Return a short reason string if `content_type`/`url` indicate a
    non-HTML binary asset that must never reach the HTML extractors, or
    None if extraction should proceed as normal.

    Content-Type is authoritative when present. It is only absent (e.g. in
    tests, or a server that omits the header) do we fall back to a
    conservative URL-extension check -- anything ambiguous passes through
    unchanged, preserving existing behavior for normal pages.
    """
    if content_type:
        primary = content_type.split(";", 1)[0].strip().lower()
        if primary.startswith(_BINARY_CONTENT_TYPE_PREFIXES) or primary in _BINARY_CONTENT_TYPES:
            return f"content_type={primary}"
        return None

    path = urlparse(url).path.lower()
    if path.endswith(_BINARY_URL_EXTENSIONS):
        return f"url_extension={path.rsplit('.', 1)[-1]}"
    return None


def extract_article(html: str, url: str, *, content_type: Optional[str] = None) -> ExtractedArticle:
    if not html or not html.strip():
        return ExtractedArticle(
            url=url, status=ExtractionStatus.FAILED, extraction_method=ExtractionMethod.NONE,
            metadata={"reason": "empty_html"},
        )

    unsupported = _unsupported_content_reason(content_type, url)
    if unsupported is not None:
        logger.info("Rejecting unsupported binary content before extraction for %s (%s)", url, unsupported)
        return ExtractedArticle(
            url=url, status=ExtractionStatus.FAILED, extraction_method=ExtractionMethod.NONE,
            metadata={"reason": "unsupported_content_type", "detail": unsupported},
        )

    if TRAFILATURA_AVAILABLE:
        result = _extract_with_trafilatura(html, url)
        if result is not None:
            return _apply_metadata_date_fallbacks(_reclassify_hub_shell(result), html, url)
        logger.info("trafilatura found no usable content for %s, falling back to bs4", url)

    return _apply_metadata_date_fallbacks(_reclassify_hub_shell(_extract_with_bs4(html, url)), html, url)


def _apply_metadata_date_fallbacks(result: ExtractedArticle, html: str, url: str) -> ExtractedArticle:
    """Only ever fires when NOTHING more reliable was found (preserves the
    precedence rule: any real metadata/corroborated date always wins over
    this). Tried in order, each generic (never keyed to one specific
    article/title) and each stopping at the first real date found:

    0. Standard published-date tags trafilatura's own `.date` heuristic
       failed to surface correctly (`_standard_meta_or_time_date`) -- see
       the forensic comment at the call site below.
    1. Extended standard-ish HTML metadata tag names (`_extended_meta_date`)
       -- the same `<meta>`-tag concept `_extract_with_bs4`'s own date_str
       lookup already uses, just a wider set of real-world publication-date
       tag names it didn't check.
    2. Generic embedded hydration/application-state publication date
       (`_hydration_state_date`) -- a non-JSON-LD `<script>` block (an
       analytics/tag-manager payload, a framework hydration blob, ...)
       carrying a recognizable publication-date KEY, confirmed real on a
       live Deloitte page (2026-10-07 CURRENT run) whose `datePublished`
       lived inside a `dataLayer.page = {...}` tag-manager object, not
       JSON-LD, so `_jsonld_date` never found it.
    2b. Generic GA4 `PublishDate` analytics-event call (`_ga4_publish_date`)
       -- `ga_four_event('PublishDate', '<action>', '<date>')`, confirmed
       real on a live NVIDIA newsroom page (2026-10-07) whose only date
       signal anywhere on the page was this call's third argument; not
       JSON-LD, not a `<meta>` tag, not a `"key":"value"` hydration pair,
       so neither of the two checks above ever found it. Matched purely by
       the generic, semantic event name `PublishDate` -- never a
       site-specific selector.
    3. Visible, explicitly-labelled publication date in the article's own
       text (`_visible_published_label_date`) -- only when preceded by an
       explicit label word ("Published", "Posted", ...), never a blind
       scan for the first date-shaped string anywhere on the page (which
       would catch copyright/footer/navigation/historical dates).
    3b. Visible, UNLABELLED publication-date header (`_visible_unlabeled_
       header_date`) -- a short, isolated text node appearing before the
       article's own body prose begins, whose entire content is nothing
       but a date (optionally with a time/timezone) -- recognized by
       structural position/isolation rather than a label word, still never
       a scan of the article's own prose.
    4. Generic, source-agnostic URL-path date (`_generic_url_date_
       fallback`) -- a full year/month/day or year/month date encoded in
       the URL's own path, deterministic and domain-independent (contrast
       the narrow, single-domain fallback below). Only ever used when
       nothing above found anything; an ambiguous or coarser URL date never
       overrides a real metadata/content date, and never gains an invented
       time-of-day.
    5. Two pre-existing narrow, source-aware fallbacks -- Bank of America's
       URL-derived month fallback, then PwC's non-standard
       `pwcReleaseDate` meta tag (see `_pwc_release_date_fallback`).

    None of these ever read `dateModified`/`lastModified`/`updatedAt`-style
    keys as if they were the publication date -- each only matches
    publication-specific key/label names."""
    if result.published_at is not None:
        return result

    # 0. Re-check the STANDARD published-date tags (`_extract_with_bs4`'s own
    # date_str lookup names, plus `<time datetime>` and JSON-LD) independently
    # of trafilatura's own `.date` heuristic. Forensic finding (2026-10-12):
    # trafilatura's own metadata engine sometimes guesses a publication date
    # from the URL or body prose INSTEAD OF a real, present standard tag
    # (confirmed: `trafilatura.extract_metadata(...).date` returns a
    # URL-encoded date even when the page also carries a genuine
    # `article:published_time` meta tag with a different date) -- when that
    # guess disagrees with every tag on the page, `_corroborate_against_page_
    # metadata` (in `_extract_with_trafilatura`) correctly discards it, but
    # nothing previously re-tried the real standard tag afterward, silently
    # losing a genuinely reliable Tier-1 date. This must run BEFORE every
    # other fallback (including the new generic URL tier below) so a lower-
    # priority guess can never be preferred over real structured metadata
    # that trafilatura itself failed to surface correctly.
    standard_meta = _standard_meta_or_time_date(html)
    if standard_meta is not None:
        parsed = parse_date(standard_meta)
        if parsed is not None:
            precision = "exact" if date_string_has_time_component(standard_meta) else "date_only"
            logger.info(
                "Recovering a publication date for %s from a standard metadata/time/JSON-LD tag "
                "that trafilatura's own date heuristic did not surface (no corroborated date found "
                "any other way): %s", url, parsed.date().isoformat(),
            )
            return result.model_copy(update={"published_at": parsed, "published_at_precision": precision})

    extended_meta = _extended_meta_date(html)
    if extended_meta is not None:
        parsed = parse_date(extended_meta)
        if parsed is not None:
            logger.info(
                "Recovering a publication date for %s from an extended metadata tag "
                "(no standard date metadata exists on this page): %s", url, parsed.date().isoformat(),
            )
            return result.model_copy(update={"published_at": parsed})

    hydration_date = _hydration_state_date(html)
    if hydration_date is not None:
        parsed = parse_date(hydration_date)
        if parsed is not None:
            logger.info(
                "Recovering a publication date for %s from an embedded hydration/application-state "
                "object (no JSON-LD/standard metadata date exists on this page): %s",
                url, parsed.date().isoformat(),
            )
            return result.model_copy(update={"published_at": parsed})

    ga4_date = _ga4_publish_date(html)
    if ga4_date is not None:
        parsed = parse_date(ga4_date)
        if parsed is not None:
            logger.info(
                "Recovering a publication date for %s from a GA4 'PublishDate' analytics event call "
                "(no JSON-LD/standard metadata/hydration date exists on this page): %s",
                url, parsed.date().isoformat(),
            )
            return result.model_copy(update={"published_at": parsed})

    visible_date = _visible_published_label_date(html)
    if visible_date is not None:
        parsed = parse_date(visible_date)
        if parsed is not None:
            precision = "exact" if date_string_has_time_component(visible_date) else "date_only"
            logger.info(
                "Recovering a publication date for %s from a labelled publication date in the "
                "visible article text (no metadata date exists on this page): %s",
                url, parsed.date().isoformat(),
            )
            return result.model_copy(update={"published_at": parsed, "published_at_precision": precision})

    header_date = _visible_unlabeled_header_date(html)
    if header_date is not None:
        parsed = parse_date(header_date)
        if parsed is not None:
            precision = "exact" if date_string_has_time_component(header_date) else "date_only"
            logger.info(
                "Recovering a publication date for %s from an unlabelled, structurally-isolated "
                "header timestamp in the visible article text (no labelled or metadata date exists "
                "on this page): %s",
                url, parsed.date().isoformat(),
            )
            return result.model_copy(update={"published_at": parsed, "published_at_precision": precision})

    url_fallback = _generic_url_date_fallback(url)
    if url_fallback is not None:
        fallback_date, precision = url_fallback
        logger.info(
            "Recovering a %s-precision publication date for %s from its own URL structure "
            "(no metadata or content date exists on this page): %s",
            precision, url, fallback_date.date().isoformat(),
        )
        return result.model_copy(update={"published_at": fallback_date, "published_at_precision": precision})

    bofa_fallback = _bank_of_america_month_fallback(url)
    if bofa_fallback is not None:
        fallback_date, precision = bofa_fallback
        logger.info(
            "Recovering a month-precision publication date for %s from its URL structure "
            "(no tag-level date metadata exists on this page): %s",
            url, fallback_date.date().isoformat(),
        )
        return result.model_copy(update={"published_at": fallback_date, "published_at_precision": precision})

    pwc_fallback = _pwc_release_date_fallback(html)
    if pwc_fallback is not None:
        logger.info(
            "Recovering a publication date for %s from its pwcReleaseDate meta tag "
            "(no standard date metadata exists on this page): %s",
            url, pwc_fallback.date().isoformat(),
        )
        return result.model_copy(update={"published_at": pwc_fallback})

    return result


# Section 5B: a wider, still-generic set of real-world publication-date
# `<meta>` tag names beyond the ones `_extract_with_bs4`'s own date_str
# lookup already checks (article:published_time, name=date,
# name=publish-date, itemprop=datePublished). Every name here is a
# recognized, widely-used publication-date convention (Dublin Core,
# Parse.ly, generic CMS exports, ...) -- never a single site's own
# invented tag name (those stay as their own narrow, documented fallbacks,
# e.g. `_pwc_release_date_fallback`). Deliberately excludes any
# modification-dated name (e.g. "last-modified", "dc.date.modified") --
# publication date must never be confused with modification date.
#
# `name="date"` (CURRENT run 7aab105954d38497 forensic finding, PR
# Newswire): this is ALSO one of `_extract_with_bs4`'s own standard
# tag names, but that fallback extractor only ever runs when trafilatura
# itself returns None entirely -- it is never consulted when trafilatura
# returns a SUCCESS result (real body text found) with `published_at`
# merely left None, which is exactly what trafilatura's own metadata
# engine did on a real, minimal-markup PR Newswire page carrying nothing
# but this one tag. Without it here too, that page's own standard,
# unambiguous date tag was silently lost with no fallback recovery at
# all -- not a one-source hack, `name="date"` is one of the most common
# publication-date tag names on the web.
_EXTENDED_PUBLISHED_META: List[Tuple[str, str]] = [
    ("name", "date"),
    ("name", "datePublished"),
    ("name", "publication_date"),
    ("name", "publish_date"),
    ("name", "pub_date"),
    ("name", "timestamp"),
    ("name", "parsely-pub-date"),
    ("name", "dc.date"),
    ("name", "dc.date.issued"),
    ("name", "dcterms.date"),
]


def _extended_meta_date(html: str) -> Optional[str]:
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        return None
    return _first_meta(soup, _EXTENDED_PUBLISHED_META)


# Same standard tag names `_extract_with_bs4` already checks -- re-checked
# independently here because that backend only ever runs when trafilatura
# itself returns nothing at all, never when trafilatura succeeds but its own
# `.date` heuristic picked an uncorroborated (and therefore discarded) guess
# instead of the real tag. See the call site's comment above for the
# forensic finding this closes.
_STANDARD_PUBLISHED_META: List[Tuple[str, str]] = [
    ("property", "article:published_time"),
    ("name", "date"),
    ("name", "publish-date"),
    ("itemprop", "datePublished"),
]


def _sole_consistent_time_tag_value(soup: BeautifulSoup) -> Optional[str]:
    """A page's `<time>` tags are trustworthy as a byline date ONLY when
    every one of them agrees on the same calendar day (see
    `_corroborate_against_page_metadata`'s forensic comment: a page that
    mixes several distinct dates among its `<time>` tags is a listing/
    carousel of OTHER items, e.g. OpenAI's "latest from OpenAI" widget --
    not this article's own byline). Returns the first tag's raw
    `datetime` string when the whole page agrees, else None -- shared by
    both the trafilatura-guess corroboration path and this module's own
    metadata-fallback path so neither can independently re-trust an
    ambiguous `<time>` tag."""
    distinct_days = set()
    first_value: Optional[str] = None
    for tag in soup.find_all("time", attrs={"datetime": True}):
        value = tag["datetime"]
        if first_value is None:
            first_value = value
        parsed = parse_date(value)
        if parsed is not None:
            distinct_days.add(parsed.date())
    if first_value is not None and len(distinct_days) == 1:
        return first_value
    return None


def _standard_meta_or_time_date(html: str) -> Optional[str]:
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        return None
    value = _first_meta(soup, _STANDARD_PUBLISHED_META)
    if value:
        return value
    time_value = _sole_consistent_time_tag_value(soup)
    if time_value:
        return time_value
    return _jsonld_date(soup, keys=("datePublished", "dateCreated"))


# Section 5F: generic embedded hydration/application-state publication
# date. Matches ONLY a small set of well-known, generic publication-date
# KEY NAMES (the same names JSON-LD/meta-tag checks already treat as
# publication-specific) inside a quoted-JSON-shaped `"key":"value"` pair
# anywhere in the raw page text -- deliberately not a full JS/JSON parse
# (these blocks, e.g. a `dataLayer.page = {...}` tag-manager payload, are
# often plain JS object literals, not strict JSON), and deliberately never
# a site-specific key or class name. A sibling `dateModified`/`updatedAt`
# key appearing elsewhere in the same blob is never matched by this
# pattern, so it can never be mistaken for the publication date.
#
# Tried as an ORDERED LIST of individual keys, never a single alternation
# -- confirmed real bug on a live Deloitte page (2026-10-07 run): its
# tracking payload carries BOTH `"createDate": "14 Jan 2026"` (earlier in
# the text) and `"datePublished": "20 Jan 2026"` (later); a single
# alternation regex's `.search()` returns whichever key happens to appear
# FIRST POSITIONALLY, not the more authoritative one. Checking each
# publication-specific key in priority order first, and only falling back
# to the weaker "created" family if none of those exist anywhere, mirrors
# the same `("datePublished", "dateCreated")` precedence `_jsonld_date`
# already applies.
_HYDRATION_PUBLISHED_KEYS: Tuple[str, ...] = ("datePublished", "publishedAt", "publication_date", "publish_date", "pub_date")
_HYDRATION_CREATED_KEYS: Tuple[str, ...] = ("createDate", "dateCreated")


def _hydration_state_date(html: str) -> Optional[str]:
    for key in (*_HYDRATION_PUBLISHED_KEYS, *_HYDRATION_CREATED_KEYS):
        match = re.search(rf'"{re.escape(key)}"\s*:\s*"([^"]+)"', html, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


# Generic GA4 "PublishDate" custom-analytics-event call -- confirmed real
# on a live NVIDIA newsroom page (2026-10-07 CURRENT run): the page's ONLY
# date signal anywhere was `ga_four_event('PublishDate','View','September
# 28, 2026')`, a Google-Analytics-4 custom event whose first argument is
# the semantic, generic event CATEGORY name "PublishDate" (never a
# site-specific key) -- not JSON-LD, not a `<meta>` tag, not a
# `"key":"value"` hydration pair. Matched purely by that generic event
# name, with reasonable quote-style ('/") and whitespace variation; the
# event's own ACTION argument (e.g. "View") is deliberately ignored -- only
# the third (label/date) argument is read. Never a blind scan of every
# `ga_four_event(...)` call on the page -- only one whose first argument is
# exactly "PublishDate" ever qualifies, so an unrelated analytics event
# (e.g. a "ButtonClick" or "PageView" category) can never be mistaken for a
# publication date.
_GA4_PUBLISH_DATE_RE = re.compile(
    r"""ga_four_event\s*\(\s*['"]PublishDate['"]\s*,\s*['"][^'"]*['"]\s*,\s*['"]([^'"]+)['"]\s*\)""",
    re.IGNORECASE,
)


def _ga4_publish_date(html: str) -> Optional[str]:
    match = _GA4_PUBLISH_DATE_RE.search(html)
    return match.group(1) if match else None


# Section 5D: visible, EXPLICITLY-labelled publication date in the
# article's own rendered text. Requires one of a small set of real
# publication-label words/phrases immediately before the date -- never a
# blind "first date-shaped string on the page" scan, which would just as
# happily match a copyright year, a footer date, a navigation widget, or a
# historical date mentioned inside the article's own body prose. Matches a
# "Month DD, YYYY", "DD Month YYYY", ISO "YYYY-MM-DD", or slash "YYYY/MM/DD"
# shape immediately following the label, optionally followed by a
# time-of-day and timezone abbreviation (e.g. "October 9, 2026, 10:00 ET")
# -- generic date-shape patterns, never a publisher-specific format.
_DATE_SHAPE_PATTERN = (
    r"[A-Za-z]+\.?\s+\d{1,2},?\s+\d{4}"  # October 9, 2026 / Oct. 9, 2026
    r"|\d{1,2}\s+[A-Za-z]+\.?\s+\d{4}"  # 9 October 2026 / 09 October 2026
    r"|\d{4}-\d{2}-\d{2}"  # 2026-10-09
    r"|\d{4}/\d{2}/\d{2}"  # 2026/10/09
)
_OPTIONAL_TIME_TZ_SUFFIX = r"(?:,?\s+\d{1,2}:\d{2}(?:\s*[AaPp]\.?[Mm]\.?)?\s*(?:[A-Za-z]{2,5}\b)?)?"

_VISIBLE_PUBLISHED_LABEL_RE = re.compile(
    r"(?:Originally\s+published|Published(?:\s+on)?|Posted(?:\s+on)?|Release\s+date|"
    r"Announcement\s+date|Publication\s+date)"
    r"\s*[:\-]?\s*"
    rf"((?:{_DATE_SHAPE_PATTERN}){_OPTIONAL_TIME_TZ_SUFFIX})",
    re.IGNORECASE,
)

# Section 5G: recognizable, UNLABELLED publication-date header -- a short
# text node (e.g. a `<time>`/byline `<span>`/`<div>` sitting near the
# headline) whose ENTIRE stripped content is just a date, optionally with a
# time-of-day/timezone (the same wire-service byline convention as
# "October 9, 2026, 10:00 ET" or "Oct 09, 2026, 10:00 ET"). This is
# deliberately never a scan of arbitrary prose: a date mentioned inside a
# full sentence ("Back in March 15, 2019 the company first launched...")
# never matches because the WHOLE node text must be nothing but the date
# (and optional time/tz) -- the same structural signal a human reader uses
# to recognize an unlabelled timestamp header as opposed to a date quoted
# in body copy.
_UNLABELLED_HEADER_DATE_RE = re.compile(
    rf"^(?:{_DATE_SHAPE_PATTERN}){_OPTIONAL_TIME_TZ_SUFFIX}$", re.IGNORECASE,
)
_UNLABELLED_HEADER_MAX_CHARS = 60

# Boilerplate tags stripped before scanning visible text -- mirrors
# `_BOILERPLATE_TAGS` above, so this never matches a label/date sitting in
# navigation, a footer, or a script/style block (copyright notices, site
# chrome) -- only text that is part of the rendered article page body.
_VISIBLE_TEXT_STRIP_TAGS = _BOILERPLATE_TAGS


def _visible_published_label_date(html: str) -> Optional[str]:
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        return None
    for tag_name in _VISIBLE_TEXT_STRIP_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()
    text = soup.get_text(" ", strip=True)
    match = _VISIBLE_PUBLISHED_LABEL_RE.search(text)
    return match.group(1) if match else None


def _visible_unlabeled_header_date(html: str) -> Optional[str]:
    """See `_UNLABELLED_HEADER_DATE_RE` above: a `<time>` element -- the one
    HTML element whose semantic meaning is specifically "this is a
    machine-readable timestamp" (never a bare `<span>`/`<div>`/`<p>`, which
    could just as easily be a listing card's date for a DIFFERENT article,
    a caption, or anything else -- confirmed real false-positive risk,
    PR Newswire topic-hub-page regression fixture) -- whose entire content
    is just a date, appearing BEFORE the article's own body prose begins
    (the first paragraph with real substance, `MIN_PARTIAL_CHARS` or more),
    exactly where a byline/dateline timestamp sits and a date quoted inside
    the body, or a sibling listing card's own date, never does."""
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        return None
    for tag_name in _VISIBLE_TEXT_STRIP_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()
    for element in soup.find_all(["time", "p"]):
        if element.find(True) is not None:
            continue  # only leaf nodes: a wrapping container's aggregate text is not this node's own content
        text = element.get_text(" ", strip=True)
        if not text:
            continue
        if len(text) >= MIN_PARTIAL_CHARS:
            # Reached the start of real body prose -- any date-only node
            # from here on is inside the article's own content, not a
            # header, and must go through the labelled check instead.
            break
        if element.name == "time" and len(text) <= _UNLABELLED_HEADER_MAX_CHARS and _UNLABELLED_HEADER_DATE_RE.match(text):
            return text
    return None


# Tier 3 (generic URL date): deterministic, source-agnostic recovery of a
# publication date encoded in the article's own URL PATH -- never the query
# string or fragment (tracking/session parameters must never be read as a
# date), and never gated on any specific domain (contrast the narrow,
# domain-gated `_bank_of_america_month_fallback` above, which stays as its
# own Tier 4 fallback). Tried in strict specificity order so a full
# year/month/day date always wins over a weaker partial year/month one, and
# a plausible-but-ambiguous numeric fragment (an article id, a product SKU,
# a tracking number) is never mistaken for a date: every candidate is range-
# validated (a real calendar year, a real month 1-12, a real day 1-31)
# before being accepted, and a numeric run that is part of a longer digit
# sequence (e.g. an 8-digit id embedded in a longer numeric string) is
# rejected via explicit path-boundary anchoring.
_URL_DATE_YMD_SLASH_RE = re.compile(r"/(\d{4})/(\d{1,2})/(\d{1,2})(?:/|$)")
_URL_DATE_ISO_RE = re.compile(r"/(\d{4})-(\d{2})-(\d{2})(?:/|$)")
_URL_DATE_COMPACT_RE = re.compile(r"/(\d{4})(\d{2})(\d{2})(?:/|$)")
_URL_DATE_YM_SLASH_RE = re.compile(r"/(\d{4})/(\d{1,2})(?:/(?!\d))")

_URL_DATE_YEAR_MIN = 2000
_URL_DATE_YEAR_MAX = 2100


def _valid_ymd(year: int, month: int, day: int) -> bool:
    return _URL_DATE_YEAR_MIN <= year <= _URL_DATE_YEAR_MAX and 1 <= month <= 12 and 1 <= day <= 31


def _valid_ym(year: int, month: int) -> bool:
    return _URL_DATE_YEAR_MIN <= year <= _URL_DATE_YEAR_MAX and 1 <= month <= 12


def _generic_url_date_fallback(url: str) -> Optional[Tuple[datetime, str]]:
    """Returns `(datetime, precision)` ("date_only" or "month") recovered
    deterministically from `url`'s own path structure, or `None` if no
    pattern with a defensible publication-date interpretation matches.
    Never inspects the query string or fragment. Never invents a time of
    day -- a URL-derived date can only ever be DATE_ONLY or MONTH, never
    EXACT (a URL path has no time-of-day signal to recover)."""
    path = urlparse(url).path

    match = _URL_DATE_YMD_SLASH_RE.search(path)
    if match:
        year, month, day = (int(g) for g in match.groups())
        if _valid_ymd(year, month, day):
            return datetime(year, month, day, tzinfo=timezone.utc), "date_only"

    match = _URL_DATE_ISO_RE.search(path)
    if match:
        year, month, day = (int(g) for g in match.groups())
        if _valid_ymd(year, month, day):
            return datetime(year, month, day, tzinfo=timezone.utc), "date_only"

    match = _URL_DATE_COMPACT_RE.search(path)
    if match:
        year, month, day = (int(g) for g in match.groups())
        if _valid_ymd(year, month, day):
            return datetime(year, month, day, tzinfo=timezone.utc), "date_only"

    match = _URL_DATE_YM_SLASH_RE.search(path)
    if match:
        year, month = int(match.group(1)), int(match.group(2))
        if _valid_ym(year, month):
            return datetime(year, month, 1, tzinfo=timezone.utc), "month"

    return None


def _reclassify_hub_shell(result: ExtractedArticle) -> ExtractedArticle:
    """See `_HUB_SHELL_MAX_CHARS` above. Applied once, uniformly, regardless
    of which backend produced `result` -- never a source/URL-specific
    check.

    Final-gate fix (confirmed defect, caught by a direct test): the
    downstream evidence gate (`evidence/conversion.py::article_to_evidence`)
    has always decided purely from `clean_text`/`content_hash` truthiness,
    never from `extraction_status` -- true for every OTHER NOT_ARTICLE/
    FAILED path in this project (confirmed: every one of those leaves
    `text=None`). This reclassification was the ONE path that changed only
    the status label while leaving real text in place, so a hub-shell page
    in the 50-460 character band could still produce usable Evidence and
    reach an agent despite being marked NOT_ARTICLE. Clearing `text` here
    (not adding a new, broader check to the evidence gate, which would
    change its contract for every other caller) keeps this fix scoped to
    exactly the one path that was actually wrong.
    """
    if result.status not in (ExtractionStatus.SUCCESS, ExtractionStatus.PARTIAL):
        return result
    title = result.title or ""
    text_len = len(result.text) if result.text else 0
    if _HUB_SHELL_PIPE_MARKER in title and text_len < _HUB_SHELL_MAX_CHARS:
        logger.info(
            "Reclassifying %s as NOT_ARTICLE: pipe-style title (%r) with only %d chars of body "
            "-- looks like a listing/index page shell, not a real article",
            result.url, title, text_len,
        )
        return result.model_copy(update={"status": ExtractionStatus.NOT_ARTICLE, "text": None})
    return result


# ---------------------------------------------------------------------------
# trafilatura backend
# ---------------------------------------------------------------------------


def _extract_with_trafilatura(html: str, url: str) -> Optional[ExtractedArticle]:
    try:
        text = trafilatura.extract(html, url=url, include_comments=False, include_tables=False)
    except Exception as exc:
        logger.warning("trafilatura raised for %s: %s", url, exc)
        return None

    if not text or len(text.strip()) < MIN_PARTIAL_CHARS:
        return None

    title = author = date_str = publisher = None
    try:
        meta = trafilatura.extract_metadata(html, default_url=url)
    except Exception as exc:
        meta = None
        logger.info("trafilatura metadata extraction failed for %s: %s", url, exc)
    if meta is not None:
        title = getattr(meta, "title", None)
        author = getattr(meta, "author", None)
        date_str = getattr(meta, "date", None)
        publisher = getattr(meta, "sitename", None)

    modified_str = None
    jsonld_published_str = None
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception as exc:
        soup = None
        logger.info("could not parse HTML for JSON-LD date fallback on %s: %s", url, exc)
    trafilatura_date_str = date_str
    if soup is not None:
        jsonld_published_str = _jsonld_date(soup, keys=("datePublished", "dateCreated"))
        if not date_str:
            date_str = jsonld_published_str
        modified_str = _jsonld_date(soup, keys=("dateModified",))

    # Precision/precedence fix (forensic finding, PR Newswire CURRENT run
    # `e818e08adc2e4dd1`): trafilatura's own metadata date (`.extract_
    # metadata().date`) is frequently a bare calendar date with no time-of-
    # day, while the SAME page's own JSON-LD `datePublished` often carries a
    # full, timezone-aware timestamp -- previously always discarded once
    # trafilatura found *anything* (see the `if not date_str:` check above,
    # which only ever consulted JSON-LD when trafilatura found NOTHING at
    # all). Six real PR Newswire releases (HeyGen, Lean SuperIntelligence,
    # TestMu, KONST, Pine AI, Driven Tech) lost their real publish time this
    # way, collapsing to midnight UTC and failing CURRENT's zero-slack
    # wide-lookback window. When JSON-LD offers a MORE PRECISE (time-
    # bearing) value than trafilatura's own date_str, and the two agree on
    # the same calendar day (this only ever upgrades precision on a date
    # trafilatura already got right -- it never overrides a genuinely
    # conflicting date with a different one), the more precise JSON-LD
    # value wins. Generic: driven only by whether a time component is
    # present in each candidate STRING, never a publisher name or domain.
    if (
        jsonld_published_str
        and date_string_has_time_component(jsonld_published_str)
        and not (trafilatura_date_str and date_string_has_time_component(trafilatura_date_str))
    ):
        precise_candidate = parse_date(jsonld_published_str)
        trafilatura_candidate = parse_date(trafilatura_date_str) if trafilatura_date_str else None
        if precise_candidate is not None and (
            trafilatura_candidate is None or precise_candidate.date() == trafilatura_candidate.date()
        ):
            date_str = jsonld_published_str

    published_at = parse_date(date_str)
    if soup is not None and trafilatura_date_str and date_str == trafilatura_date_str:
        # The date came from trafilatura's OWN internal heuristic (its
        # `extract_metadata().date`), not from our own JSON-LD check above
        # -- this is the one path proven, via live inspection, to
        # sometimes mine a date mentioned in the article's own BODY PROSE
        # (e.g. "a February 2023 investigation" inside a 2026 report)
        # rather than its real publication date, when the page carries no
        # date metadata of any kind. Require it to be corroborated by SOME
        # actual tag-level value on the page -- any `<meta>` content,
        # `<time>` value, or JSON-LD date, not necessarily a "standard"
        # date-named one (this is what lets a real but non-standard field,
        # e.g. a `alg-search-date` epoch timestamp, still validate a
        # correct date without hardcoding that tag name anywhere). If
        # nothing on the page corroborates it, the date is unknown --
        # never a guess.
        published_at = _corroborate_against_page_metadata(published_at, soup)

    # Precision hint (same forensic fix as above): generic, string-shape
    # derived -- never inferred from whether the parsed UTC instant happens
    # to land on midnight (an explicit "...T00:00:00Z" source string is
    # still EXACT; a bare "2026-10-09" is DATE_ONLY even though both parse
    # to the identical midnight `datetime`). Consumed by
    # `evidence.normalizer._normalize_publication_date`, which only falls
    # back to its own midnight-heuristic when this hint is absent --
    # exactly the pre-existing behavior for every OTHER extraction path.
    published_at_precision = None
    if published_at is not None:
        published_at_precision = "exact" if date_string_has_time_component(date_str) else "date_only"

    text = text.strip()
    status = ExtractionStatus.SUCCESS if len(text) >= MIN_SUCCESS_CHARS else ExtractionStatus.PARTIAL

    return ExtractedArticle(
        url=url,
        title=title,
        author=author,
        published_at=published_at,
        published_at_precision=published_at_precision,
        modified_at=parse_date(modified_str),
        publisher=publisher,
        text=text,
        extraction_method=ExtractionMethod.TRAFILATURA,
        status=status,
    )


def _corroborate_against_page_metadata(candidate: Optional[datetime], soup: BeautifulSoup) -> Optional[datetime]:
    """Returns `candidate` unchanged if some independent tag-level value on
    the page (any `<meta content=...>`, `<time>`, or JSON-LD date field --
    not a fixed named list) parses to the same calendar day (UTC);
    otherwise returns None. See the call site above for why this exists."""
    if candidate is None:
        return None
    for tag in soup.find_all("meta"):
        content = tag.get("content")
        if content:
            parsed = parse_date(content)
            if parsed is not None and parsed.date() == candidate.date():
                return candidate
    # `<time>` corroboration (forensic fix, 2026-10-16): confirmed real
    # regression -- two DIFFERENT OpenAI articles ("GPT-5.6" and
    # "Introducing GPT-5.5") both extracted the identical wrong
    # published_at (2026-10-15T19:00Z) because the page template embeds a
    # "latest from OpenAI" carousel of OTHER posts' `<time>` tags alongside
    # the real article, and trafilatura's own date guess happened to match
    # one of those unrelated cards -- the real article (per manager report)
    # was actually from 2026-07-09, three months earlier. A bare "does ANY
    # `<time>` tag on the page match" check cannot distinguish a genuine
    # single byline timestamp from one card in a multi-item listing of
    # unrelated dates. Requiring every `<time>` tag on the page to agree on
    # the same calendar day keeps the common case (one real byline `<time>`,
    # or the same date repeated as both "published" and "updated" display)
    # working exactly as before, while rejecting a page that mixes several
    # distinct dates -- the actual shape of a carousel/listing component.
    sole_time_value = _sole_consistent_time_tag_value(soup)
    if sole_time_value:
        parsed = parse_date(sole_time_value)
        if parsed is not None and parsed.date() == candidate.date():
            return candidate
    # Deliberately ONLY the publication-dated JSON-LD keys here (never
    # "dateModified") -- confirmed real gap: a page whose ONLY JSON-LD date
    # is `dateModified` (no `datePublished`/`dateCreated` anywhere) must
    # never have that modification date corroborate trafilatura's own
    # guess into becoming this article's PUBLICATION date. Publication and
    # modification dates are never collapsed into one concept.
    jsonld_value = _jsonld_date(soup, keys=("datePublished", "dateCreated"))
    if jsonld_value:
        parsed = parse_date(jsonld_value)
        if parsed is not None and parsed.date() == candidate.date():
            return candidate
    return None


# ---------------------------------------------------------------------------
# BeautifulSoup fallback backend
# ---------------------------------------------------------------------------


def _extract_with_bs4(html: str, url: str) -> ExtractedArticle:
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception as exc:
        logger.warning("Could not parse HTML for %s: %s", url, exc)
        return ExtractedArticle(
            url=url, status=ExtractionStatus.FAILED, extraction_method=ExtractionMethod.NONE,
            metadata={"reason": "unparseable_html"},
        )

    jsonld_published = _jsonld_date(soup, keys=("datePublished", "dateCreated"))
    jsonld_modified = _jsonld_date(soup, keys=("dateModified",))

    for tag_name in _BOILERPLATE_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    title = _first_meta(soup, [("property", "og:title"), ("name", "twitter:title")])
    if not title and soup.title and soup.title.string:
        title = soup.title.string.strip()
    if not title:
        h1 = soup.find("h1")
        if h1:
            title = h1.get_text(strip=True) or None

    author = _first_meta(soup, [("name", "author"), ("property", "article:author"), ("name", "twitter:creator")])
    publisher = _first_meta(soup, [("property", "og:site_name")])

    date_str = _first_meta(
        soup,
        [("property", "article:published_time"), ("name", "date"), ("name", "publish-date"), ("itemprop", "datePublished")],
    )
    if not date_str:
        time_tag = soup.find("time", attrs={"datetime": True})
        if time_tag:
            date_str = time_tag.get("datetime")
    if not date_str:
        date_str = jsonld_published

    modified_str = _first_meta(soup, [("property", "article:modified_time"), ("itemprop", "dateModified")])
    if not modified_str:
        modified_str = jsonld_modified

    container = soup.find("article") or soup.find("main")
    paragraphs = container.find_all("p") if container is not None else soup.find_all("p")
    text = "\n\n".join(p.get_text(" ", strip=True) for p in paragraphs if p.get_text(strip=True))
    text = text.strip() or None

    text_len = len(text) if text else 0
    if text_len >= MIN_SUCCESS_CHARS:
        status = ExtractionStatus.SUCCESS
    elif text_len >= MIN_PARTIAL_CHARS:
        status = ExtractionStatus.PARTIAL
    else:
        status = ExtractionStatus.NOT_ARTICLE

    return ExtractedArticle(
        url=url,
        title=title,
        author=author,
        published_at=parse_date(date_str),
        modified_at=parse_date(modified_str),
        publisher=publisher,
        text=text,
        extraction_method=ExtractionMethod.BS4_FALLBACK,
        status=status,
    )


def _first_meta(soup: BeautifulSoup, candidates: List[Tuple[str, str]]) -> Optional[str]:
    for attr, value in candidates:
        tag = soup.find("meta", attrs={attr: value})
        if tag and tag.get("content"):
            return tag["content"].strip()
    return None


def _flatten_jsonld_items(node) -> List[dict]:
    """Collects every dict-shaped JSON-LD node reachable from `node`,
    including nodes nested inside a `"@graph": [...]` array -- the
    standard schema.org convention for bundling multiple entities
    (e.g. a WebSite node alongside an Article node) in one script block,
    which the previous implementation never looked inside. Handles a flat
    dict, a top-level list of dicts (both pre-existing, unaffected
    behaviors), and arbitrarily nested `@graph` arrays generically -- no
    site-specific structure is assumed."""
    items: List[dict] = []
    stack = [node]
    while stack:
        current = stack.pop()
        if isinstance(current, list):
            stack.extend(current)
        elif isinstance(current, dict):
            items.append(current)
            graph = current.get("@graph")
            if isinstance(graph, list):
                stack.extend(graph)
    return items


def _is_article_like_type(type_value) -> bool:
    """True for "Article", "NewsArticle", "BlogPosting", etc. -- anything
    whose schema.org `@type` contains "article" -- never a hardcoded exact
    list, so it generalizes to types this project hasn't seen yet."""
    if isinstance(type_value, list):
        return any(_is_article_like_type(t) for t in type_value)
    return isinstance(type_value, str) and "article" in type_value.lower()


def _jsonld_date(soup: BeautifulSoup, *, keys: Tuple[str, ...]) -> Optional[str]:
    """Searches every JSON-LD script block on the page (there may be
    several) for the first of `keys` present on a node. An Article-like
    node (`@type` containing "article") is checked FIRST, across all
    blocks/nodes, so a real publication date on the actual article is
    never shadowed by an unrelated date on a sibling node (e.g. a WebSite
    or Organization node in the same `@graph`) merely because that sibling
    happens to appear first. Only if no Article-like node has the key do
    we fall back to any node at all, preserving the original behavior for
    pages that don't use `@type` this way."""
    all_items: List[dict] = []
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            data = json.loads(script.string or "")
        except (json.JSONDecodeError, TypeError):
            continue
        all_items.extend(_flatten_jsonld_items(data))

    article_like = [item for item in all_items if _is_article_like_type(item.get("@type"))]
    for pool in (article_like, all_items):
        for item in pool:
            for key in keys:
                value = item.get(key)
                if value:
                    return value
    return None
