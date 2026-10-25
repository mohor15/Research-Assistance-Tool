"""RSS/Atom feed parsing.

`feedparser` is the preferred, more forgiving parser and is used when it's
installed. It is an optional dependency here: this module also ships a
pure-stdlib (`xml.etree.ElementTree`) fallback for RSS 2.0 and Atom 1.0 so
feed discovery works even without it, and never raises on a malformed feed —
a bad feed logs a warning and yields zero candidates rather than aborting
the retrieval run.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import List, Optional

from research_assistant.retrieval.dates import parse_date, struct_time_to_datetime
from research_assistant.retrieval.models import CandidateURL, DiscoveryMethod

logger = logging.getLogger(__name__)

try:
    import feedparser  # type: ignore

    FEEDPARSER_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when feedparser is absent
    FEEDPARSER_AVAILABLE = False

_ATOM_NS = "{http://www.w3.org/2005/Atom}"
_DC_NS = "{http://purl.org/dc/elements/1.1/}"


def is_feedparser_available() -> bool:
    return FEEDPARSER_AVAILABLE


def looks_like_feed(content: str, content_type: Optional[str]) -> bool:
    """Heuristic: does this response look like an RSS/Atom feed rather than an HTML page?"""
    if content_type and any(marker in content_type.lower() for marker in ("rss", "atom", "xml")):
        return True
    head = (content or "").lstrip()[:500].lower()
    return head.startswith("<?xml") or "<rss" in head or "<feed" in head


def parse_feed(content: str, *, source_name: str, domain: str) -> List[CandidateURL]:
    if not content or not content.strip():
        return []

    if FEEDPARSER_AVAILABLE:
        try:
            return _parse_with_feedparser(content, source_name=source_name, domain=domain)
        except Exception as exc:  # feedparser should not throw, but never trust a 3rd-party parser blindly
            logger.warning("feedparser failed for %s, falling back to stdlib parser: %s", source_name, exc)

    try:
        return _parse_with_stdlib(content, source_name=source_name, domain=domain)
    except ET.ParseError as exc:
        logger.warning("Malformed feed for %s: %s", source_name, exc)
        return []


# ---------------------------------------------------------------------------
# feedparser backend
# ---------------------------------------------------------------------------


def _parse_with_feedparser(content: str, *, source_name: str, domain: str) -> List[CandidateURL]:
    parsed = feedparser.parse(content)
    now = datetime.now(timezone.utc)
    is_atom = "atom" in (parsed.get("version") or "")
    method = DiscoveryMethod.ATOM if is_atom else DiscoveryMethod.RSS

    entries: List[CandidateURL] = []
    for entry in parsed.get("entries", []):
        url = entry.get("link")
        if not url:
            continue
        published = struct_time_to_datetime(entry.get("published_parsed")) or struct_time_to_datetime(
            entry.get("updated_parsed")
        )
        entries.append(
            CandidateURL(
                url=url,
                source_name=source_name,
                domain=domain,
                discovery_method=method,
                discovered_at=now,
                title=entry.get("title"),
                published_at=published,
                feed_summary=entry.get("summary"),
                feed_author=entry.get("author"),
            )
        )
    return entries


# ---------------------------------------------------------------------------
# stdlib fallback backend
# ---------------------------------------------------------------------------


def _parse_with_stdlib(content: str, *, source_name: str, domain: str) -> List[CandidateURL]:
    root = ET.fromstring(content)
    now = datetime.now(timezone.utc)
    tag = root.tag.lower()

    if tag.endswith("rss") or root.find("channel") is not None:
        return _parse_rss_items(root, source_name=source_name, domain=domain, now=now)
    if tag.endswith("feed"):
        return _parse_atom_entries(root, source_name=source_name, domain=domain, now=now)

    logger.warning("Unrecognized feed root tag <%s> for %s", root.tag, source_name)
    return []


def _parse_rss_items(root: ET.Element, *, source_name: str, domain: str, now: datetime) -> List[CandidateURL]:
    channel = root.find("channel")
    items = channel.findall("item") if channel is not None else []
    entries: List[CandidateURL] = []
    for item in items:
        url = _text(item, "link")
        if not url:
            continue
        entries.append(
            CandidateURL(
                url=url,
                source_name=source_name,
                domain=domain,
                discovery_method=DiscoveryMethod.RSS,
                discovered_at=now,
                title=_text(item, "title"),
                published_at=parse_date(_text(item, "pubDate")),
                feed_summary=_text(item, "description"),
                feed_author=_text(item, "author") or _text(item, f"{_DC_NS}creator"),
            )
        )
    return entries


def _parse_atom_entries(root: ET.Element, *, source_name: str, domain: str, now: datetime) -> List[CandidateURL]:
    entries: List[CandidateURL] = []
    for entry in root.findall(f"{_ATOM_NS}entry"):
        url = _atom_link(entry)
        if not url:
            continue

        author = None
        author_el = entry.find(f"{_ATOM_NS}author/{_ATOM_NS}name")
        if author_el is not None:
            author = author_el.text

        entries.append(
            CandidateURL(
                url=url,
                source_name=source_name,
                domain=domain,
                discovery_method=DiscoveryMethod.ATOM,
                discovered_at=now,
                title=_text(entry, f"{_ATOM_NS}title"),
                published_at=parse_date(_text(entry, f"{_ATOM_NS}updated") or _text(entry, f"{_ATOM_NS}published")),
                feed_summary=_text(entry, f"{_ATOM_NS}summary") or _text(entry, f"{_ATOM_NS}content"),
                feed_author=author,
            )
        )
    return entries


def _atom_link(entry: ET.Element) -> Optional[str]:
    links = entry.findall(f"{_ATOM_NS}link")
    for link in links:
        if link.get("rel", "alternate") == "alternate" and link.get("href"):
            return link.get("href")
    if links:
        return links[0].get("href")
    return None


def _text(elem: ET.Element, tag: str) -> Optional[str]:
    child = elem.find(tag)
    if child is None or child.text is None:
        return None
    text = child.text.strip()
    return text or None
