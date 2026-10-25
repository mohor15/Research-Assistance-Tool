"""robots.txt awareness.

`respect_robots_txt: true` in config/settings.yaml is the single switch that
controls this module's behavior — when it is false, `RobotsChecker` allows
everything, matching the requirement that the configured setting is what
decides the behavior, not a hardcoded default. When robots.txt cannot be
fetched at all, the absence is treated as "no restrictions", which is the
conventional interpretation (and never allows a *found* Disallow rule to be
bypassed).
"""

from __future__ import annotations

import logging
from typing import Dict
from urllib import robotparser
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

DEFAULT_ROBOTS_TIMEOUT_SECONDS = 10.0


class RobotsChecker:
    def __init__(
        self,
        *,
        enabled: bool,
        user_agent: str,
        timeout: float = DEFAULT_ROBOTS_TIMEOUT_SECONDS,
    ) -> None:
        self._enabled = enabled
        self._user_agent = user_agent
        self._timeout = timeout
        self._cache: Dict[str, robotparser.RobotFileParser] = {}

    def is_allowed(self, url: str) -> bool:
        if not self._enabled:
            return True

        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        parser = self._cache.get(origin)
        if parser is None:
            parser = self._fetch_and_parse(origin)
            self._cache[origin] = parser

        allowed = parser.can_fetch(self._user_agent, url)
        if not allowed:
            logger.info("robots.txt disallows %s for UA %r", url, self._user_agent)
        return allowed

    def _fetch_and_parse(self, origin: str) -> robotparser.RobotFileParser:
        parser = robotparser.RobotFileParser()
        robots_url = f"{origin}/robots.txt"
        try:
            response = requests.get(
                robots_url, timeout=self._timeout, headers={"User-Agent": self._user_agent}
            )
        except requests.RequestException as exc:
            logger.info("robots.txt unreachable for %s (%s); treating as unrestricted", origin, exc)
            parser.parse([])
            return parser

        if response.status_code >= 400:
            # No robots.txt published => no restrictions, per convention.
            parser.parse([])
        else:
            parser.parse(response.text.splitlines())
        return parser
