"""Fetches plain text from CPHOS official web pages.

Security: only URLs whose host is cphos.cn (or a subdomain) are allowed.
All other domains raise ValueError so the LLM-provided URL is always
validated before any network request is made.
"""
from __future__ import annotations

import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser

from utils.logger import get_logger

logger = get_logger(__name__)

# ── Security: hard-coded domain allowlist ────────────────────────────────────
_ALLOWED_BASE_DOMAIN = "cphos.cn"

_FETCH_TIMEOUT = 10      # seconds
_MAX_CHARS     = 4_000   # characters returned to the model


# ── HTML → plain-text extractor ──────────────────────────────────────────────

class _TextExtractor(HTMLParser):
    """Strips tags; accumulates visible text, discarding script/style blocks."""

    _SKIP_TAGS = frozenset({"script", "style", "head", "meta", "link", "noscript"})

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.lower() in self._SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self._SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            stripped = data.strip()
            if stripped:
                self._parts.append(stripped)

    def get_text(self) -> str:
        return "\n".join(self._parts)


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_page(url: str, max_chars: int = _MAX_CHARS) -> str:
    """Fetch *url* and return extracted plain text (at most *max_chars* chars).

    Args:
        url:       The page to fetch.  Must be under cphos.cn.
        max_chars: Maximum characters to return (excess is truncated).

    Returns:
        Extracted plain text, or an ``[Error: …]`` string on network failure
        so that the calling LLM can handle unavailable pages gracefully.

    Raises:
        ValueError: if the URL's host is not under the allowed domain.
    """
    parsed = urllib.parse.urlparse(url)
    netloc = parsed.netloc.lower()
    if netloc != _ALLOWED_BASE_DOMAIN and not netloc.endswith("." + _ALLOWED_BASE_DOMAIN):
        raise ValueError(
            f"Fetching '{parsed.netloc}' is not permitted. "
            f"Only pages under {_ALLOWED_BASE_DOMAIN} may be accessed."
        )

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "text/html,application/xhtml+xml,*/*",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=_FETCH_TIMEOUT) as resp:
            ct      = resp.headers.get("Content-Type", "")
            charset = "utf-8"
            if "charset=" in ct:
                charset = ct.split("charset=")[-1].strip().split(";")[0].strip()
            html = resp.read().decode(charset, errors="replace")
    except urllib.error.HTTPError as exc:
        logger.warning("fetch_page HTTP %s: %s", exc.code, url)
        return f"[Error: HTTP {exc.code} when fetching {url}]"
    except urllib.error.URLError as exc:
        logger.warning("fetch_page URLError for %s: %s", url, exc.reason)
        return f"[Error: could not reach {url} — {exc.reason}]"
    except Exception as exc:
        logger.warning("fetch_page unexpected error for %s: %s", url, exc)
        return f"[Error: {exc}]"

    parser = _TextExtractor()
    parser.feed(html)
    text = parser.get_text()

    if len(text) > max_chars:
        text = text[:max_chars] + "\n…[content truncated]"

    logger.info("fetch_page: fetched %d chars from %s", len(text), url)
    return text or "[Error: page has no extractable text content]"
