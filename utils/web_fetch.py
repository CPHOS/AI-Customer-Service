"""Fetches content from the CPHOS official website via WordPress REST API.

The CPHOS website is a WordPress site with JS-rendered front-end, so plain
HTML fetching returns empty content. Instead, we use the public WP REST API
(``/wp-json/wp/v2/posts``) which returns structured JSON with full text.

Only a hard-coded set of page keys is allowed. The LLM picks a key
(not a raw URL), and this module resolves it to the appropriate API query.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from html.parser import HTMLParser

from utils.logger import get_logger

logger = get_logger(__name__)

# ── WordPress REST API base ──────────────────────────────────────────────────
_WP_API_BASE = "https://cphos.cn/wp-json/wp/v2"

# category slug → WP category ID (from /wp-json/wp/v2/categories)
_CATEGORY_IDS: dict[str, int] = {
    "notification": 7,    # 联考通知
    "events":       30,   # 活动
    "data":         4,    # 数据分析
    "exam-analysis": 3,   # 试题分析
    "about-us":     6,    # 关于我们
}

# ── Security: hard-coded page key allowlist ──────────────────────────────────
# The LLM picks one of these keys. Each key maps to a WP REST API query.

ALLOWED_PAGES: dict[str, str] = {
    "notification": "联考通知 — 最新联考公告、报名与赛历",
    "events":       "往期活动 — CPHOS 活动回顾",
    "about":        "关于我们 — CPHOS 组织介绍",
    "latest":       "最新文章 — 全站最新发布的内容",
}

_FETCH_TIMEOUT = 10      # seconds
_MAX_CHARS     = 4_000   # characters returned to the model
_MAX_POSTS     = 5       # number of posts to fetch per query


# ── HTML strip helper ────────────────────────────────────────────────────────

class _HTMLStripper(HTMLParser):
    """Minimal HTML tag stripper for WP excerpt/content fields."""

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self._parts.append(data)

    def get_text(self) -> str:
        return "".join(self._parts).strip()


def _strip_html(html: str) -> str:
    """Remove HTML tags and return plain text."""
    s = _HTMLStripper()
    s.feed(html)
    return s.get_text()


# ── Internal fetcher ─────────────────────────────────────────────────────────

def _api_get(url: str) -> list | dict:
    """GET a WP REST API endpoint and return parsed JSON."""
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "CPHOS-AI-Chatbot/1.0",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=_FETCH_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _fetch_posts(category_id: int | None = None, per_page: int = _MAX_POSTS) -> str:
    """Fetch recent posts, optionally filtered by category, and format as text."""
    params = f"per_page={per_page}&_fields=title,date,link,excerpt"
    if category_id is not None:
        params += f"&categories={category_id}"
    url = f"{_WP_API_BASE}/posts?{params}"

    data = _api_get(url)
    if not data:
        return "(该分类下暂无文章)"

    lines: list[str] = []
    for post in data:
        title = _strip_html(post.get("title", {}).get("rendered", ""))
        date  = post.get("date", "")[:10]   # YYYY-MM-DD
        link  = post.get("link", "")
        excerpt_html = post.get("excerpt", {}).get("rendered", "")
        excerpt = _strip_html(excerpt_html)
        # Clean up WP's [...] and &nbsp;
        excerpt = re.sub(r"\s*\[&hellip;\]", "…", excerpt)
        excerpt = excerpt.replace("\xa0", " ").strip()

        lines.append(f"📌 {title}")
        lines.append(f"   日期: {date}")
        if excerpt:
            lines.append(f"   摘要: {excerpt}")
        if link:
            lines.append(f"   链接: {link}")
        lines.append("")

    return "\n".join(lines).strip()


def _fetch_page_by_slug(slug: str) -> str:
    """Fetch a single WordPress page by slug (for 'about' etc.)."""
    url = f"{_WP_API_BASE}/pages?slug={slug}&_fields=title,content"
    data = _api_get(url)
    if not data:
        return "(未找到该页面)"
    page = data[0]
    title = _strip_html(page.get("title", {}).get("rendered", ""))
    content = _strip_html(page.get("content", {}).get("rendered", ""))
    return f"{title}\n\n{content}"


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_page(page_key: str, max_chars: int = _MAX_CHARS) -> str:
    """Fetch CPHOS content by page_key via WordPress REST API.

    Args:
        page_key:  One of the keys in :data:`ALLOWED_PAGES`.
        max_chars: Maximum characters to return (excess is truncated).

    Returns:
        Formatted plain text, or an ``[Error: …]`` string on failure.

    Raises:
        ValueError: if *page_key* is not in the allowlist.
    """
    if page_key not in ALLOWED_PAGES:
        valid = ", ".join(sorted(ALLOWED_PAGES))
        raise ValueError(
            f"Unknown page_key '{page_key}'. Valid keys: {valid}"
        )

    try:
        if page_key == "notification":
            text = _fetch_posts(category_id=_CATEGORY_IDS["notification"])
        elif page_key == "events":
            text = _fetch_posts(category_id=_CATEGORY_IDS["events"])
        elif page_key == "about":
            text = _fetch_page_by_slug("organization")
        elif page_key == "latest":
            text = _fetch_posts()   # no category filter → all recent
        else:
            text = "(unknown page_key)"
    except urllib.error.HTTPError as exc:
        logger.warning("fetch_page HTTP %s for key '%s'", exc.code, page_key)
        return f"[Error: HTTP {exc.code} when fetching {page_key}]"
    except urllib.error.URLError as exc:
        logger.warning("fetch_page URLError for key '%s': %s", page_key, exc.reason)
        return f"[Error: could not reach CPHOS API — {exc.reason}]"
    except Exception as exc:
        logger.warning("fetch_page error for key '%s': %s", page_key, exc)
        return f"[Error: {exc}]"

    if len(text) > max_chars:
        text = text[:max_chars] + "\n…[content truncated]"

    logger.info("fetch_page[%s]: fetched %d chars via WP REST API", page_key, len(text))
    return text or "[Error: no content returned]"
