"""Web image search for the face finder.

Two key-free providers, both with documented public APIs:

* **Wikimedia Commons** — excellent coverage of public figures, high
  resolution, and everything on it is freely licensed. The default.
* **Openverse** — a broader index of openly-licensed images, useful when
  Commons has nothing.

DuckDuckGo was the obvious candidate and is deliberately not here: its image
endpoint is undocumented and now returns 403 behind bot protection, and
working around that is not something this app should be doing. Sources with
real APIs are also the ones whose licensing is clear.

Nothing here downloads full images — that happens later, only for the
results the user actually picks, via ui_face_browser.fetch_image().
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, List, Optional

import requests

# Wikimedia asks API clients to identify themselves.
USER_AGENT = "Morphify/1.0 (desktop face library; onnxruntime)"

TIMEOUT = 25


@dataclass(frozen=True)
class SearchResult:
    title: str
    thumbnail_url: str
    image_url: str
    width: int = 0
    height: int = 0
    source: str = ""

    @property
    def label(self) -> str:
        return self.title or self.source or self.image_url


class SearchError(RuntimeError):
    """A provider failed in a way worth showing the user."""


def _clean_title(title: str) -> str:
    title = re.sub(r"^File:", "", title or "")
    return re.sub(r"\.(jpg|jpeg|png|webp|gif|tif|tiff)$", "", title, flags=re.I)


# ─── Wikimedia Commons ───────────────────────────────────────────────────


# Words people naturally add to an image search that mean "a picture of",
# and which a literal text index will simply fail to match.
_PHOTO_QUALIFIERS = {
    "portrait", "portraits", "photo", "photos", "photograph", "picture",
    "pictures", "image", "images", "headshot", "headshots", "close", "closeup",
    "up", "face", "faces", "hd", "4k", "high", "res", "resolution", "profile",
    "selfie", "pic", "pics", "shot",
}


def relax_query(query: str) -> str:
    """Drop generic photo words, keeping the actual subject.

    Wikimedia Commons matches text literally, so "kai cenat portrait" finds
    nothing while "kai cenat" finds plenty. People type the first form, so
    the search has to cope with it rather than shrugging.
    """
    words = [w for w in re.split(r"\s+", (query or "").strip()) if w]
    kept = [w for w in words if w.lower().strip(",.-") not in _PHOTO_QUALIFIERS]
    return " ".join(kept) if kept else (query or "").strip()


def search_wikimedia(query: str, limit: int = 30,
                     safe: bool = True) -> List[SearchResult]:
    results = _wikimedia_once(query, limit)
    if not results:
        relaxed = relax_query(query)
        if relaxed and relaxed.lower() != query.strip().lower():
            results = _wikimedia_once(relaxed, limit)
    return results


def _wikimedia_once(query: str, limit: int) -> List[SearchResult]:
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": "6",          # the File: namespace
        "gsrlimit": str(min(limit, 50)),
        "prop": "imageinfo",
        "iiprop": "url|size|mime",
        "iiurlwidth": "320",          # server-rendered thumbnail
        "format": "json",
    }
    response = requests.get(
        "https://commons.wikimedia.org/w/api.php",
        params=params, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
    response.raise_for_status()
    pages = (response.json().get("query") or {}).get("pages") or {}

    results = []
    for page in pages.values():
        info = (page.get("imageinfo") or [{}])[0]
        url = info.get("url")
        mime = info.get("mime") or ""
        if not url or not mime.startswith("image/"):
            continue
        # Vector and multi-page formats are not portraits.
        if mime in ("image/svg+xml", "image/tiff"):
            continue
        results.append(SearchResult(
            title=_clean_title(page.get("title", "")),
            thumbnail_url=info.get("thumburl") or url,
            image_url=url,
            width=int(info.get("width") or 0),
            height=int(info.get("height") or 0),
            source="Wikimedia Commons",
        ))
    return results[:limit]


# ─── Openverse ───────────────────────────────────────────────────────────


def search_openverse(query: str, limit: int = 30,
                     safe: bool = True) -> List[SearchResult]:
    params = {
        "q": query,
        "page_size": str(min(limit, 50)),
        "mature": "false" if safe else "true",
    }
    response = requests.get(
        "https://api.openverse.org/v1/images/",
        params=params, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
    response.raise_for_status()

    results = []
    for item in response.json().get("results", [])[:limit]:
        url = item.get("url") or ""
        if not url:
            continue
        results.append(SearchResult(
            title=item.get("title", ""),
            thumbnail_url=item.get("thumbnail") or url,
            image_url=url,
            width=int(item.get("width") or 0),
            height=int(item.get("height") or 0),
            source=item.get("source", "Openverse"),
        ))
    return results


PROVIDERS = {
    "Wikimedia": search_wikimedia,
    "Openverse": search_openverse,
}

DEFAULT_PROVIDER = "Wikimedia"


def search(query: str, provider: str = DEFAULT_PROVIDER, limit: int = 30,
           safe: bool = True) -> List[SearchResult]:
    """Run a search, raising SearchError with something worth displaying."""
    query = (query or "").strip()
    if not query:
        return []

    finder: Optional[Callable] = PROVIDERS.get(provider)
    if finder is None:
        raise SearchError(f"Unknown image source: {provider}")

    try:
        return finder(query, limit=limit, safe=safe)
    except SearchError:
        raise
    except requests.RequestException as exc:
        raise SearchError(f"Could not reach {provider}: {exc}") from exc
    except Exception as exc:
        raise SearchError(f"{provider} search failed: {exc}") from exc


def search_all(query: str, limit: int = 30, safe: bool = True) -> List[SearchResult]:
    """Query every provider, best coverage first, ignoring individual failures.

    A single dead provider should thin the results, not empty the grid.
    """
    collected: List[SearchResult] = []
    errors = []
    for name in ("Wikimedia", "Openverse"):
        try:
            collected.extend(search(query, provider=name, limit=limit, safe=safe))
        except SearchError as exc:
            errors.append(str(exc))
        if len(collected) >= limit:
            break
    if not collected and errors:
        raise SearchError(errors[0])
    return collected[:limit]


def slugify(text: str, fallback: str = "face") -> str:
    """Turn a search query into a safe filename stem.

    Faces get saved under the term that found them, which is the point of
    the search box: a library of ``random-20260902-153001.jpg`` cannot be
    searched, one of ``lebron-james-01.jpg`` can.
    """
    cleaned = re.sub(r"[^\w\s-]", "", (text or "").strip().lower())
    cleaned = re.sub(r"[\s_]+", "-", cleaned).strip("-")
    return cleaned[:48] or fallback
