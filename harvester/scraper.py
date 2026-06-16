from __future__ import annotations
import re
import sys
import time
import xml.etree.ElementTree as ET

import requests

from .config import Config

UA = {"User-Agent": "ts-research-harvester/1.0 (academic use)"}

_VENUE_RE = re.compile(
    r"(?:in\s+|[\-,|]\s*|\[)\s*\*?([A-Z][A-Za-z&]{1,40}?)\*?['\s,]*\s*(\d{2,4})\s*[\]\.,|]?"
)
_ARXIV_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})", re.I)
_SURVEY_RE = re.compile(r"\b(survey|a review|review on|benchmark)\b", re.I)
_URL_RE = re.compile(r"https?://[^\s\)\]\|]+")
_CITEKEY_RE = re.compile(r"^\[[A-Za-z]+\d{2,4}[a-z]?\]\s*")

_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV_NS = "{http://arxiv.org/schemas/atom}"


def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())


class AwesomeScraper:
    def __init__(self, config: Config):
        self.config = config

    def harvest(self) -> list[dict]:
        rows: list[dict] = []
        for repo, branch in self.config.awesome_lists:
            url = f"https://raw.githubusercontent.com/{repo}/{branch}/README.md"
            try:
                md = requests.get(url, headers=UA, timeout=30).text
            except Exception as e:
                print(f"  [skip] {repo}: {e}", file=sys.stderr)
                continue
            n0 = len(rows)
            rows.extend(self._parse_readme(md, src=f"awesome:{repo.split('/')[-1]}"))
            print(f"  {repo}: +{len(rows) - n0}")
        return rows

    def _parse_readme(self, md: str, src: str) -> list[dict]:
        rows: list[dict] = []
        last_text = ""
        min_len = self.config.title_min_len

        for line in md.splitlines():
            s = line.lstrip()
            is_item = s[:1] in ("*", "-", "+", "|")
            clean = line.replace("\\", "")
            has_url = "http" in clean.lower()

            txt = re.sub(r"\[[^\]]*\]\([^\)]*\)", "", clean).strip(" -*+>#|.").strip()
            if txt and not has_url and len(txt) >= min_len:
                last_text = txt

            if not has_url:
                continue

            urls = _URL_RE.findall(clean)
            paper_urls = [u for u in urls if "github.com" not in u.lower()] or urls
            if not paper_urls:
                continue

            body = _CITEKEY_RE.sub("", clean.lstrip(" -*+>#|").strip())
            m = re.match(r"\[([^\]]{%d,}?)\]\(http" % min_len, body)
            if m:
                title = m.group(1)
            else:
                title = re.split(r",?\s+in\s+\*?[A-Za-z]", body)[0]
                title = re.split(r"\s*\[|\|", title)[0]
            title = title.strip(" .,*")

            if len(title) < min_len or title.lower() in ("paper", "pdf", "link"):
                title = last_text
            if len(title) < min_len:
                continue

            am = _ARXIV_RE.search(clean)
            vm = _VENUE_RE.search(body)

            if not (is_item or am):
                continue
            if not (vm or am or re.search(r"\b(19|20)\d{2}\b", clean)):
                continue

            rows.append({
                "title": title,
                "venue": f"{vm.group(1)} {vm.group(2)}" if vm else None,
                "arxiv_id": am.group(1) if am else None,
                "url": paper_urls[0],
                "doi": None,
                "abstract": "",
                "is_survey": bool(_SURVEY_RE.search(clean) or _SURVEY_RE.search(last_text)),
                "source": src,
            })
        return rows


class ArxivScraper:
    def __init__(self, config: Config):
        self.config = config

    def harvest(self) -> list[dict]:
        rows: list[dict] = []
        for q in self.config.arxiv_queries:
            batch_rows = self._fetch_query(q)
            rows.extend(batch_rows)
            print(f"  arxiv '{q}': ~{len(batch_rows)}")
        return rows

    def _fetch_query(self, query: str) -> list[dict]:
        rows: list[dict] = []
        start = 0
        max_results = self.config.arxiv_max_per_query

        while len(rows) < max_results:
            batch = min(100, max_results - len(rows))
            params = {
                "search_query": f"all:{query}",
                "start": start,
                "max_results": batch,
                "sortBy": "relevance",
            }
            try:
                r = requests.get(
                    "http://export.arxiv.org/api/query",
                    params=params, headers=UA, timeout=60,
                )
                root = ET.fromstring(r.text)
            except Exception as e:
                print(f"  [arxiv err] {query}: {e}", file=sys.stderr)
                break

            entries = root.findall(f"{_ATOM}entry")
            if not entries:
                break

            for e in entries:
                aid_full = e.findtext(f"{_ATOM}id", "")
                aid = aid_full.rsplit("/", 1)[-1].split("v")[0]
                cats = {c.get("term") for c in e.findall(f"{_ATOM}category")}
                if self.config.arxiv_cats and not (cats & self.config.arxiv_cats):
                    continue
                title = " ".join(e.findtext(f"{_ATOM}title", "").split())
                summary = " ".join(e.findtext(f"{_ATOM}summary", "").split())
                jref = e.findtext(f"{_ARXIV_NS}journal_ref")
                doi = e.findtext(f"{_ARXIV_NS}doi")
                rows.append({
                    "title": title,
                    "venue": jref,
                    "arxiv_id": aid,
                    "url": f"https://arxiv.org/abs/{aid}",
                    "doi": doi,
                    "abstract": summary,
                    "is_survey": bool(_SURVEY_RE.search(title + " " + summary)),
                    "source": "arxiv",
                })

            start += len(entries)
            time.sleep(self.config.arxiv_sleep)

        return rows
