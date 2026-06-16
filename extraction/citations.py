from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import requests

UA = {"User-Agent": "evidence-ai/1.0 (academic use)"}
_HF_PAPERS_URL = "https://huggingface.co/api/papers/search"
_ARXIV_RE = re.compile(r"arxiv[:\s]*?(\d{4}\.\d{4,5})", re.I)
_YEAR_RE = re.compile(r"\b(19|20)\d{2}[a-z]?\b")
_REF_HEADING = re.compile(r"^#+\s*(references|bibliography)\s*$", re.I | re.M)
_FIRST_SURNAME = re.compile(r"^([A-Z][a-zA-Z\-']+)")
_AUTHOR_YEAR = re.compile(r"([A-Z][a-zA-Z\-']+)\s+et al\.?,?\s*\(?((?:19|20)\d{2})")
# "<authors>. <year>. <title>. <venue>" -> capture the title between the year and the next period.
_TITLE_AFTER_YEAR = re.compile(r"(?:19|20)\d{2}[a-z]?\.\s*([^.]{8,200})\.")


@dataclass(frozen=True)
class BibEntry:
    index: str | None       # "12" for [12]/12. numbered styles
    surname: str            # first author's surname (author-year matching)
    year: str
    arxiv_id: str
    title: str
    text: str


class BibliographyParser:
    """Parse the References section of a parsed-markdown paper into BibEntry rows.

    Handles both numbered ("[12] ...") and author-year ("Author et al. 2022. Title...")
    styles; multi-line entries separated by blank lines are merged into one block.
    """

    def parse(self, markdown: str) -> list[BibEntry]:
        m = _REF_HEADING.search(markdown)
        if not m:
            return []
        body = markdown[m.end():]

        # Merge continuation lines into reference blocks.
        # A new block starts at a blank line, a [n]/n. numbered marker, or a section heading.
        blocks: list[str] = []
        current: list[str] = []
        for line in body.splitlines():
            stripped = " ".join(line.split())
            if stripped.startswith("#"):
                break
            if not stripped:
                if current:
                    blocks.append(" ".join(current))
                    current = []
                continue
            # Numbered-style marker always starts a new entry even without a blank line gap
            if re.match(r"\[?\d+[\]\.]\s", stripped) and current:
                blocks.append(" ".join(current))
                current = []
            current.append(stripped)
        if current:
            blocks.append(" ".join(current))

        entries = []
        for chunk in blocks:
            if len(chunk) < 25 or not _YEAR_RE.search(chunk):
                continue
            idx = re.match(r"\[?(\d+)[\]\.]\s", chunk)
            body_txt = chunk[idx.end():] if idx else chunk
            first_author = re.split(r",| and ", body_txt)[0].strip()
            # Strip "et al." so "Brown et al." → surname "Brown", not "al."
            clean_author = re.sub(r"\s+et\s+al\.?$", "", first_author, flags=re.I).strip()
            parts = (clean_author or first_author).split()
            surname = parts[-1] if parts else ""
            arxiv = _ARXIV_RE.search(chunk)
            year = _YEAR_RE.search(chunk)
            title_m = _TITLE_AFTER_YEAR.search(chunk)
            # Reject venue fragments with fewer than 3 words ("Featured Certification")
            raw_title = title_m.group(1).strip() if title_m else ""
            title = raw_title if len(raw_title.split()) >= 3 else ""
            entries.append(BibEntry(
                index=idx.group(1) if idx else None,
                surname=surname,
                year=year.group(0)[:4] if year else "",
                arxiv_id=arxiv.group(1) if arxiv else "",
                title=title,
                text=chunk[:800],
            ))
        return entries


class HFPapersClient:
    """Resolve a paper title to an arxiv id via the (firewall-safe) HF Papers API."""

    def __init__(self):
        self._cache: dict[str, str] = {}

    def title_to_arxiv(self, title: str) -> str:
        title = (title or "").strip()
        if len(title) < 8:
            return ""
        if title in self._cache:
            return self._cache[title]
        arxiv_id = ""
        try:
            resp = requests.get(_HF_PAPERS_URL, params={"q": title}, headers=UA, timeout=20)
            resp.raise_for_status()
            for p in resp.json()[:5]:
                # Response shape: {"title": "...", "paper": {"id": "2304.xxxx", ...}, ...}
                p_title = (p.get("title") or p.get("paper", {}).get("title") or "").lower()
                p_id = p.get("paper", {}).get("id") or p.get("id") or ""
                ratio = difflib.SequenceMatcher(None, title.lower(), p_title).ratio()
                if ratio >= 0.6 and p_id:
                    arxiv_id = p_id
                    break
        except Exception:
            arxiv_id = ""
        self._cache[title] = arxiv_id
        return arxiv_id


class CitationResolver:
    """Map an LLM-extracted citation marker to the cited paper's arxiv id / url / ref text.

    Only resolves markers that the LLM explicitly extracted as subject_citation.
    No guessing, no fallbacks based on subject name alone.
    """

    def __init__(self, hf_client: HFPapersClient | None = None):
        self.hf = hf_client

    def resolve(self, citation: str, bib: list[BibEntry]) -> dict:
        """Resolve a citation marker to arxiv metadata.

        Args:
            citation: the exact marker the LLM extracted, e.g. "[12]" or "Brown et al., 2020"
            bib: parsed bibliography of the paper

        Returns dict with subject_arxiv_id, subject_arxiv_url, subject_ref.
        All values are empty string if unresolvable — never None.
        """
        entry = self._match_bib(citation, bib)
        arxiv_id = entry.arxiv_id if entry else ""
        title = (entry.title or "") if entry else ""
        ref_text = (entry.text or "") if entry else ""

        # If bib entry has no embedded arxiv id, try HF Papers search by title
        if not arxiv_id and self.hf and title:
            arxiv_id = self.hf.title_to_arxiv(title)

        return {
            "subject_arxiv_id": arxiv_id,
            "subject_arxiv_url": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else "",
            "subject_ref": (title or ref_text)[:200],
        }

    @staticmethod
    def _match_bib(citation: str, bib: list[BibEntry]) -> BibEntry | None:
        """Match a citation marker to a bibliography entry.

        Supports:
        - Numeric: "[12]" or "12" → entry whose index == "12"
        - Author-year: "Brown et al., 2020" or "Brown 2020" → entry with matching surname + year
        Returns None if no match; never falls back on subject name.
        """
        cite = (citation or "").strip()
        if not cite:
            return None

        # Numeric citation: [12] or bare 12
        if re.match(r"\[?\d", cite):
            num = re.findall(r"\d+", cite)[0]
            for e in bib:
                if e.index == num:
                    return e
            return None  # numeric but not found — don't fall through to author-year

        # Author-year citation
        ay = _AUTHOR_YEAR.search(cite) or _FIRST_SURNAME.match(cite)
        year = re.search(r"(?:19|20)\d{2}", cite)
        if ay:
            surname = ay.group(1).lower()
            for e in bib:
                if e.surname.lower() == surname and (not year or year.group(0) == e.year):
                    return e

        return None


def enrich_jsonl(markdown: str, jsonl_path: Path, resolver: CitationResolver) -> int:
    """Resolve LLM-extracted subject_citation markers to arxiv IDs.

    Only processes extractions where the LLM set subject_citation explicitly.
    Does NOT attempt to guess or fill missing citations — that would introduce noise.
    """
    lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return 0
    doc = json.loads(lines[0])
    extractions = doc.get("extractions", [])
    bib = BibliographyParser().parse(markdown)

    n = 0
    for ext in extractions:
        attrs = ext.setdefault("attributes", {})
        citation = attrs.get("subject_citation", "")
        if not citation:
            continue  # only resolve what the LLM explicitly extracted
        resolved = resolver.resolve(citation, bib)
        if resolved["subject_arxiv_id"] or resolved["subject_ref"]:
            attrs.update(resolved)
            n += 1

    jsonl_path.write_text(json.dumps(doc, ensure_ascii=False) + "\n", encoding="utf-8")
    return n
