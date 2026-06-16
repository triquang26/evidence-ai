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
# Inline citation patterns used by fill_missing_citations
_CITE_NUMERIC = re.compile(r"\[(\d+(?:[,;]\s*\d+)*)\]")
_CITE_AUTHYEAR = re.compile(
    r"([A-Z][a-zA-Z\-']+(?:\s+et\s+al\.?)?),?\s*\(?((?:19|20)\d{2})\)?"
)


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
    styles; each reference line/block becomes one entry.
    """

    def parse(self, markdown: str) -> list[BibEntry]:
        m = _REF_HEADING.search(markdown)
        if not m:
            return []
        body = markdown[m.end():]

        # Merge continuation lines into reference blocks.
        # A new block starts at: a blank line separator, a [n]/n. numbered marker, or the
        # next section heading. This handles both single-line and multi-line bib formats.
        blocks: list[str] = []
        current: list[str] = []
        for line in body.splitlines():
            stripped = " ".join(line.split())
            if stripped.startswith("#"):
                break  # next section
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
            # Strip "et al." so "Brown et al." yields surname "Brown", not "al."
            clean_author = re.sub(r"\s+et\s+al\.?$", "", first_author, flags=re.I).strip()
            parts = (clean_author or first_author).split()
            surname = parts[-1] if parts else ""
            arxiv = _ARXIV_RE.search(chunk)
            year = _YEAR_RE.search(chunk)
            title_m = _TITLE_AFTER_YEAR.search(chunk)
            # Reject venue fragments with fewer than 3 words ("Featured Certification", "arXiv preprint")
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
    """Map a subject's inline citation to the cited paper's arxiv id / url."""

    def __init__(self, hf_client: HFPapersClient | None = None):
        self.hf = hf_client

    def resolve(self, subject: str, citation: str, bib: list[BibEntry]) -> dict:
        entry = self._match(subject, citation, bib)
        arxiv_id = entry.arxiv_id if entry else ""
        title = (entry.title or "") if entry else ""
        ref_text = (entry.text or "") if entry else ""
        if not arxiv_id and self.hf and title:
            arxiv_id = self.hf.title_to_arxiv(title)
        return {
            "subject_arxiv_id": arxiv_id,
            "subject_arxiv_url": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else "",
            "subject_ref": (title or ref_text)[:200],
        }

    @staticmethod
    def _match(subject: str, citation: str, bib: list[BibEntry]) -> BibEntry | None:
        cite = (citation or "").strip()
        if re.match(r"\[?\d", cite):  # numeric citation [12] -> bib entry with that index
            num = re.findall(r"\d+", cite)[0]
            for e in bib:
                if e.index == num:
                    return e
        ay = _AUTHOR_YEAR.search(cite) or _FIRST_SURNAME.match(cite)
        year = re.search(r"(?:19|20)\d{2}", cite)
        if ay:  # "Author et al., 2022" -> entry with that surname (+ year if given)
            surname = ay.group(1).lower()
            for e in bib:
                if e.surname.lower() == surname and (not year or year.group(0) == e.year):
                    return e
        if subject:  # fall back: subject name appears verbatim in an entry
            for e in bib:
                if subject.lower() in e.text.lower():
                    return e
        return None


def _paper_body(markdown: str) -> str:
    """Return the text before the References/Bibliography section."""
    m = _REF_HEADING.search(markdown)
    return markdown[:m.start()] if m else markdown


# Patterns for "SubjectName [12]" or "SubjectName (Author, year)" in body text
_GLOBAL_NUM_RE = re.compile(
    r"([A-Z][A-Za-z0-9](?:[A-Za-z0-9\-\.]*(?:\s+[A-Za-z0-9\-\.]+){0,4})?)\s*\[(\d+)\]"
)
_GLOBAL_AUTH_RE = re.compile(
    r"([A-Z][A-Za-z0-9](?:[A-Za-z0-9\-\.]*(?:\s+[A-Za-z0-9\-\.]+){0,4})?)"
    r"\s*\(([A-Z][a-zA-Z\-']+(?:\s+et\s+al\.?)?,?\s*(?:19|20)\d{2})\)"
)


def _build_citation_map(body: str) -> dict[str, str]:
    """Scan the entire paper body and map subject-name (lower) → citation marker.

    Captures both "Name [n]" and "Name (Author, year)" styles.  First occurrence wins
    (earlier mention is usually the defining/introducing one).
    """
    cmap: dict[str, str] = {}
    for m in _GLOBAL_NUM_RE.finditer(body):
        key = m.group(1).strip().lower()
        if key and key not in cmap:
            cmap[key] = f"[{m.group(2)}]"
    for m in _GLOBAL_AUTH_RE.finditer(body):
        key = m.group(1).strip().lower()
        if key and key not in cmap:
            cmap[key] = m.group(2).strip()
    return cmap


def fill_missing_citations(markdown: str, extractions: list[dict]) -> int:
    """Regex post-processing: recover subject_citation markers the LLM missed.

    Two-phase search for each baseline/ablation without a citation:
    1. Local: scan a window around the extraction's char_interval for an adjacent marker.
    2. Global: match against a citation map built from the whole paper body — catches
       citations in related-work sections that are not adjacent to the results table.
    """
    body = _paper_body(markdown)
    body_len = len(body)
    cmap = _build_citation_map(body)
    filled = 0

    for ext in extractions:
        attrs = ext.get("attributes") or {}
        if attrs.get("subject_citation"):
            continue
        if attrs.get("role") not in ("baseline", "ablation"):
            continue
        subject = (attrs.get("subject") or "").strip()
        if not subject:
            continue

        citation = ""

        # Phase 1: local window search around the char_interval (skip if out-of-body)
        ci = ext.get("char_interval") or {}
        raw_mid = ci.get("start_pos") or 0
        if raw_mid < body_len:
            citation = _find_citation_near_subject(body, subject, raw_mid, radius=600)

        # Phase 2: global citation map lookup
        if not citation:
            citation = cmap.get(subject.lower(), "")
            # Fuzzy: try longest token of multi-word names (e.g. "BEVFormer" from "BEVFormer (large)")
            if not citation and " " in subject:
                for token in subject.split():
                    if len(token) >= 5:
                        citation = cmap.get(token.lower(), "")
                        if citation:
                            break

        if citation:
            attrs["subject_citation"] = citation
            filled += 1

    return filled


def _find_citation_near_subject(body: str, subject: str, mid: int, radius: int) -> str:
    """Search for a citation marker within `radius` chars of `mid`, near the subject name."""
    window_start = max(0, mid - radius)
    window_end = min(len(body), mid + radius)
    window = body[window_start:window_end]

    subj_pat = re.compile(re.escape(subject), re.I)
    subj_m = subj_pat.search(window)
    if not subj_m and " " in subject:
        first = subject.split()[0]
        if len(first) >= 4:
            subj_m = re.compile(re.escape(first), re.I).search(window)
    if not subj_m:
        return ""

    s, e = subj_m.start(), subj_m.end()
    vicinity = window[max(0, s - 30): e + 120]

    num = _CITE_NUMERIC.search(vicinity)
    if num:
        return f"[{num.group(1)}]"
    auth = _CITE_AUTHYEAR.search(vicinity)
    if auth:
        return f"{auth.group(1)}, {auth.group(2)}"
    return ""


def enrich_jsonl(markdown: str, jsonl_path: Path, resolver: CitationResolver) -> int:
    """Fill subject_citation markers (regex), then resolve to arxiv IDs via bib + HF API."""
    lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return 0
    doc = json.loads(lines[0])
    extractions = doc.get("extractions", [])
    bib = BibliographyParser().parse(markdown)

    # Step 1: regex pass to recover citation markers the LLM missed
    fill_missing_citations(markdown, extractions)

    # Step 2: resolve each citation marker to arxiv id / url / ref text
    n = 0
    for ext in extractions:
        attrs = ext.setdefault("attributes", {})
        subject, citation = attrs.get("subject", ""), attrs.get("subject_citation", "")
        if not subject or not (bib or resolver.hf):
            continue
        resolved = resolver.resolve(subject, citation, bib)
        if resolved["subject_arxiv_id"] or resolved["subject_ref"]:
            attrs.update(resolved)
            n += 1

    jsonl_path.write_text(json.dumps(doc, ensure_ascii=False) + "\n", encoding="utf-8")
    return n
