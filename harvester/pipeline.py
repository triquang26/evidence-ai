from __future__ import annotations
import csv
import json
import re

from .config import Config


def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())


class Deduplicator:
    def __init__(self, config: Config):
        self.config = config

    def run(self, rows: list[dict]) -> list[dict]:
        by_key: dict[str, dict] = {}
        for r in rows:
            title = r.get("title") or ""
            if len(title) < self.config.title_min_len:
                continue
            if not self._is_relevant(title + " " + r.get("abstract", "")):
                continue
            key = r.get("arxiv_id") or r.get("doi") or _norm_title(title)
            if key in by_key:
                self._merge(by_key[key], r)
            else:
                by_key[key] = dict(r)

        papers = list(by_key.values())
        papers = self._filter_venue(papers)
        self._tag(papers)
        return papers

    def _is_relevant(self, text: str) -> bool:
        t = text.lower()
        return any(k in t for k in self.config.keep_keywords)

    @staticmethod
    def _filter_venue(papers: list[dict]) -> list[dict]:
        return [p for p in papers if p.get("venue")]

    @staticmethod
    def _tag(papers: list[dict]) -> None:
        for p in papers:
            tags = []
            if p.get("is_survey"):
                tags.append("survey")
            p["tags"] = tags

    @staticmethod
    def _merge(cur: dict, new: dict) -> None:
        for f in ("venue", "arxiv_id", "url", "doi", "abstract"):
            if not cur.get(f) and new.get(f):
                cur[f] = new[f]
        cur["is_survey"] = cur["is_survey"] or new.get("is_survey", False)
        if new.get("source") and new["source"] not in cur.get("source", ""):
            cur["source"] = cur.get("source", "") + "+" + new["source"]


class Exporter:
    _FIELDS = ["title", "venue", "year_guess", "arxiv_id", "doi", "url", "tags", "source"]

    def __init__(self, config: Config):
        self.config = config

    def save(self, papers: list[dict]) -> None:
        self.config.out_dir.mkdir(parents=True, exist_ok=True)
        self._save_jsonl(papers)
        self._save_csv(papers)

    def _save_jsonl(self, papers: list[dict]) -> None:
        path = self.config.out_dir / "papers.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for p in papers:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"Saved: {path}")

    def _save_csv(self, papers: list[dict]) -> None:
        path = self.config.out_dir / "papers.csv"
        yr_re = re.compile(r"(\d{4})")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self._FIELDS)
            w.writeheader()
            for p in papers:
                yr = yr_re.search(p.get("venue") or "")
                w.writerow({
                    "title": p["title"],
                    "venue": p.get("venue"),
                    "year_guess": yr.group(1) if yr else "",
                    "arxiv_id": p.get("arxiv_id"),
                    "doi": p.get("doi"),
                    "url": p.get("url"),
                    "tags": ",".join(p.get("tags") or []),
                    "source": p.get("source"),
                })
        print(f"Saved: {path}")
