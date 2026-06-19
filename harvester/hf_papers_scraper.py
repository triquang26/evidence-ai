from __future__ import annotations
import sys
import time

import requests

from .config import Config

UA = {"User-Agent": "ts-research-harvester/1.0 (academic use)"}
_HF_PAPERS_URL = "https://huggingface.co/api/papers"


class HFPapersScraper:
    def __init__(self, config: Config, max_per_query: int = 1000):
        self.config = config
        self.max_per_query = max_per_query

    def harvest(self) -> list[dict]:
        rows: list[dict] = []
        for q in self.config.arxiv_queries:
            batch = self._fetch_query(q)
            rows.extend(batch)
            print(f"  hf-papers '{q}': {len(batch)}")
            time.sleep(0.5)
        return rows

    def _is_on_topic(self, title: str, abstract: str) -> bool:
        text = (title + " " + abstract).lower()
        has_ts   = any(k in text for k in self.config.ts_keywords)
        has_task = any(k in text for k in self.config.task_keywords)
        return has_ts and has_task

    def _fetch_query(self, query: str) -> list[dict]:
        rows: list[dict] = []
        page = 1
        while len(rows) < self.max_per_query:
            try:
                resp = requests.get(
                    _HF_PAPERS_URL,
                    params={"q": query, "limit": 100, "p": page},
                    headers=UA,
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                print(f"  [hf-papers err] {query} p{page}: {e}", file=sys.stderr)
                break

            if not data:
                break

            for p in data:
                arxiv_id = p.get("id", "")
                title    = p.get("title", "")
                abstract = p.get("summary") or p.get("ai_summary") or ""
                # HF Papers API returns trending papers — filter to TS tasks only
                if not self._is_on_topic(title, abstract):
                    continue
                rows.append({
                    "title": title,
                    "venue": None,
                    "arxiv_id": arxiv_id,
                    "url": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else None,
                    "doi": None,
                    "abstract": abstract,
                    "is_survey": False,
                    "source": "hf_papers",
                })

            if len(data) < 100:
                break
            page += 1

        return rows
