from __future__ import annotations
import re
import time

import requests

from .config import Config

UA = {"User-Agent": "ts-research-harvester/1.0 (academic use)"}


def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())


class PDFDownloader:
    def __init__(self, config: Config):
        self.config = config

    def download(self, papers: list[dict]) -> None:
        self.config.pdf_dir.mkdir(parents=True, exist_ok=True)
        ok = skip = fail = 0
        for i, p in enumerate(papers):
            url = self._pick_url(p)
            if not url:
                continue
            name = (p.get("arxiv_id") or _norm_title(p["title"])[:60]).replace("/", "_")
            path = self.config.pdf_dir / f"{name}.pdf"
            if path.exists():
                skip += 1
                continue
            if self._fetch(url, path):
                ok += 1
            else:
                fail += 1
            if i % 25 == 0:
                print(f"  pdf {i}/{len(papers)}  ok={ok} skip={skip} fail={fail}")
            time.sleep(1)
        print(f"PDF done: ok={ok}  skip(cached)={skip}  fail={fail}")

    @staticmethod
    def _pick_url(p: dict) -> str | None:
        if p.get("arxiv_id"):
            return f"https://arxiv.org/pdf/{p['arxiv_id']}.pdf"
        u = p.get("url") or ""
        if u.endswith(".pdf") or "openaccess" in u or "/pdf/" in u:
            return u
        return None

    @staticmethod
    def _fetch(url: str, path) -> bool:
        for attempt in range(4):
            try:
                resp = requests.get(url, headers=UA, timeout=90)
                resp.raise_for_status()
                if resp.content[:4] != b"%PDF":
                    raise ValueError("not a PDF")
                path.write_bytes(resp.content)
                return True
            except Exception:
                time.sleep(2 ** attempt)
        return False
