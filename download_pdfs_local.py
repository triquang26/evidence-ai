#!/usr/bin/env python3
"""
download_pdfs_local.py — chạy trên máy LOCAL (có internet tới arxiv.org)

Bước 1: Tải papers.jsonl từ GitHub
Bước 2: Tải PDFs từ arxiv.org (có throttle, retry, skip nếu đã có)
Bước 3: Push PDFs lên HF bucket  hf://buckets/twanghcmut/evidence-ai/pdfs/

Chạy:
    pip install requests huggingface_hub
    huggingface-cli login          # login HF 1 lần
    python download_pdfs_local.py

Tùy chọn:
    python download_pdfs_local.py --limit 100        # chỉ tải 100 paper đầu
    python download_pdfs_local.py --out ./my_pdfs    # thư mục khác
    python download_pdfs_local.py --skip-upload      # tải PDF nhưng không push HF
    python download_pdfs_local.py --only-upload      # chỉ push HF (đã có PDF rồi)
    python download_pdfs_local.py --delay 3          # delay giữa mỗi request (giây)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

PAPERS_URL = (
    "https://raw.githubusercontent.com/triquang26/evidence-ai/main/outputs/papers.jsonl"
)
HF_BUCKET = "hf://buckets/twanghcmut/evidence-ai/pdfs"
UA = "ts-research-harvester/1.0 (academic use; mailto:2003huynhquynhanh@gmail.com)"


# ──────────────────────────────────────────────────────────────────────────────
# 1. Download papers.jsonl
# ──────────────────────────────────────────────────────────────────────────────

def fetch_papers(out_dir: Path) -> list[dict]:
    meta_path = out_dir / "papers.jsonl"
    if meta_path.exists():
        print(f"[skip] papers.jsonl already at {meta_path}")
    else:
        print(f"[fetch] papers.jsonl from GitHub …")
        req = urllib.request.Request(PAPERS_URL, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=60) as r:
            meta_path.write_bytes(r.read())
        print(f"  saved → {meta_path}")

    papers = [json.loads(l) for l in meta_path.read_text().splitlines() if l.strip()]
    print(f"  loaded {len(papers)} papers")
    return papers


# ──────────────────────────────────────────────────────────────────────────────
# 2. Download PDFs
# ──────────────────────────────────────────────────────────────────────────────

def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())


def _pdf_url(p: dict) -> str | None:
    if p.get("arxiv_id"):
        return f"https://arxiv.org/pdf/{p['arxiv_id']}.pdf"
    u = p.get("url") or ""
    if u.endswith(".pdf") or "openaccess" in u or "/pdf/" in u:
        return u
    return None


def _paper_id(p: dict) -> str:
    base = p.get("arxiv_id") or _norm(p.get("title", ""))[:60]
    return base.replace("/", "_")


def _fetch_one(url: str, path: Path, delay: float) -> bool:
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=90) as r:
                data = r.read()
            if data[:4] != b"%PDF":
                raise ValueError("not a PDF")
            path.write_bytes(data)
            return True
        except Exception as e:
            wait = 2 ** attempt
            print(f"    attempt {attempt+1} fail ({e}), retry in {wait}s …")
            time.sleep(wait)
    return False


def download_pdfs(papers: list[dict], pdf_dir: Path, delay: float) -> list[Path]:
    pdf_dir.mkdir(parents=True, exist_ok=True)
    ok = skip = fail = no_url = 0
    done_files: list[Path] = []

    for i, p in enumerate(papers):
        url = _pdf_url(p)
        if not url:
            no_url += 1
            continue

        pid = _paper_id(p)
        path = pdf_dir / f"{pid}.pdf"

        if path.exists() and path.stat().st_size > 1000:
            skip += 1
            done_files.append(path)
            continue

        success = _fetch_one(url, path, delay)
        if success:
            ok += 1
            done_files.append(path)
        else:
            fail += 1

        if (i + 1) % 10 == 0 or (i + 1) == len(papers):
            pct = (i + 1) / len(papers) * 100
            print(
                f"  [{i+1}/{len(papers)} {pct:.0f}%] "
                f"ok={ok}  skip={skip}  fail={fail}  no_url={no_url}"
            )

        time.sleep(delay)

    print(f"\nPDF download done: ok={ok}  skip(cached)={skip}  fail={fail}  no_url={no_url}")
    return done_files


# ──────────────────────────────────────────────────────────────────────────────
# 3. Upload to HF bucket
# ──────────────────────────────────────────────────────────────────────────────

def upload_to_hf(pdf_dir: Path) -> None:
    print(f"\n[upload] Syncing {pdf_dir} → {HF_BUCKET} …")

    # Thử dùng huggingface_hub >= 1.x (có HfApi.upload_folder hoặc hf sync)
    try:
        import subprocess
        result = subprocess.run(
            ["uvx", "--from", "huggingface_hub>=1.16", "hf", "sync",
             str(pdf_dir), HF_BUCKET],
            check=True,
        )
        print("  [done] uvx hf sync completed")
        return
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass  # fallback

    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_folder(
            folder_path=str(pdf_dir),
            repo_id="twanghcmut/evidence-ai",
            repo_type="dataset",
            path_in_repo="pdfs",
        )
        print("  [done] HfApi.upload_folder completed")
        return
    except Exception as e:
        print(f"  [warn] HfApi.upload_folder failed: {e}")

    # Last resort: huggingface-cli upload-large-folder
    try:
        import subprocess
        subprocess.run(
            ["huggingface-cli", "upload", "twanghcmut/evidence-ai",
             str(pdf_dir), "pdfs", "--repo-type", "dataset"],
            check=True,
        )
        print("  [done] huggingface-cli upload completed")
    except Exception as e:
        print(f"  [error] All upload methods failed: {e}")
        print("  Manual: huggingface-cli upload twanghcmut/evidence-ai <pdf_dir> pdfs --repo-type dataset")
        sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Download arxiv PDFs → HF bucket")
    ap.add_argument("--out", default="./local_pdfs", help="local output dir (default: ./local_pdfs)")
    ap.add_argument("--limit", type=int, default=0, help="max papers to process (0 = all)")
    ap.add_argument("--delay", type=float, default=3.0, help="seconds between requests (default: 3)")
    ap.add_argument("--skip-upload", action="store_true", help="tải PDF nhưng không push lên HF")
    ap.add_argument("--only-upload", action="store_true", help="chỉ push HF, không tải PDF")
    args = ap.parse_args()

    out_dir = Path(args.out)
    pdf_dir = out_dir / "pdfs"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  evidence-ai PDF downloader")
    print(f"  output   : {pdf_dir.resolve()}")
    print(f"  hf bucket: {HF_BUCKET}")
    print(f"  delay    : {args.delay}s per request")
    print("=" * 60)

    if not args.only_upload:
        papers = fetch_papers(out_dir)
        if args.limit:
            papers = papers[: args.limit]
            print(f"  [limit] using first {args.limit} papers")

        # Filter papers có arxiv_id
        with_id = [p for p in papers if p.get("arxiv_id")]
        without  = len(papers) - len(with_id)
        print(f"  papers with arxiv_id: {len(with_id)}  without: {without}")
        print()

        download_pdfs(with_id, pdf_dir, delay=args.delay)

    if not args.skip_upload:
        upload_to_hf(pdf_dir)
    else:
        print("\n[skip] --skip-upload set, không push lên HF")

    print("\nDone! Trên GPU box chạy:")
    print(f"  uvx --from 'huggingface_hub>=1.16' hf sync {HF_BUCKET} ./local_pdfs/pdfs")
    print("  rồi set source.kind=local và source.local_dir=./local_pdfs/pdfs trong config")


if __name__ == "__main__":
    main()
