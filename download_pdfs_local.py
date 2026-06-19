#!/usr/bin/env python3
"""
download_pdfs_local.py — chạy trên máy LOCAL (có internet tới arxiv.org)

Flow:
  1. Tải papers.jsonl từ GitHub (danh sách 5692 papers)
  2. Download PDF từ arxiv.org  →  ./pdfs/<arxiv_id>.pdf
  3. Upload lên HF bucket       →  hf://buckets/twanghcmut/evidence-ai/pdfs/

Sau đó trên GPU box:
  uvx --from 'huggingface_hub>=1.16' hf sync \\
      hf://buckets/twanghcmut/evidence-ai/pdfs ./pdfs
  python extract_pipeline.py --config configs/full_run.yaml

─── Cài đặt (local) ───────────────────────────────────────────────────────────
  pip install requests 'huggingface_hub>=1.16'
  huggingface-cli login          # nhập HF token (cần write access vào bucket)

─── Chạy ──────────────────────────────────────────────────────────────────────
  python download_pdfs_local.py                  # tải tất cả (~5692 papers, vài tiếng)
  python download_pdfs_local.py --limit 50       # test nhanh 50 papers đầu
  python download_pdfs_local.py --skip-upload    # chỉ tải PDF, không push HF
  python download_pdfs_local.py --only-upload    # đã có PDF rồi, chỉ push HF
  python download_pdfs_local.py --pdf-dir ./my_pdfs  # thư mục lưu PDF tùy chọn
  python download_pdfs_local.py --delay 2        # delay giữa requests (mặc định 3s)
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


# ── 1. Lấy danh sách papers ──────────────────────────────────────────────────

def fetch_papers(out_dir: Path) -> list[dict]:
    meta_path = out_dir / "papers.jsonl"
    if meta_path.exists():
        print(f"[ok] papers.jsonl có sẵn tại {meta_path}")
    else:
        print("[download] papers.jsonl từ GitHub …")
        req = urllib.request.Request(PAPERS_URL, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=60) as r:
            meta_path.write_bytes(r.read())
        print(f"  lưu → {meta_path}")

    papers = [json.loads(l) for l in meta_path.read_text().splitlines() if l.strip()]
    print(f"  {len(papers)} papers trong danh sách")
    return papers


# ── 2. Download PDFs ─────────────────────────────────────────────────────────

def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())

def _paper_id(p: dict) -> str:
    base = p.get("arxiv_id") or _norm(p.get("title", ""))[:60]
    return base.replace("/", "_")

def _pdf_url(p: dict) -> str | None:
    if p.get("arxiv_id"):
        return f"https://arxiv.org/pdf/{p['arxiv_id']}.pdf"
    u = p.get("url") or ""
    if u.endswith(".pdf") or "openaccess" in u or "/pdf/" in u:
        return u
    return None

def _fetch_one(url: str, path: Path) -> bool:
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=90) as r:
                data = r.read()
            if data[:4] != b"%PDF":
                raise ValueError("response không phải PDF")
            path.write_bytes(data)
            return True
        except Exception as e:
            wait = 2 ** attempt
            if attempt < 3:
                print(f"    lần {attempt+1} thất bại ({e}), thử lại sau {wait}s …")
                time.sleep(wait)
    return False

def download_pdfs(papers: list[dict], pdf_dir: Path, delay: float) -> None:
    pdf_dir.mkdir(parents=True, exist_ok=True)
    ok = skip = fail = no_url = 0

    for i, p in enumerate(papers):
        url = _pdf_url(p)
        if not url:
            no_url += 1
            continue

        pid  = _paper_id(p)
        path = pdf_dir / f"{pid}.pdf"

        if path.exists() and path.stat().st_size > 1000:
            skip += 1
            continue

        if _fetch_one(url, path):
            ok += 1
        else:
            fail += 1

        if (i + 1) % 20 == 0 or (i + 1) == len(papers):
            pct = (i + 1) / len(papers) * 100
            print(
                f"  [{i+1}/{len(papers)} | {pct:.0f}%]  "
                f"ok={ok}  skip={skip}  fail={fail}  no_url={no_url}"
            )

        time.sleep(delay)

    print(f"\nxong download: ok={ok}  đã có={skip}  fail={fail}  no_url={no_url}")
    total = ok + skip
    print(f"tổng PDFs có sẵn: {total}  →  {pdf_dir}/")


# ── 3. Upload lên HF bucket ───────────────────────────────────────────────────

def upload_to_hf(pdf_dir: Path) -> None:
    print(f"\n[upload] {pdf_dir}  →  {HF_BUCKET}")
    print("  dùng: huggingface_hub >= 1.16  (hf sync)")

    import subprocess
    # Thử hf CLI trực tiếp (nếu đã có huggingface_hub >= 1.x trong PATH)
    for cmd in [
        ["hf", "sync", str(pdf_dir), HF_BUCKET],
        ["uvx", "--from", "huggingface_hub>=1.16", "hf", "sync", str(pdf_dir), HF_BUCKET],
    ]:
        try:
            subprocess.run(cmd, check=True)
            print("  [done] upload xong!")
            return
        except FileNotFoundError:
            continue
        except subprocess.CalledProcessError as e:
            print(f"  [warn] {cmd[0]} failed: {e}")
            break

    # Fallback: hướng dẫn tay
    print("\n  ⚠  Không tìm thấy 'hf' hoặc 'uvx'. Chạy tay:")
    print(f"     pip install 'huggingface_hub>=1.16'")
    print(f"     hf sync {pdf_dir} {HF_BUCKET}")
    sys.exit(1)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Download arxiv PDFs → HF bucket")
    ap.add_argument("--pdf-dir",     default="./pdfs",  help="thư mục lưu PDF (mặc định: ./pdfs)")
    ap.add_argument("--limit",       type=int, default=0, help="chỉ xử lý N papers đầu (0=tất cả)")
    ap.add_argument("--delay",       type=float, default=3.0, help="giây chờ giữa requests (mặc định: 3)")
    ap.add_argument("--skip-upload", action="store_true", help="tải PDF nhưng không push HF")
    ap.add_argument("--only-upload", action="store_true", help="chỉ push HF, không tải PDF")
    args = ap.parse_args()

    pdf_dir = Path(args.pdf_dir)
    out_dir = pdf_dir.parent

    print("=" * 60)
    print("  evidence-ai  |  PDF downloader")
    print(f"  pdf dir  : {pdf_dir.resolve()}")
    print(f"  hf bucket: {HF_BUCKET}")
    print(f"  delay    : {args.delay}s / request")
    print("=" * 60 + "\n")

    if not args.only_upload:
        papers = fetch_papers(out_dir)

        # Lọc chỉ papers có arxiv_id (mới có URL tải được)
        with_id  = [p for p in papers if p.get("arxiv_id")]
        no_id    = len(papers) - len(with_id)
        print(f"  có arxiv_id: {len(with_id)}  |  không có: {no_id} (bỏ qua)\n")

        if args.limit:
            with_id = with_id[:args.limit]
            print(f"  [--limit] chỉ xử lý {args.limit} papers đầu\n")

        download_pdfs(with_id, pdf_dir, delay=args.delay)

    if not args.skip_upload:
        upload_to_hf(pdf_dir)
    else:
        print("\n[skip] --skip-upload: không push HF")

    print("\n" + "=" * 60)
    print("XONG! Bước tiếp theo trên GPU box:")
    print()
    print("  # Pull PDFs về")
    print(f"  uvx --from 'huggingface_hub>=1.16' hf sync \\")
    print(f"      {HF_BUCKET} ./pdfs")
    print()
    print("  # Chạy extraction pipeline")
    print("  python extract_pipeline.py --config configs/full_run.yaml")
    print("=" * 60)


if __name__ == "__main__":
    main()
