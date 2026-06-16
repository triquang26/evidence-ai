"""
Sync project outputs to HuggingFace bucket: hf://buckets/twanghcmut/evidence-ai

Usage:
    python sync_to_hf.py                  # sync default outputs/ folder
    python sync_to_hf.py --src results/   # sync specific folder
    python sync_to_hf.py --src results/ --dst experiments/run1
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


BUCKET = "hf://buckets/twanghcmut/evidence-ai"
# `hf sync` (bucket support) needs huggingface_hub >= 1.x. The pipeline venv pins
# 0.36.2 (vllm/mineru), so fall back to running a modern hf in isolation via uvx.
HF_PIN = "huggingface_hub>=1.16"


class HFBucketSync:
    def __init__(self, src: str, dst: str | None = None):
        self.src = Path(src)
        self.dst_path = f"{BUCKET}/{dst}" if dst else f"{BUCKET}/{self.src.name}"

    def validate(self):
        if not self.src.exists():
            raise FileNotFoundError(f"Source path does not exist: {self.src}")
        if not self.src.is_dir():
            raise ValueError(f"Source must be a directory: {self.src}")

    def _sync_cmd(self) -> list[str]:
        if self._hf_has_sync():
            return ["hf", "sync", str(self.src), self.dst_path]
        if shutil.which("uvx"):
            return ["uvx", "--from", HF_PIN, "hf", "sync", str(self.src), self.dst_path]
        raise RuntimeError("Need `hf` with bucket sync (huggingface_hub>=1.x) or `uvx` on PATH.")

    @staticmethod
    def _hf_has_sync() -> bool:
        if not shutil.which("hf"):
            return False
        out = subprocess.run(["hf", "--help"], capture_output=True, text=True)
        return "sync" in out.stdout

    def sync(self):
        self.validate()
        cmd = self._sync_cmd()
        print(f"Syncing {self.src} → {self.dst_path}")
        return subprocess.run(cmd, check=True).returncode


def parse_args():
    parser = argparse.ArgumentParser(description="Sync local outputs to HuggingFace bucket")
    parser.add_argument("--src", default="outputs", help="Local folder to upload (default: outputs/)")
    parser.add_argument("--dst", default=None, help="Destination subfolder in bucket (default: same as src name)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    syncer = HFBucketSync(src=args.src, dst=args.dst)
    sys.exit(syncer.sync())
