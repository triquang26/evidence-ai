"""
Sync project outputs to HuggingFace bucket: hf://buckets/twanghcmut/evidence-ai

Usage:
    python sync_to_hf.py                  # sync default outputs/ folder
    python sync_to_hf.py --src results/   # sync specific folder
    python sync_to_hf.py --src results/ --dst experiments/run1
"""

import argparse
import subprocess
import sys
from pathlib import Path


BUCKET = "hf://buckets/twanghcmut/evidence-ai"


class HFBucketSync:
    def __init__(self, src: str, dst: str | None = None):
        self.src = Path(src)
        self.dst_path = f"{BUCKET}/{dst}" if dst else f"{BUCKET}/{self.src.name}"

    def validate(self):
        if not self.src.exists():
            raise FileNotFoundError(f"Source path does not exist: {self.src}")
        if not self.src.is_dir():
            raise ValueError(f"Source must be a directory: {self.src}")

    def sync(self):
        self.validate()
        cmd = ["hf", "sync", str(self.src), self.dst_path]
        print(f"Syncing {self.src} → {self.dst_path}")
        result = subprocess.run(cmd, check=True)
        return result.returncode


def parse_args():
    parser = argparse.ArgumentParser(description="Sync local outputs to HuggingFace bucket")
    parser.add_argument("--src", default="outputs", help="Local folder to upload (default: outputs/)")
    parser.add_argument("--dst", default=None, help="Destination subfolder in bucket (default: same as src name)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    syncer = HFBucketSync(src=args.src, dst=args.dst)
    sys.exit(syncer.sync())
