from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class StageTimer:
    """Accumulates wall-clock per named stage; usable as a context manager.

        timer = StageTimer()
        with timer("parse"):
            ...
    """

    elapsed: dict[str, float] = field(default_factory=dict)
    _stage: str | None = None
    _start: float = 0.0

    def __call__(self, stage: str) -> "StageTimer":
        self._stage = stage
        return self

    def __enter__(self) -> "StageTimer":
        self._start = time.monotonic()
        return self

    def __exit__(self, *exc) -> None:
        assert self._stage is not None
        self.elapsed[self._stage] = self.elapsed.get(self._stage, 0.0) + (time.monotonic() - self._start)
        self._stage = None


def write_timing_report(
    path: Path,
    *,
    run_name: str,
    seed: int,
    attempted: int,
    parsed: int,
    extracted: int,
    elapsed: dict[str, float],
    corpus_size: int,
) -> Path:
    """Write the small human-readable timing/estimate .txt with a full-corpus projection."""
    parse_rate = elapsed.get("parse", 0.0) / parsed if parsed else 0.0
    extract_rate = elapsed.get("extract", 0.0) / extracted if extracted else 0.0
    total = sum(elapsed.values())
    per_paper = total / attempted if attempted else 0.0

    def line(name: str, key: str, rate: float | None = None, unit: str = "paper") -> str:
        secs = elapsed.get(key, 0.0)
        suffix = f"   ({rate:.1f} s/{unit})" if rate else ""
        return f"{name:<14}: {secs:8.1f} s{suffix}"

    proj_parse_h = parse_rate * corpus_size / 3600
    proj_extract_h = extract_rate * corpus_size / 3600
    proj_total_h = per_paper * corpus_size / 3600

    text = "\n".join(
        [
            f"evidence-ai extraction run: {run_name}        seed={seed}",
            f"papers attempted: {attempted}   |   parsed: {parsed}   |   extracted: {extracted}",
            "-" * 62,
            line("download", "download", elapsed.get("download", 0.0) / attempted if attempted else None),
            line("parse  (VLM)", "parse", parse_rate),
            line("extract (LX)", "extract", extract_rate),
            line("audit", "audit"),
            "-" * 62,
            f"{'total':<14}: {total:8.1f} s   ({per_paper:.1f} s/paper)",
            "-" * 62,
            f"PROJECTION -> full corpus ({corpus_size} papers): "
            f"parse ~ {proj_parse_h:.1f} h, extract ~ {proj_extract_h:.1f} h, total ~ {proj_total_h:.1f} h",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
