from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from .schemas import get_schema

_EMPTY = {
    "", "none", "n/a", "not specified", "not mentioned", "none reported", "0",
    "unspecified", "unknown", "not available", "not provided", "not applicable",
    "na", "n/a", "tbd", "?", "-",
}
# Fields that are experiment-wide: if ANY record in the paper has them, propagate to all records.
_CONTEXT_FIELDS = ("compute_hardware", "system_framework", "system_precision", "compute_budget")
_ARXIV_RE = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")
# Drop figure-legend artifacts the model sometimes emits as "subjects" (e.g. "Purple Line").
_LEGEND_RE = re.compile(
    r"^(purple|blue|orange|green|red|black|yellow|gray|grey|cyan|magenta|brown|pink)\b"
    r".*(line|dashed|\(s\)|curve|bar)?",
    re.I,
)
# Additional noise patterns: scatter-plot labels, unnamed series, generic placeholders.
_NOISE_RE = re.compile(
    r"^(unlabeled|unnamed|unknown|scatter|series\s*\d*|group\s*\d*|item\s*\d*)\b",
    re.I,
)


_LATEX = {r"\times": "x", r"\mathcal{O}": "O", r"\log": "log", r"\approx": "~", r"\sim": "~",
          r"\%": "%", r"\,": " ", "$": "", "{": "", "}": ""}


def _clean(value: str) -> str:
    """Strip LaTeX wrappers / collapse whitespace so CSV cells are readable."""
    v = (value or "").strip()
    for a, b in _LATEX.items():
        v = v.replace(a, b)
    v = re.sub(r"\\[a-zA-Z]+", "", v)        # drop remaining \commands
    return re.sub(r"\s+", " ", v).strip()


def _is_noise_subject(subject: str) -> bool:
    s = (subject or "").strip().lower()
    if s in _EMPTY:
        return True
    if _NOISE_RE.match(s):
        return True
    return bool(_LEGEND_RE.match(s)) and any(w in s for w in ("line", "dashed", "(s)", "curve", "bar"))


def _paper_context(extractions: list[dict]) -> dict[str, str]:
    """Collect paper-level context fields (hardware/framework/precision) from all extractions.

    Only propagates a field when ALL records that mention it agree on the same value.
    If two records disagree (e.g. "4x A100" vs "8x H100"), the field is NOT inherited —
    the disagreement signals multiple experimental setups, so we cannot safely fill the rest.
    """
    # Map field → {lower-cased value → original value} to detect conflicts
    seen: dict[str, dict[str, str]] = {f: {} for f in _CONTEXT_FIELDS}
    for ext in extractions:
        attrs = ext.get("attributes") or {}
        for f in _CONTEXT_FIELDS:
            v = (attrs.get(f) or "").strip()
            if v and v.lower() not in _EMPTY:
                seen[f].setdefault(v.lower(), v)
    # Only return fields with exactly one distinct value across the paper
    return {f: next(iter(vals.values())) for f, vals in seen.items() if len(vals) == 1}
# Provenance columns first so a downstream engine can re-fetch the paper PDF.
_PROVENANCE = ["paper_id", "arxiv_id", "arxiv_url", "title"]


def _is_real(extraction: dict) -> bool:
    """Keep only well-grounded, non-placeholder spans (anti-hallucination filter).

    Rejects spans whose aligned character range is far shorter than the quoted text —
    that signals a poor/failed alignment (often few-shot leakage), not a real quote.
    """
    text = (extraction.get("extraction_text") or "").strip()
    if text.lower() in _EMPTY:
        return False
    ci = extraction.get("char_interval")
    if ci is None or ci.get("start_pos") is None or ci.get("end_pos") is None:
        return False
    span = ci["end_pos"] - ci["start_pos"]
    return span >= 0.5 * len(text)


def _provenance(rec: dict) -> dict:
    """paper_id / arxiv_id / arxiv_url / title so the PDF can be retrieved later."""
    pid = rec.get("id")
    arxiv_id = rec.get("arxiv_id") or (pid if pid and _ARXIV_RE.match(str(pid)) else "")
    url = rec.get("url") or (f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else "")
    return {"paper_id": pid, "arxiv_id": arxiv_id, "arxiv_url": url, "title": rec.get("title")}


def _attr_val(col: str, attrs: dict, ctx: dict) -> str:
    """Return the best non-placeholder value for `col` from attrs, falling back to ctx."""
    v = _clean(attrs.get(col) or "")
    if v.lower() in _EMPTY:
        v = ""
    if not v:
        v = _clean(ctx.get(col) or "")
        if v.lower() in _EMPTY:
            v = ""
    return v


def _schema_columns(schema) -> list[str]:
    if schema.columns:
        return list(schema.columns)
    cols: list[str] = []
    for cls in schema.json_schema.get("properties", {}).values():
        for attr in cls.get("properties", {}):
            if attr not in cols:
                cols.append(attr)
    return cols


class CsvExporter:
    """Flatten LangExtract annotated jsonl into tabular CSVs.

    `evidence_eval` (row_per_record) -> one row per subject-anchored evidence record,
    each carrying paper provenance + grounding. Attribute-bag schemas
    (compute_efficiency) -> one aggregated row per paper.
    """

    def __init__(self, schema_name: str):
        self.schema = get_schema(schema_name)
        self.columns = _schema_columns(self.schema)

    def _load(self, jsonl_path: Path) -> list[dict]:
        lines = jsonl_path.read_text(encoding="utf-8").splitlines()
        return json.loads(lines[0]).get("extractions", []) if lines else []

    def export(self, records: list[dict], run_dir: Path) -> tuple[Path, Path]:
        rows_main, rows_long = [], []
        for rec in records:
            jsonl = rec.get("extracted")
            if not jsonl or not Path(jsonl).exists():
                continue
            prov = _provenance(rec)
            extractions = self._load(Path(jsonl))
            if self.schema.row_per_record:
                paper_rows = self._record_rows(prov, extractions)
                rows_main.extend(paper_rows)
                self._write_paper_card(run_dir / "papers", prov, paper_rows)
            else:
                rows_main.append(self._paper_row(prov, extractions))
            rows_long.extend(self._long_rows(prov, extractions))

        main_cols = _PROVENANCE + self.columns + (
            ["extraction_text", "char_start", "char_end", "alignment_status"]
            if self.schema.row_per_record else []
        )
        evidence = self._write(run_dir / "evidence.csv", main_cols, rows_main)
        long = self._write(
            run_dir / "extractions_long.csv",
            _PROVENANCE + ["extraction_class", "extraction_text", "char_start", "char_end",
                           "alignment_status", "attributes"],
            rows_long,
        )
        return evidence, long

    def _write_paper_card(self, papers_dir: Path, prov: dict, rows: list[dict]) -> Path:
        """One self-contained JSON per paper: provenance + its evidence records (nested)."""
        papers_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for r in rows:
            rec = {c: r[c] for c in self.columns if r.get(c)}
            rec["source"] = {"text": r.get("extraction_text"),
                             "char_start": r.get("char_start"), "char_end": r.get("char_end")}
            records.append(rec)
        card = {**prov, "schema": self.schema.name, "n_records": len(records), "records": records}
        target = papers_dir / f"{prov['paper_id']}.json"
        target.write_text(json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")
        return target

    def _record_rows(self, prov: dict, extractions: list[dict]) -> list[dict]:
        ctx = _paper_context(extractions)
        rows = []
        for ext in extractions:
            if not _is_real(ext):
                continue
            attrs = ext.get("attributes") or {}
            if _is_noise_subject(attrs.get("subject", "")):
                continue  # no subject / figure-legend artifact -> not comparable
            ci = ext.get("char_interval") or {}
            row = dict(prov)
            # For sparse context fields, fall back to paper-level context so a results-table
            # record inherits hardware/framework even when those were in the setup section.
            # Placeholder values ("unspecified", "unknown", …) are suppressed at both layers.
            row.update({c: _attr_val(c, attrs, ctx) for c in self.columns})
            row.update({
                "extraction_text": ext.get("extraction_text"),
                "char_start": ci.get("start_pos"), "char_end": ci.get("end_pos"),
                "alignment_status": ext.get("alignment_status"),
            })
            rows.append(row)
        return rows

    def _paper_row(self, prov: dict, extractions: list[dict]) -> dict:
        row = dict(prov)
        for ext in extractions:
            if not _is_real(ext):
                continue
            for k, v in (ext.get("attributes") or {}).items():
                if k not in self.columns or str(v).strip().lower() in _EMPTY:
                    continue
                if row.get(k) and str(v) not in row[k]:
                    row[k] = f"{row[k]}; {v}"
                elif not row.get(k):
                    row[k] = str(v)
        return row

    @staticmethod
    def _long_rows(prov: dict, extractions: list[dict]) -> list[dict]:
        out = []
        for ext in extractions:
            if not _is_real(ext):
                continue
            ci = ext.get("char_interval") or {}
            row = dict(prov)
            row.update({
                "extraction_class": ext.get("extraction_class"),
                "extraction_text": ext.get("extraction_text"),
                "char_start": ci.get("start_pos"), "char_end": ci.get("end_pos"),
                "alignment_status": ext.get("alignment_status"),
                "attributes": json.dumps(ext.get("attributes") or {}, ensure_ascii=False),
            })
            out.append(row)
        return out

    @staticmethod
    def _write(path: Path, fieldnames: list[str], rows: list[dict]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        return path


def build_corpus_comparison(out_root: Path) -> Path | None:
    """Concatenate every run's evidence.csv into out_root/comparison.csv for cross-paper comparison."""
    out_root = Path(out_root)
    evidence_files = sorted(out_root.glob("*/evidence.csv"))
    if not evidence_files:
        return None
    rows, fields = [], []
    for ef in evidence_files:
        with ef.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for col in reader.fieldnames or []:  # union of all runs' columns
                if col not in fields:
                    fields.append(col)
            for r in reader:
                r["run"] = ef.parent.name
                rows.append(r)
    target = out_root / "comparison.csv"
    with target.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["run", *fields], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return target
