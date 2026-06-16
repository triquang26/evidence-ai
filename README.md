# evidence-ai

Harvest ML papers and extract **grounded** system-config / compute / efficiency evidence from their PDFs.

Two stages, two packages:

- `harvester/` — collect paper metadata → `outputs/papers.jsonl` (+ curated `test/`). See `harvest.py`.
- `extraction/` — PDF → markdown (**MinerU2.5 VLM** on vLLM) → grounded JSON (**LangExtract** → **Qwen3**
  on vLLM) → audit HTML. See `extract_pipeline.py`.

Anti-hallucination is two-layer grounding: MinerU keeps each block's `page_idx`/`bbox`; LangExtract maps
every extracted value to a char-offset in the source text. Optional `guided_json` (vLLM xgrammar) adds a
JSON-validity guarantee on top — it guarantees *shape*, grounding guarantees *provenance*.

## Setup (reproducible — every dep is declared in config, no ad-hoc installs)

Both paths read the **same** dependency set; `pyproject.toml` is the single source of truth.

**Fast path — uv (recommended):**

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e . --torch-backend=cu126   # cu126 build (see CUDA note below)
mineru --help                               # confirm the VLM backend flag, patch configs/pipeline.yaml if it differs
```

> **CUDA / driver note.** This host runs NVIDIA driver 535 (CUDA 12.2). The newest
> `vllm`/`torch` ship cu130 wheels that the driver is too old to run; `cu126` is the
> forward-compat ceiling that works on this H100. `pyproject.toml` caps `vllm<0.21` /
> `torch<2.9` and sets `[tool.uv] torch-backend = "cu126"`, so `uv pip install -e .`
> already does the right thing. On a host with a newer driver, drop those caps.

**Conda path (alternative):**

```bash
conda env create -f environment.yml
conda activate evidence-ai
pip install -e .
```

Target: 1× H100 80GB, CUDA 12.x.

## Run

Config lives in `configs/pipeline.yaml` (models, ports, schema, passes, seed). Override key bits from the CLI:

```bash
# end-to-end on the curated test set (sequential: parse all -> free GPU -> serve Qwen3 -> extract all)
python extract_pipeline.py --run-name test_smoke --input test/papers/papers.jsonl

python extract_pipeline.py --guided          # also emit guided_json-validated JSON
python extract_pipeline.py --no-server        # reuse an externally-served Qwen (scripts/serve_extractor.sh)
```

Sequential serving is automatic: MinerU runs the VLM **in-process** (GPU freed on exit), then `VLLMServer`
starts Qwen3 once for the whole extract phase and tears it down after.

## Firewalled host (arxiv blocked) — PDF handoff via the bucket

This GPU box can reach **HuggingFace + GitHub only**; `arxiv.org` is blocked, so PDFs can't be
downloaded here. The query→PDF code (`fetch_pdfs.py`, reusing `harvester.PDFDownloader`) still works —
just run it where arxiv is reachable and hand the PDFs over through the bucket:

```bash
# (1) on a host that CAN reach arxiv — your laptop / Colab:
python fetch_pdfs.py --input test/papers/papers.jsonl --out test_pdfs
hf sync test_pdfs hf://buckets/twanghcmut/evidence-ai/test/pdfs    # needs huggingface_hub>=1.x

# (2) on the GPU box — pull PDFs down and run the real test set:
hf sync hf://buckets/twanghcmut/evidence-ai/test/pdfs test_pdfs
python extract_pipeline.py --config configs/test_set.yaml
```

`source.kind: local` (and a directory `--input`) makes the pipeline read PDFs from disk instead of
fetching arxiv. HF buckets need `hf sync` (huggingface_hub ≥ 1.x); the pinned env has 0.36.2, so
`sync_to_hf.py` shells out to a modern `hf` via `uvx` automatically.

## Output (synced to HF as-is)

```
outputs/extract/<run-name>/
  run_config.yaml      # resolved config snapshot (reproducibility)
  pdfs/                # downloaded PDFs
  parsed/<id>/.../<id>.md  + <id>_content_list.json   # markdown + page_idx/bbox
  extracted/<id>.jsonl # LangExtract annotated, grounded
  validated/<id>.json  # optional, guided_json
  audit/<id>.html      # interactive grounding viewer
  manifest.jsonl       # per-paper status, paths, model ids, seed
  timing_estimate.txt  # per-stage wall-clock + full-corpus GPU-hour projection
```

```bash
python sync_to_hf.py --src outputs/extract/<run-name> --dst extract/<run-name>
```

## Extending

- New extraction target → register an `ExtractionSchema` in `extraction/schemas.py`, set
  `extractor.schema` in the YAML.
- New parser/extractor backend → implement `BaseParser` / `BaseExtractor`, register in
  `extraction/factory.py`. Nothing else changes.
```