from __future__ import annotations

import textwrap
from dataclasses import dataclass

import langextract as lx


@dataclass(frozen=True)
class ExtractionSchema:
    """Bundle of everything an extractor needs for one extraction target.

    Keeps prompt + few-shot grounding examples (for LangExtract) and a JSON
    Schema (for the optional vLLM guided_json layer) together so new targets
    plug in by registering one of these.
    """

    name: str
    prompt: str
    examples: list[lx.data.ExampleData]
    json_schema: dict
    columns: tuple = ()  # ordered CSV columns; falls back to json_schema keys when empty
    row_per_record: bool = False  # True -> one CSV row per extraction (subject-anchored records)


_COMPUTE_PROMPT = textwrap.dedent("""\
    Extract system configuration, computational resources, and efficiency
    figures reported in this ML paper. Only extract what literally appears in
    the text; DO NOT infer or estimate. Use the exact source span for
    extraction_text and omit any attribute that is not stated.""")

_COMPUTE_EXAMPLES = [
    lx.data.ExampleData(
        text=(
            "We train on 64 NVIDIA A100 80GB GPUs for 120 hours using PyTorch 2.1 "
            "with bf16 mixed precision. The model reaches 3.2k tokens/sec and "
            "consumes 410 TFLOPs at batch size 256."
        ),
        extractions=[
            lx.data.Extraction(
                extraction_class="hardware",
                extraction_text="64 NVIDIA A100 80GB GPUs for 120 hours",
                attributes={"gpu_type": "NVIDIA A100 80GB", "gpu_count": "64", "gpu_hours": "7680"},
            ),
            lx.data.Extraction(
                extraction_class="system",
                extraction_text="PyTorch 2.1 with bf16 mixed precision",
                attributes={"framework": "PyTorch", "version": "2.1", "precision": "bf16"},
            ),
            lx.data.Extraction(
                extraction_class="efficiency",
                extraction_text="3.2k tokens/sec and consumes 410 TFLOPs at batch size 256",
                attributes={"throughput": "3.2k tokens/sec", "flops": "410 TFLOPs", "batch_size": "256"},
            ),
        ],
    )
]

_COMPUTE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "hardware": {
            "type": "object",
            "properties": {
                "gpu_type": {"type": "string"},
                "gpu_count": {"type": "string"},
                "gpu_hours": {"type": "string"},
            },
        },
        "system": {
            "type": "object",
            "properties": {
                "framework": {"type": "string"},
                "version": {"type": "string"},
                "precision": {"type": "string"},
            },
        },
        "efficiency": {
            "type": "object",
            "properties": {
                "throughput": {"type": "string"},
                "flops": {"type": "string"},
                "batch_size": {"type": "string"},
            },
        },
    },
}

# --------------------------------------------------------------------------------------
# evidence_eval: subject-anchored evaluation records (Stanford CS197 DV/IV/Task/Threats).
# Every figure is tied to a SUBJECT (the system measured) so rows are comparable.
# --------------------------------------------------------------------------------------

_EVIDENCE_PROMPT = textwrap.dedent("""\
    Extract EVALUATION EVIDENCE from this ML paper so systems can be compared.
    Emit a record only for SUBSTANTIVE measurements ANCHORED to the SUBJECT (the model,
    method, or system being measured): the hardware/compute used, training or inference
    cost, the headline efficiency metrics (throughput, latency, FLOPs, memory, GPU-hours),
    the system configuration (framework/precision), and the key result metric for each
    evaluated system — the proposed method AND its baselines / ablations.

    SKIP (do NOT emit records for): figure-legend entries; every individual cell of large
    tables; qualitative prose with no number; restatements/duplicates of a value already
    captured; incidental numbers (equation indices, citation years, section numbers). Aim
    for the few records that actually let you compare systems — not an exhaustive dump.

    Each record follows the evaluation architecture:
      - subject: the system/method/model the figure is about (REQUIRED; never report a value without it)
      - role: proposed | baseline | ablation
      - dv_name / dv_value: the measured outcome = the efficiency OF the subject (throughput,
        latency, GPU-hours, FLOPs, memory, accuracy, ...) and its value with unit
      - iv_name / iv_value: what was varied to cause the change (batch size, model size, hardware)
      - task_dataset: the task / benchmark / dataset it was measured on
      - system_framework / system_precision: system configuration (framework/library; precision/parallelism)
      - compute_hardware / compute_budget: computational resources (GPU type & count; GPU-hours / time / cost)
      - claim: a comparative claim if stated ("X is better than Y at task Z on metric M")
      - threats: stated assumptions or limitations ("single seed", "only 1 epoch")

    Only extract what literally appears in the text; DO NOT infer or estimate. Use the exact
    source span for extraction_text. Omit any attribute that is not stated. If a results table
    compares systems, emit one record per system. Never produce a record without a subject.
    A subject must be a NAMED system, model, method, dataset, or algorithm — do NOT treat figure
    legend artifacts (chart line colors/styles such as "Purple Line", "Red (dashed)", "Blue (s)")
    as subjects.""")

_EVIDENCE_EXAMPLES = [
    lx.data.ExampleData(
        text=(
            "We train Med-U1 on 4 NVIDIA A100 80G GPUs for 1 epoch using the verl framework "
            "with bf16. It reaches 3.2k tokens/sec at batch size 4, outperforming the "
            "LLaMA-Factory baseline on the MedQA benchmark."
        ),
        extractions=[
            lx.data.Extraction(
                extraction_class="evidence_record",
                extraction_text="train Med-U1 on 4 NVIDIA A100 80G GPUs for 1 epoch using the verl framework with bf16. It reaches 3.2k tokens/sec at batch size 4",
                attributes={
                    "subject": "Med-U1", "role": "proposed",
                    "dv_name": "training throughput", "dv_value": "3.2k tokens/sec",
                    "iv_name": "batch size", "iv_value": "4", "task_dataset": "MedQA",
                    "system_framework": "verl", "system_precision": "bf16",
                    "compute_hardware": "4x NVIDIA A100 80G", "compute_budget": "1 epoch",
                    "claim": "Med-U1 better than LLaMA-Factory on MedQA", "threats": "single epoch",
                },
            ),
            lx.data.Extraction(
                extraction_class="evidence_record",
                extraction_text="LLaMA-Factory baseline on the MedQA benchmark",
                attributes={"subject": "LLaMA-Factory", "role": "baseline", "task_dataset": "MedQA"},
            ),
        ],
    ),
    lx.data.ExampleData(
        text=(
            "Table 2 reports latency on ImageNet: our TriPlane tokenizer runs at 3 Hz on a "
            "single A100 80GB, versus 1.2 Hz for the BEVFormer baseline."
        ),
        extractions=[
            lx.data.Extraction(
                extraction_class="evidence_record",
                extraction_text="our TriPlane tokenizer runs at 3 Hz on a single A100 80GB",
                attributes={
                    "subject": "TriPlane tokenizer", "role": "proposed",
                    "dv_name": "inference latency", "dv_value": "3 Hz", "task_dataset": "ImageNet",
                    "compute_hardware": "1x A100 80GB",
                    "claim": "TriPlane faster than BEVFormer on ImageNet",
                },
            ),
            lx.data.Extraction(
                extraction_class="evidence_record",
                extraction_text="1.2 Hz for the BEVFormer baseline",
                attributes={
                    "subject": "BEVFormer", "role": "baseline",
                    "dv_name": "inference latency", "dv_value": "1.2 Hz", "task_dataset": "ImageNet",
                },
            ),
        ],
    ),
]

_EVIDENCE_COLUMNS = (
    "subject", "role", "task_dataset",
    "dv_name", "dv_value", "iv_name", "iv_value",
    "system_framework", "system_precision", "compute_hardware", "compute_budget",
    "claim", "threats",
)

_EVIDENCE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "records": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {c: {"type": "string"} for c in _EVIDENCE_COLUMNS},
                "required": ["subject"],
            },
        }
    },
    "required": ["records"],
}

_REGISTRY: dict[str, ExtractionSchema] = {
    "compute_efficiency": ExtractionSchema(
        name="compute_efficiency",
        prompt=_COMPUTE_PROMPT,
        examples=_COMPUTE_EXAMPLES,
        json_schema=_COMPUTE_JSON_SCHEMA,
    ),
    "evidence_eval": ExtractionSchema(
        name="evidence_eval",
        prompt=_EVIDENCE_PROMPT,
        examples=_EVIDENCE_EXAMPLES,
        json_schema=_EVIDENCE_JSON_SCHEMA,
        columns=_EVIDENCE_COLUMNS,
        row_per_record=True,
    ),
}


def get_schema(name: str) -> ExtractionSchema:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown schema '{name}'. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def register_schema(schema: ExtractionSchema) -> None:
    _REGISTRY[schema.name] = schema
