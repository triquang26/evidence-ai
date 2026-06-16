from __future__ import annotations

import json
from pathlib import Path

from openai import OpenAI

from ..config import ExtractorConfig
from ..schemas import get_schema


class GuidedJsonValidator:
    """Optional structure layer: vLLM guided_json (xgrammar) for 100% valid JSON.

    Complements — does not replace — LangExtract: guided_json guarantees the
    SHAPE of the output, grounding guarantees the PROVENANCE of the values.
    """

    def __init__(self, config: ExtractorConfig):
        self.config = config
        self.schema = get_schema(config.schema)
        self._client = OpenAI(base_url=config.base_url, api_key=config.api_key)

    def validate(self, text: str, paper_id: str, out_dir: Path) -> Path | None:
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            resp = self._client.chat.completions.create(
                model=self.config.model_id,
                messages=[{"role": "user", "content": f"{self.schema.prompt}\n\n{text}"}],
                extra_body={"guided_json": self.schema.json_schema},
            )
            payload = json.loads(resp.choices[0].message.content)
        except Exception:  # noqa: BLE001 - optional layer, never fatal
            return None
        target = out_dir / f"{paper_id}.json"
        target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return target
