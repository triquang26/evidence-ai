from __future__ import annotations

import subprocess
import time
import urllib.error
import urllib.request

from .config import ExtractorConfig, ServerConfig


class VLLMServer:
    """Context manager owning a `vllm serve` process for the extractor model.

    Realizes sequential serving: the parse stage runs first (MinerU frees the
    GPU on exit), then `with VLLMServer(...)` brings Qwen3 up for extraction and
    tears it down afterwards. If ServerConfig.enabled is False (server already
    running elsewhere), this is a no-op wrapper.
    """

    def __init__(self, server: ServerConfig, extractor: ExtractorConfig):
        self.server = server
        self.extractor = extractor
        self._proc: subprocess.Popen | None = None

    @property
    def health_url(self) -> str:
        return f"http://localhost:{self.server.port}/health"

    def __enter__(self) -> "VLLMServer":
        if not self.server.enabled:
            return self
        self._proc = subprocess.Popen(self._command())
        self._wait_until_healthy()
        return self

    def __exit__(self, *exc) -> None:
        if self._proc is None:
            return
        self._proc.terminate()
        try:
            self._proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        self._proc = None

    def _command(self) -> list[str]:
        cmd = [
            "vllm", "serve", self.extractor.model_id,
            "--port", str(self.server.port),
            "--gpu-memory-utilization", str(self.server.gpu_memory_utilization),
            "--max-model-len", str(self.server.max_model_len),
        ]
        cmd += list(self.server.extra_args)
        return cmd

    def _wait_until_healthy(self) -> None:
        deadline = time.monotonic() + self.server.startup_timeout_s
        while time.monotonic() < deadline:
            if self._proc and self._proc.poll() is not None:
                raise RuntimeError(f"vllm serve exited early (code {self._proc.returncode})")
            try:
                with urllib.request.urlopen(self.health_url, timeout=5) as resp:
                    if resp.status == 200:
                        return
            except (urllib.error.URLError, ConnectionError, OSError):
                pass
            time.sleep(5)
        raise TimeoutError(f"vLLM server not healthy within {self.server.startup_timeout_s}s")
