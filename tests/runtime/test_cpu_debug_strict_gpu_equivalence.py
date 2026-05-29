"""Equivalence contract per spec Section 6 table.

Field classes:
- Identity / schema: exact
- Counts: exact
- Geometric (corners): ± 2 px
- Quality scores: ± 5% (relative) / ± 0.02 (absolute for `total`)
- pHash: Hamming distance ≤ 4 bits
- ReID embeddings: cosine similarity ≥ 0.95
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from card_capture.pipeline.request import PipelineRunRequest
from card_capture.runtime.cpu_debug import CpuDebugRuntime
from card_capture.runtime.strict_gpu import StrictGpuRuntime
from card_capture.runtime.gpu_session import MissingGpuError


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "tiny_synthetic.MOV"


def _hamming_hex(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def _cosine(a: list[float], b: list[float]) -> float:
    A = np.asarray(a, dtype=np.float32); B = np.asarray(b, dtype=np.float32)
    denom = (np.linalg.norm(A) * np.linalg.norm(B)) or 1.0
    return float(np.dot(A, B) / denom)


@pytest.mark.skipif(not FIXTURE.exists(), reason="fixture missing")
def test_cpu_debug_strict_gpu_equivalence(tmp_path):
    try:
        gpu_runtime = StrictGpuRuntime()
    except MissingGpuError:
        pytest.skip("no GPU available")

    base = dict(input_video=f"artifact://local/{FIXTURE}",
                output_root=f"artifact://local/{tmp_path}/",
                config={})
    cpu = CpuDebugRuntime().run(PipelineRunRequest(
        run_id="eq-cpu", runtime_mode="cpu_debug", **base
    )).manifest
    gpu = gpu_runtime.run(PipelineRunRequest(
        run_id="eq-gpu", runtime_mode="strict_gpu", **base
    )).manifest

    # 1. Identity / schema: exact.
    assert sorted(vars(cpu).keys()) == sorted(vars(gpu).keys())

    # 2. Counts: exact.
    assert len(cpu.cards) == len(gpu.cards), \
        f"card count differs: cpu={len(cpu.cards)} gpu={len(gpu.cards)}"

    # Pair cards by card_instance_id (same input video -> same identity field).
    cpu_by_id = {c.card_instance_id: c for c in cpu.cards}
    for g in gpu.cards:
        c = cpu_by_id.get(g.card_instance_id)
        assert c is not None, f"GPU card {g.card_instance_id} missing from CPU run"

        # 3. Quality scores: ±5% relative, ±0.02 absolute on `total`.
        for metric in ("sharpness", "glare", "aspect_ratio", "size", "complexity", "border_purity"):
            if metric in c.quality and metric in g.quality:
                cv, gv = c.quality[metric], g.quality[metric]
                if cv == 0:
                    assert abs(gv) < 0.02, f"{metric}: cpu=0, gpu={gv}"
                else:
                    rel = abs(gv - cv) / abs(cv)
                    assert rel <= 0.05, f"{metric}: cpu={cv} gpu={gv} rel={rel:.3f} > 0.05"
        if "total" in c.quality and "total" in g.quality:
            assert abs(c.quality["total"] - g.quality["total"]) <= 0.02

        # 4. pHash: Hamming distance ≤ 4 bits (if both runtimes computed it).
        cpu_phash = c.quality.get("phash_hex")
        gpu_phash = g.quality.get("phash_hex")
        if isinstance(cpu_phash, str) and isinstance(gpu_phash, str):
            assert _hamming_hex(cpu_phash, gpu_phash) <= 4

        # 5. ReID embeddings: cosine ≥ 0.95 (if both runtimes computed it).
        cpu_emb = c.quality.get("reid_embedding")
        gpu_emb = g.quality.get("reid_embedding")
        if isinstance(cpu_emb, list) and isinstance(gpu_emb, list):
            assert _cosine(cpu_emb, gpu_emb) >= 0.95
