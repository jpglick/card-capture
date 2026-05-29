# V5.5 CI Lane Commands

## One-time setup

```bash
python3 -m pip install -e ".[dev]"
# If lint-imports is not on PATH, add user-site bin:
export PATH="$(python3 -m site --user-base)/bin:$PATH"
```

## Fast PR lane (default; no GPU, no credentials, no real videos)

```bash
python3 -m pytest tests/ -q
python3 -m pytest tests/architecture/ -q
python3 -m pytest tests/performance/test_perf_harness_smoke.py -q
PYTHONPATH=src:. lint-imports
```

The `addopts` line in `pyproject.toml` already deselects `quarantine` and `benchmark` markers.

## Optional hardware lane (CUDA)

```bash
python3 -m pytest tests/ -q -m cuda
```

## Optional hardware lane (MPS)

```bash
python3 -m pytest tests/ -q -m mps
```

## Provider lane (requires credentials)

```bash
python3 -m pytest tests/ -q -m provider
```
