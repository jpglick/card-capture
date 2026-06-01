# Use TestClient as Context Manager Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update all FastAPI integration tests to use `TestClient` as a context manager to ensure lifespan events are triggered.

**Architecture:** Convert direct `TestClient` instantiations to `with` blocks and update `pytest` fixtures to use `yield` with context managers.

**Tech Stack:** Python, FastAPI, pytest

---

### Task 1: Update `tests/app/test_integration.py`

**Files:**
- Modify: `tests/app/test_integration.py`

- [ ] **Step 1: Convert `client` fixture to a yield fixture**

```python
@pytest.fixture()
def client(tmp_db: Path) -> TestClient:
    """TestClient backed by a fresh DB."""
    from app.main import create_app
    app = create_app(db_path=tmp_db)
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c
```

- [ ] **Step 2: Run tests to verify**

Run: `pytest tests/app/test_integration.py`
Expected: PASS

### Task 2: Update `tests/app/test_config_presets.py`

**Files:**
- Modify: `tests/app/test_config_presets.py`

- [ ] **Step 1: Convert `client` fixture to a yield fixture**

```python
@pytest.fixture
def client(tmp_path):
    db = tmp_path / "cards.sqlite"
    app = create_app(db_path=db)
    with TestClient(app) as c:
        yield c
```

- [ ] **Step 2: Run tests to verify**

Run: `pytest tests/app/test_config_presets.py`
Expected: PASS

### Task 3: Update `tests/app/test_training_endpoints.py`

**Files:**
- Modify: `tests/app/test_training_endpoints.py`

- [ ] **Step 1: Convert `client` fixture to a yield fixture**

```python
@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    """TestClient with a fresh migrated database."""
    db = tmp_path / "test.sqlite"
    sqlite3.connect(db).close()
    apply_migrations(db)
    with TestClient(create_app(db_path=db)) as c:
        yield c
```

- [ ] **Step 2: Run tests to verify**

Run: `pytest tests/app/test_training_endpoints.py`
Expected: PASS

### Task 4: Update `tests/app/test_api_contract.py`

**Files:**
- Modify: `tests/app/test_api_contract.py`

- [ ] **Step 1: Update `test_all_required_routes_registered`**

```python
def test_all_required_routes_registered():
    """Every Contract-2 route must appear in the FastAPI route table."""
    with TestClient(create_app()) as client:
        registered = _registered_routes(client.app)
        for method, path, _ in ROUTES_REQUIRED:
            assert (method, path) in registered, f"missing route: {method} {path}"
```

- [ ] **Step 2: Update `test_openapi_includes_v1_routes`**

```python
def test_openapi_includes_v1_routes():
    """The OpenAPI schema must document all Contract-2 routes."""
    with TestClient(create_app()) as client:
        schema = client.get("/openapi.json").json()
        paths = schema["paths"]
        for method, path, _ in ROUTES_REQUIRED:
            assert path in paths, f"missing from OpenAPI: {path}"
            assert method.lower() in paths[path], (
                f"method {method} missing from OpenAPI path {path}"
            )
```

- [ ] **Step 3: Update `test_implemented_route_shapes`**

```python
def test_implemented_route_shapes():
    """Verify that implemented routes return the correct Contract-2 shapes."""
    with TestClient(create_app()) as client:
        # POST bodies for routes that need them
        post_bodies: dict[str, dict] = {
            "/api/v1/videos": {"filename": "test.mov"},
        }
        # ... rest of the function ...
```

- [ ] **Step 4: Update `test_config_presets_returns_builtin_presets`**

```python
def test_config_presets_returns_builtin_presets():
    """GET /api/v1/config/presets returns the three built-in preset names."""
    with TestClient(create_app()) as client:
        response = client.get("/api/v1/config/presets")
        assert response.status_code == 200
        names = {p["preset_name"] for p in response.json()}
        assert {"fast", "balanced", "quality"} == names
```

- [ ] **Step 5: Update `test_stub_routes_return_501`**

```python
def test_stub_routes_return_501():
    """Unimplemented routes must return 501 Not Implemented."""
    with TestClient(create_app(), raise_server_exceptions=False) as client:
        # ... rest of the function ...
```

- [ ] **Step 6: Run tests to verify**

Run: `pytest tests/app/test_api_contract.py`
Expected: PASS

### Task 5: Update `tests/app/test_batch_api.py`

**Files:**
- Modify: `tests/app/test_batch_api.py`

- [ ] **Step 1: Convert `_make_client` to a context manager helper**

```python
import contextlib

@contextlib.contextmanager
def _client_ctx():
    from app.main import create_app
    from pathlib import Path
    import tempfile, sqlite3
    tmp = tempfile.mkdtemp()
    db = Path(tmp) / "cards.sqlite"
    sqlite3.connect(db).close()
    app = create_app(db_path=db)
    with TestClient(app) as client:
        yield client
```

- [ ] **Step 2: Update test functions to use the context manager**

Update `test_batch_requires_video_ids`, `test_batch_returns_batch_id`, and `test_batch_status_404_unknown`.

- [ ] **Step 3: Run tests to verify**

Run: `pytest tests/app/test_batch_api.py`
Expected: PASS

### Task 6: Update `tests/app/test_vastai_worker.py`

**Files:**
- Modify: `tests/app/test_vastai_worker.py`

- [ ] **Step 1: Update all test functions to use `with TestClient(_app()) as client:`**

Update `test_health_returns_ok`, `test_submit_job_enqueues`, `test_status_pending_after_submit`, `test_status_404_for_unknown_job`, `test_results_409_when_not_complete`, and `test_confirm_deletes_job`.

- [ ] **Step 2: Run tests to verify**

Run: `pytest tests/app/test_vastai_worker.py`
Expected: PASS

### Task 7: Update `tests/app/test_sse_events.py`

**Files:**
- Modify: `tests/app/test_sse_events.py`

- [ ] **Step 1: Convert `_make_client` to return a context manager**

```python
import contextlib

@contextlib.contextmanager
def _client_ctx(bus: EventBus):
    """Context manager for TestClient whose app uses *bus* as its event bus."""
    app = create_app()
    app.state.event_bus = bus
    with TestClient(app) as client:
        yield client
```

- [ ] **Step 2: Update test functions to use the new context manager**

Update `test_sse_emits_stage_progress_in_order`, `test_sse_data_lines_contain_stage_field`, and `test_sse_terminates_on_run_failed`.

- [ ] **Step 3: Run tests to verify**

Run: `pytest tests/app/test_sse_events.py`
Expected: PASS
