.PHONY: dev kill-servers harness baseline test

LOG := /tmp/card-capture-dev.log
API_PORT := 8002
WEB_PORT := 5173

# Stop any dev servers left over from a previous `make dev`.
# A stale vite still holding :$(WEB_PORT) is the usual reason UI edits "don't
# show up": the old server keeps serving the page your browser is pointed at
# while a freshly started vite silently moves to the next free port. pkill
# catches uvicorn's reload parent+worker (and any orphaned vite under this
# repo); lsof is the backstop for whatever still owns the ports.
kill-servers:
	@echo "==> Stopping stale dev servers (uvicorn :$(API_PORT), vite :$(WEB_PORT))"
	@-pkill -9 -f 'uvicorn app.main:app' 2>/dev/null || true
	@-pkill -9 -f 'app/web/node_modules/.bin/vite' 2>/dev/null || true
	@-lsof -ti tcp:$(API_PORT) 2>/dev/null | xargs kill -9 2>/dev/null || true
	@-lsof -ti tcp:$(WEB_PORT) 2>/dev/null | xargs kill -9 2>/dev/null || true
	@sleep 1

dev: kill-servers
	@echo "==> Logging uvicorn output to $(LOG)"
	@echo "==> UI  (open this): http://localhost:$(WEB_PORT)"
	@echo "==> API (backend):   http://localhost:$(API_PORT)"
	@trap 'kill 0' EXIT INT TERM; \
	.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port $(API_PORT) --reload 2>&1 | tee $(LOG) & \
	cd app/web && npm run dev -- --port $(WEB_PORT) --strictPort

test:
	.venv/bin/python -m pytest tests/

harness:
	.venv/bin/python -m card_capture.cli harness run

baseline:
	.venv/bin/python -m card_capture.cli harness run --baseline reports/baseline_v3.json
