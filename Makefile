.PHONY: dev harness baseline test

LOG := /tmp/card-capture-dev.log

dev:
	@echo "Logging uvicorn output to $(LOG)"
	@trap 'kill 0' SIGINT; \
	.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload 2>&1 | tee $(LOG) & \
	cd app/web && npm run dev

test:
	.venv/bin/python -m pytest tests/

harness:
	.venv/bin/python -m card_capture.cli harness run

baseline:
	.venv/bin/python -m card_capture.cli harness run --baseline reports/baseline_v3.json
