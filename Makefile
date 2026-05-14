.PHONY: dev harness baseline test

dev:
	@trap 'kill 0' SIGINT; \
	.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload & \
	cd app/web && npm run dev

test:
	.venv/bin/python -m pytest tests/

harness:
	.venv/bin/python -m card_capture.cli harness run

baseline:
	.venv/bin/python -m card_capture.cli harness run --baseline reports/baseline_v3.json
