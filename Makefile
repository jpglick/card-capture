.PHONY: harness baseline test

test:
	.venv/bin/python -m pytest tests/

harness:
	.venv/bin/python -m card_capture.cli harness run

baseline:
	.venv/bin/python -m card_capture.cli harness run --baseline reports/baseline_v3.json
