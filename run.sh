#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <path_to_video>"
  exit 1
fi

VIDEO_PATH="$1"
OUTPUT_DIR="card_capture_output"
DB_PATH="${OUTPUT_DIR}/cards.sqlite"
CONFIG_PATH="card_capture_config.json"

echo "=== Processing Video ==="
.venv/bin/python -m card_capture.cli process "$VIDEO_PATH" --output-dir "$OUTPUT_DIR" --db "$DB_PATH" --config "$CONFIG_PATH"

if [ $? -ne 0 ]; then
  echo "Processing failed. Exiting."
  exit 1
fi

echo "=== Starting Review Server ==="
.venv/bin/python -m card_capture.cli review --db "$DB_PATH" --host 0.0.0.0 --port 8001
