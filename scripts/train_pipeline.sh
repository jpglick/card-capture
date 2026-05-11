#!/bin/bash
# Pipeline V4 Phase A — Dataset generation + classifier training
# Run this in your local terminal (requires MPS/GPU for training)
set -e

cd "$(dirname "$0")/.."

CARD_CAPTURE=".venv/bin/card-capture"
if [ ! -f "$CARD_CAPTURE" ]; then
  echo "ERROR: $CARD_CAPTURE not found. Run: pip install -e '.[review]'" >&2
  exit 1
fi

DB="${DB:-card_capture_output/cards.sqlite}"
DATASET_DIR="${DATASET_DIR:-data/presence_dataset}"
MODEL_OUT="${MODEL_OUT:-models/presence_classifier.pt}"
EPOCHS="${EPOCHS:-8}"
BATCH_SIZE="${BATCH_SIZE:-64}"

echo "=== Task 8: Export training dataset ==="
$CARD_CAPTURE dataset export \
  --db "$DB" \
  --out-dir "$DATASET_DIR"

echo ""
echo "Dataset counts:"
echo "  Positives: $(ls "$DATASET_DIR/positives/" 2>/dev/null | wc -l | tr -d ' ')"
echo "  Negatives: $(ls "$DATASET_DIR/negatives/" 2>/dev/null | wc -l | tr -d ' ')"

POS_COUNT=$(ls "$DATASET_DIR/positives/" 2>/dev/null | wc -l | tr -d ' ')
if [ "$POS_COUNT" -lt 200 ]; then
  echo "WARNING: Only $POS_COUNT positives found (target >= 200). Consider processing more videos."
  echo "You can also lower the confidence floor: DB=$DB DATASET_DIR=$DATASET_DIR $0"
fi

echo ""
echo "=== Task 9: Train presence classifier ==="
$CARD_CAPTURE train presence \
  --data "$DATASET_DIR" \
  --out "$MODEL_OUT" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE"

echo ""
echo "=== Done ==="
echo "Model saved to: $MODEL_OUT"
echo ""
echo "Next step — commit the model:"
echo "  git add $MODEL_OUT"
echo '  git commit -m "feat(presence): train MobileNetV3-Small classifier (val_acc=X.XX)"'
