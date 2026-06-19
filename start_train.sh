#!/bin/bash
set -e

SPLIT_STRATEGY="image"
ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --split_strategy)
            SPLIT_STRATEGY="$2"
            shift 2
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

case "$SPLIT_STRATEGY" in
    patient)
        TRAIN_SCRIPT="cli/train_patient_split.py"
        ;;
    image)
        TRAIN_SCRIPT="cli/train.py"
        ;;
    *)
        echo "Fehler: unbekannte split_strategy '$SPLIT_STRATEGY'"
        echo "Erlaubt: patient, image"
        exit 1
        ;;
esac

LOG_DIR="logs/training"

RUN_ID=$(python "$TRAIN_SCRIPT" "${ARGS[@]}" --dry_run_name | awk -F': ' '/Run ID/{print $2}')

if [ -z "$RUN_ID" ]; then
    echo "Fehler: RUN_ID konnte nicht generiert werden."
    exit 1
fi

mkdir -p "$LOG_DIR"

LOGFILE="$LOG_DIR/${RUN_ID}.out"

echo "======================="
echo "Split Strategy: $SPLIT_STRATEGY"
echo "Run: $RUN_ID"
echo "Script: $TRAIN_SCRIPT"
echo "Log: $LOGFILE"
echo "======================="

nohup python -u "$TRAIN_SCRIPT" "${ARGS[@]}" > "$LOGFILE" 2>&1 &

PID=$!

echo "Started PID: $PID"
echo
echo "Following log..."
echo

sleep 2
tail --pid=$PID -f "$LOGFILE"

echo
echo "Training finished."