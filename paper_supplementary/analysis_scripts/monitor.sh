#!/usr/bin/env bash
# Monitor running AURA experiments
# Usage: ./scripts/monitor.sh

RESULTS_DIR="evaluation/results"

echo "=== AURA Experiment Monitor ==="
echo ""

# Check progress file
PROGRESS=$(ls -t "$RESULTS_DIR"/.progress_* 2>/dev/null | head -1)
if [ -n "$PROGRESS" ]; then
    echo "Current status: $(cat "$PROGRESS")"
else
    echo "No active experiment detected."
fi

# Check latest log
LOG=$(ls -t "$RESULTS_DIR"/experiment_*.log 2>/dev/null | head -1)
if [ -n "$LOG" ]; then
    echo ""
    echo "Latest log ($LOG) — last 20 lines:"
    echo "---"
    tail -20 "$LOG"
fi

# List result files
echo ""
echo "Result files:"
ls -la "$RESULTS_DIR"/rq*.json 2>/dev/null || echo "  (none yet)"
ls -la "$RESULTS_DIR"/*multiseed*.json 2>/dev/null || true

# Check if experiment process is running
echo ""
if pgrep -f "evaluation.run_experiments" > /dev/null 2>&1; then
    echo "Experiment process: RUNNING (PID $(pgrep -f 'evaluation.run_experiments'))"
else
    echo "Experiment process: NOT RUNNING"
fi

if pgrep -f "demo.town.server" > /dev/null 2>&1; then
    echo "Town server: RUNNING (PID $(pgrep -f 'demo.town.server'))"
else
    echo "Town server: NOT RUNNING"
fi
