#!/usr/bin/env bash
echo "=== AURA Experiment Progress ==="
echo "Time: $(date)"
echo ""
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="$ROOT/logs/main_exp_final.log"
PID=$(ps aux | grep "run_experiment" | grep -v grep | awk "{print \$2}" | head -1)
if [ -n "$PID" ]; then
    echo "Status: RUNNING (PID $PID)"
else
    echo "Status: STOPPED"
fi
echo ""
echo "Progress:"
echo "  RQ1 seeds done: $(grep -c "Saved to.*rq1_grounding" "$LOG" 2>/dev/null || echo 0)"
echo "  RQ2 seeds done: $(grep -c "Saved to.*rq2_factual" "$LOG" 2>/dev/null || echo 0)"
echo "  RQ3 seeds done: $(grep -c "Saved to.*rq3_ablation" "$LOG" 2>/dev/null || echo 0)"
echo "  RQ6 seeds done: $(grep -c "Saved to.*rq6_probe" "$LOG" 2>/dev/null || echo 0)"
echo "  Multi-seed summaries: $(grep -c "Multi-Seed Summary" "$LOG" 2>/dev/null || echo 0)"
echo ""
echo "Last 5 lines:"
tail -5 "$LOG" 2>/dev/null
