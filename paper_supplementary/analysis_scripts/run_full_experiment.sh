#!/usr/bin/env bash
# =============================================================================
# AURA Full Experiment Pipeline (v2 — with multi-seed + monitoring)
#
# Usage:
#   ./scripts/run_full_experiment.sh                  # Run all RQs (single seed)
#   ./scripts/run_full_experiment.sh --multi-seed     # Run with 3 seeds for stats
#   ./scripts/run_full_experiment.sh --rq 1 2         # Run specific RQs
#   ./scripts/run_full_experiment.sh --smoke           # Quick smoke test
#
# Background usage (user can disconnect):
#   nohup ./scripts/run_full_experiment.sh --multi-seed > experiment.log 2>&1 &
# =============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log() { echo -e "$(date '+%H:%M:%S') ${GREEN}[AURA]${NC} $*"; }
warn() { echo -e "$(date '+%H:%M:%S') ${YELLOW}[WARN]${NC} $*"; }
fail() { echo -e "$(date '+%H:%M:%S') ${RED}[FAIL]${NC} $*"; exit 1; }
status() { echo -e "$(date '+%H:%M:%S') ${CYAN}[STATUS]${NC} $*"; }

# ── Pre-flight checks ──────────────────────────────────
log "Pre-flight checks..."

if [ ! -f .env ]; then
    fail ".env file not found. Copy .env.example and fill in your API key."
fi

python3 -c "import openai" 2>/dev/null || fail "openai package not installed. Run: make install"

# ── Parse args ──────────────────────────────────────────
SMOKE=false
MULTI_SEED=false
RQ_ARGS="--rq all"
STEPS=100
QUERIES=50

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke)
            SMOKE=true
            STEPS=5
            QUERIES=5
            shift
            ;;
        --multi-seed)
            MULTI_SEED=true
            shift
            ;;
        --rq)
            shift
            RQ_ARGS="--rq $*"
            break
            ;;
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --queries)
            QUERIES="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# ── Timestamps & logging ─────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="evaluation/results"
LOG_FILE="${RESULTS_DIR}/experiment_${TIMESTAMP}.log"
PROGRESS_FILE="${RESULTS_DIR}/.progress_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

# Progress tracking
update_progress() {
    echo "$1" > "$PROGRESS_FILE"
    status "$1"
}

# ── Step 1: Quick sanity tests ──────────────────────────
update_progress "Step 1/5: Running sanity tests..."
log "Running architecture tests..."
if python3 -m pytest tests/test_architecture.py -v --tb=short 2>&1 | tail -5; then
    log "Architecture tests passed"
else
    warn "Some architecture tests failed (continuing anyway)"
fi

log "Running core pipeline tests..."
if python3 -m pytest tests/test_core_pipeline.py -v --tb=short 2>&1 | tail -5; then
    log "Core tests passed"
else
    warn "Some core tests failed (continuing anyway)"
fi

# ── Step 2: Start server ───────────────────────────────
update_progress "Step 2/5: Starting server..."
SERVER_URL="http://127.0.0.1:7861"
SERVER_PID=""

if curl -s "$SERVER_URL/api/health" | grep -q '"ok"' 2>/dev/null; then
    log "Server already running at $SERVER_URL"
else
    log "Starting AURA Town server..."
    python3 -m demo.town.server &
    SERVER_PID=$!

    for i in $(seq 1 60); do
        if curl -s "$SERVER_URL/api/health" | grep -q '"ok"' 2>/dev/null; then
            log "Server ready (took ${i}s)"
            break
        fi
        sleep 1
        if [ "$i" -eq 60 ]; then
            fail "Server failed to start within 60s"
        fi
    done
fi

# Cleanup on exit
cleanup() {
    if [ -n "$SERVER_PID" ]; then
        log "Stopping server (PID $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    rm -f "$PROGRESS_FILE"
    log "Cleanup done."
}
trap cleanup EXIT

# ── Step 3: Run main experiments ────────────────────────
update_progress "Step 3/5: Running main experiments (steps=$STEPS, queries=$QUERIES)..."

MULTI_SEED_FLAG=""
if [ "$MULTI_SEED" = true ]; then
    MULTI_SEED_FLAG="--multi-seed --seeds 42 123 456"
    log "Multi-seed mode enabled (seeds: 42, 123, 456)"
fi

python3 -m evaluation.run_experiments \
    $RQ_ARGS \
    --steps "$STEPS" \
    --queries "$QUERIES" \
    --server "$SERVER_URL" \
    $MULTI_SEED_FLAG \
    2>&1 | tee "$LOG_FILE"

# ── Step 4: Run InteractiveBench (if not smoke) ─────────
if [ "$SMOKE" = false ]; then
    update_progress "Step 4/5: Running InteractiveBench..."

    # Trust Game (our strongest result)
    log "Running Trust Game benchmark..."
    python3 -m evaluation.interactivebench.run_experiments \
        --bench trust \
        --repeats 5 \
        --delta 0.9 \
        2>&1 | tee -a "$LOG_FILE" || warn "Trust game had issues"

    # Math (with probing enabled)
    log "Running Math benchmark..."
    python3 -m evaluation.interactivebench.run_experiments \
        --bench math \
        --max_rounds 20 \
        --max_n 20 \
        2>&1 | tee -a "$LOG_FILE" || warn "Math benchmark had issues"
else
    update_progress "Step 4/5: Skipping InteractiveBench (smoke mode)..."
fi

# ── Step 5: Generate analysis ──────────────────────────
update_progress "Step 5/5: Generating analysis..."

if ls "${RESULTS_DIR}"/rq*.json 1>/dev/null 2>&1; then
    log "Results saved in ${RESULTS_DIR}/"
    ls -la "${RESULTS_DIR}"/rq*.json 2>/dev/null || true
    ls -la "${RESULTS_DIR}"/*multiseed*.json 2>/dev/null || true

    if [ -f evaluation/generate_tables.py ]; then
        log "Generating summary tables..."
        python3 -m evaluation.generate_tables 2>/dev/null || warn "Table generation had issues"
    fi
else
    warn "No result JSON files found"
fi

# ── Summary ────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  EXPERIMENT PIPELINE COMPLETE"
echo "============================================================"
echo "  Timestamp:  $TIMESTAMP"
echo "  Log file:   $LOG_FILE"
echo "  Results:    ${RESULTS_DIR}/"
echo "  Mode:       $([ "$MULTI_SEED" = true ] && echo 'Multi-seed (3 runs)' || echo 'Single seed')"
echo "  Steps:      $STEPS"
echo "  Queries:    $QUERIES"
echo "============================================================"

if ls "${RESULTS_DIR}"/rq*.json 1>/dev/null 2>&1; then
    echo ""
    echo "Result files:"
    ls -la "${RESULTS_DIR}"/rq*.json 2>/dev/null
    ls -la "${RESULTS_DIR}"/*multiseed*.json 2>/dev/null || true
fi

log "Done!"
