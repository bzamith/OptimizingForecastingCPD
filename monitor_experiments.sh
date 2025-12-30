#!/bin/bash

LOG_DIR="outputs/experiment_logs"

if [ ! -d "$LOG_DIR" ]; then
  echo "Error: Log directory '$LOG_DIR' not found"
  echo "Make sure you're running experiment_executions_improved.sh first"
  exit 1
fi

# Clear screen and show header
clear
echo "=========================================="
echo "  Experiment Progress Monitor"
echo "=========================================="
echo ""

while true; do
  # Clear from cursor position down and move cursor to top
  tput cup 4 0
  tput ed

  # Count statuses
  total_logs=$(ls -1 "$LOG_DIR"/*.log 2>/dev/null | wc -l | tr -d ' ')
  completed=$(grep -rl "COMPLETED" "$LOG_DIR" 2>/dev/null | wc -l | tr -d ' ')
  failed=$(grep -rl "FAILED" "$LOG_DIR" 2>/dev/null | wc -l | tr -d ' ')
  running=$((total_logs - completed - failed))

  # Display summary
  echo "Status Summary (Updated: $(date '+%H:%M:%S'))         "
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Total Jobs:     $total_logs"
  echo "Completed:      $completed"
  echo "Failed:         $failed"
  echo "Running:        $running"
  echo ""

  # Show recently completed
  echo "Recent Completions:"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  grep -h "COMPLETED\|FAILED" "$LOG_DIR"/*.log 2>/dev/null | tail -10 | while read -r line; do
    echo "$line"
  done
  echo ""
  echo ""

  # Show currently running (based on recent activity, excluding completed/failed)
  echo "Currently Active Jobs:"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  find "$LOG_DIR" -name "*.log" -mmin -2 2>/dev/null | while read -r logfile; do
    if ! grep -q "COMPLETED\|FAILED" "$logfile" 2>/dev/null; then
      basename "$logfile" .log
    fi
  done | head -5
  echo ""

  sleep 5
done
