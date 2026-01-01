#!/bin/bash

LOG_DIR="outputs/experiment_logs"

if [ ! -d "$LOG_DIR" ]; then
  echo "Error: Log directory '$LOG_DIR' not found"
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

  # Count statuses across all log files
  total_logs=$(find "$LOG_DIR" -name "*.log" 2>/dev/null | wc -l | tr -d ' ')
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

  # Show recently completed (newest first)
  echo "Recent Completions (newest first):"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  find "$LOG_DIR" -name "*.log" -print0 2>/dev/null | \
    xargs -0 grep -h "COMPLETED\|FAILED" 2>/dev/null | \
    sort -r | \
    head -10
  echo ""
  echo ""

  # Show currently running (based on recent activity, excluding completed/failed)
  echo "Currently Active Jobs:"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  find "$LOG_DIR" -name "*.log" -mmin -2 2>/dev/null | while read -r logfile; do
    if ! grep -q "COMPLETED\|FAILED" "$logfile" 2>/dev/null; then
      # Parse the filename and display in a more readable format
      filename=$(basename "$logfile" .log)
      # Extract key values and display compactly
      seed=$(echo "$filename" | grep -o 'seed=[^_]*' | cut -d= -f2)
      dataset_domain=$(echo "$filename" | grep -o 'dataset_domain=[^_]*' | cut -d= -f2)
      dataset_name=$(echo "$filename" | grep -o 'dataset_name=[^_]*' | cut -d= -f2)
      cpd_method=$(echo "$filename" | grep -o 'cpd_method=[^_]*' | cut -d= -f2)
      cost=$(echo "$filename" | grep -o 'cpd_cost_function=[^_]*' | cut -d= -f2)
      forecaster=$(echo "$filename" | grep -o 'forecaster_type=[^_]*' | cut -d= -f2)
      echo "  [seed=$seed] $dataset_domain $dataset_name | $cpd_method/$cost | $forecaster"
    fi
  done | head -5
  echo ""

  sleep 5
done
