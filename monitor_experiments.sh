#!/bin/bash

#═══════════════════════════════════════════════════════════════════════════════
# Experiment Progress Monitor
# Monitors log files and displays experiment status with intelligent caching
#═══════════════════════════════════════════════════════════════════════════════

#───────────────────────────────────────────────────────────────────────────────
# Configuration
#───────────────────────────────────────────────────────────────────────────────
LOG_DIR="outputs/experiment_logs"
REFRESH_INTERVAL=60
MAX_ACTIVE_JOBS_DISPLAY=4
MAX_RECENT_COMPLETIONS=10

#───────────────────────────────────────────────────────────────────────────────
# Initialization
#───────────────────────────────────────────────────────────────────────────────
if [ ! -d "$LOG_DIR" ]; then
  echo "Error: Log directory '$LOG_DIR' not found"
  exit 1
fi

# Cache storage for file status, modification times, and metadata
declare -A file_status_cache
declare -A file_mtime_cache
declare -A file_metadata_cache
declare -A recent_completions

#───────────────────────────────────────────────────────────────────────────────
# Utility Functions
#───────────────────────────────────────────────────────────────────────────────

# Create a safe cache key from filepath (avoids bash arithmetic evaluation issues)
make_cache_key() {
  echo "$1" | sed 's/[^a-zA-Z0-9]/_/g'
}

#───────────────────────────────────────────────────────────────────────────────
# Cache Functions
#───────────────────────────────────────────────────────────────────────────────

# Parse metadata from filename (cached after first parse)
parse_metadata() {
  local logfile="$1"
  local cache_key=$(make_cache_key "$logfile")
  local cached="${file_metadata_cache[$cache_key]}"

  # Return cached metadata if available
  if [[ -n "$cached" ]]; then
    echo "$cached"
    return
  fi

  # Extract metadata from filepath
  local seed=$(echo "$logfile" | grep -o 'seed=[^/]*' | cut -d= -f2)
  local dataset_domain=$(echo "$logfile" | grep -o 'dataset_domain=[^/]*' | cut -d= -f2)
  local dataset_name=$(echo "$logfile" | grep -o 'dataset_name=[^/]*' | cut -d= -f2)
  local cpd_method=$(echo "$logfile" | grep -o 'cpd_method=[^/]*' | cut -d= -f2)
  local cpd_cost_function=$(echo "$logfile" | grep -o 'cpd_cost_function=[^/]*' | cut -d= -f2)
  local forecaster=$(echo "$logfile" | grep -o 'forecaster_type=[^/]*' | cut -d= -f2)

  # Format and cache metadata
  local metadata="[seed=$seed] $dataset_domain $dataset_name | $cpd_method $cpd_cost_function | $forecaster"
  file_metadata_cache[$cache_key]="$metadata"
  echo "$metadata"
}

# Get file status (completed/failed/running) with caching based on mtime
get_file_status() {
  local logfile="$1"
  local cache_key=$(make_cache_key "$logfile")

  # Use Linux-compatible stat command
  local current_mtime=$(stat -c %Y "$logfile" 2>/dev/null || echo "0")

  local cached_mtime="${file_mtime_cache[$cache_key]}"
  local cached_status="${file_status_cache[$cache_key]}"

  # Return cached status if file hasn't been modified
  if [[ "$cached_mtime" == "$current_mtime" && -n "$cached_status" ]]; then
    echo "$cached_status"
    return
  fi

  # File changed or not cached - determine status by checking log content
  local status="running"
  if grep -q "COMPLETED" "$logfile" 2>/dev/null; then
    status="completed"
  elif grep -q "FAILED" "$logfile" 2>/dev/null; then
    status="failed"
  fi

  # Update cache with new status and mtime
  file_mtime_cache[$cache_key]="$current_mtime"
  file_status_cache[$cache_key]="$status"

  echo "$status"
}

#───────────────────────────────────────────────────────────────────────────────
# Data Collection Functions
#───────────────────────────────────────────────────────────────────────────────

# Collect all log files
get_all_logs() {
  local -a logs=()
  while IFS= read -r -d '' file; do
    logs+=("$file")
  done < <(find "$LOG_DIR" -name "*.log" -print0 2>/dev/null)
  printf '%s\n' "${logs[@]}"
}

# Count jobs by status using cached data
count_jobs_by_status() {
  local -a all_logs=("$@")
  local completed=0 failed=0 running=0

  for logfile in "${all_logs[@]}"; do
    status=$(get_file_status "$logfile")
    case "$status" in
      completed) ((completed++));;
      failed) ((failed++));;
      running) ((running++));;
    esac
  done

  echo "$completed $failed $running"
}

# Collect completion messages from finished jobs
collect_completion_messages() {
  local -a all_logs=("$@")
  recent_completions=()

  for logfile in "${all_logs[@]}"; do
    status=$(get_file_status "$logfile")
    if [[ "$status" == "completed" || "$status" == "failed" ]]; then
      msg=$(grep "COMPLETED\|FAILED" "$logfile" 2>/dev/null | tail -1)

      if [[ -n "$msg" ]]; then
        # Calculate total execution time using Linux-compatible stat
        local birth_time=$(stat -c %W "$logfile" 2>/dev/null)
        # Fall back to %Y (mtime) if birth time (%W) is not available or returns 0
        if [[ "$birth_time" == "0" || -z "$birth_time" ]]; then
          # Use first line timestamp from log if available, else use mtime
          birth_time=$(stat -c %Y "$logfile" 2>/dev/null || echo "0")
        fi
        local end_time=$(stat -c %Y "$logfile" 2>/dev/null || echo "0")

        local elapsed=$((end_time - birth_time))
        local hours=$((elapsed / 3600))
        local minutes=$(((elapsed % 3600) / 60))
        local seconds=$((elapsed % 60))
        local elapsed_str="${hours}h ${minutes}m ${seconds}s"

        # Prepend end_time (for sorting), elapsed time, and message
        recent_completions+=("$end_time|[$elapsed_str] $msg")
      fi
    fi
  done
}

# Get running jobs
get_active_jobs() {
  local -a all_logs=("$@")
  local count=0

  for logfile in "${all_logs[@]}"; do
    # Always check status to ensure cache is updated for all files
    status=$(get_file_status "$logfile")

    # Only display up to MAX_ACTIVE_JOBS_DISPLAY running jobs
    if [[ "$status" == "running" ]]; then
      if [[ $count -lt $MAX_ACTIVE_JOBS_DISPLAY ]]; then
        metadata=$(parse_metadata "$logfile")

        # Get file birth time (creation time) using Linux-compatible stat
        local birth_time=$(stat -c %W "$logfile" 2>/dev/null)
        # Fall back to mtime if birth time is not available or returns 0
        if [[ "$birth_time" == "0" || -z "$birth_time" ]]; then
          birth_time=$(stat -c %Y "$logfile" 2>/dev/null || echo "0")
        fi
        local start_time=$(date -d "@$birth_time" '+%H:%M:%S' 2>/dev/null || echo "N/A")

        # Calculate elapsed time
        local current_time=$(date +%s)
        local elapsed=$((current_time - birth_time))
        local hours=$((elapsed / 3600))
        local minutes=$(((elapsed % 3600) / 60))
        local elapsed_str="${hours}h ${minutes}m"

        echo "  [Started: $start_time | Running: $elapsed_str] $metadata"
      fi
      ((count++))
    fi
  done
}

#───────────────────────────────────────────────────────────────────────────────
# Display Functions
#───────────────────────────────────────────────────────────────────────────────

display_header() {
  clear
  echo "=========================================="
  echo "  Experiment Progress Monitor"
  echo "=========================================="
  echo ""
}

display_summary() {
  local total=$1 completed=$2 failed=$3 running=$4

  echo "Status Summary (Updated: $(date '+%H:%M:%S'))         "
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Total Jobs:     $total"
  echo "Completed:      $completed"
  echo "Failed:         $failed"
  echo "Running:        $running"
  echo ""
}

display_recent_completions() {
  echo "Recent Completions (newest first):"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  # Sort by timestamp (descending), then strip timestamp before display
  printf '%s\n' "${recent_completions[@]}" | sort -t'|' -k1 -rn | cut -d'|' -f2- | head -$MAX_RECENT_COMPLETIONS
  echo ""
  echo ""
}

display_active_jobs() {
  echo "Currently Active Jobs:"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

#───────────────────────────────────────────────────────────────────────────────
# Main Loop
#───────────────────────────────────────────────────────────────────────────────

display_header

while true; do
  # Clear screen content while preserving header
  tput cup 4 0
  tput ed

  # Collect all log files
  all_logs=()
  while IFS= read -r file; do
    all_logs+=("$file")
  done < <(get_all_logs)
  total_logs=${#all_logs[@]}

  # Count jobs by status
  read completed failed running < <(count_jobs_by_status "${all_logs[@]}")

  # Display status summary
  display_summary "$total_logs" "$completed" "$failed" "$running"

  # Collect and display recent completions
  collect_completion_messages "${all_logs[@]}"
  display_recent_completions

  # Display active jobs
  display_active_jobs
  get_active_jobs "${all_logs[@]}"
  echo ""

  sleep $REFRESH_INTERVAL
done
