#!/bin/bash
# Fast Slurm Job Monitor
# Usage: ./monitor.sh [OPTIONS]
#
# Options:
#   -r, --refresh SEC     Refresh interval in seconds (default: 10)
#   -l, --logs DIR        Log directory to monitor (default: ./logs)
#   -n, --lines N         Number of log lines to show per file (default: 5)
#   -j, --job JOBID       Follow specific job output
#   -o, --once            Run once and exit
#   -h, --help            Show this help

REFRESH=10
LOG_DIR="logs"
LINES=5
JOB=""
ONCE=false
SSH_KEY="$HOME/.ssh/dsinlp01_id_rsa"
HOST="slurm-login.lnx.biu.ac.il"

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--refresh) REFRESH="$2"; shift 2 ;;
        -l|--logs) LOG_DIR="$2"; shift 2 ;;
        -n|--lines) LINES="$2"; shift 2 ;;
        -j|--job) JOB="$2"; shift 2 ;;
        -o|--once) ONCE=true; shift ;;
        -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
        *) shift ;;
    esac
done

[[ "$LOG_DIR" != /* ]] && LOG_DIR="$(pwd)/$LOG_DIR"

# Colors
R=$'\e[31m'; G=$'\e[32m'; Y=$'\e[33m'; B=$'\e[34m'; C=$'\e[36m'
W=$'\e[1;37m'; D=$'\e[2m'; N=$'\e[0m'

# Check if local slurm
has_slurm() { command -v squeue &>/dev/null; }

# Run slurm command
slurm() {
    if has_slurm; then
        eval "$1"
    else
        ssh -i "$SSH_KEY" -o BatchMode=yes -o ConnectTimeout=3 -o StrictHostKeyChecking=no "$HOST" "$1" 2>/dev/null
    fi
}

# Follow job log
if [ -n "$JOB" ]; then
    f=$(find "$LOG_DIR" -name "*${JOB}*.out" 2>/dev/null | head -1)
    [ -f "$f" ] && exec tail -f "$f" || { echo "Log not found for $JOB"; exit 1; }
fi

# Build temp file for output
TMP=$(mktemp)
trap "rm -f $TMP; echo; exit 0" INT TERM EXIT

show() {
    > "$TMP"
    
    # Header
    echo "${C}════════════════════════════════════════════════════════════════════════════${N}" >> "$TMP"
    echo "  ${W}SLURM MONITOR${N}  │  $(date +%H:%M:%S)  │  ${D}Refresh: ${REFRESH}s${N}" >> "$TMP"
    echo "${C}════════════════════════════════════════════════════════════════════════════${N}" >> "$TMP"
    echo >> "$TMP"
    
    # Get queue (single SSH call with all data)
    local data=$(slurm 'squeue -u $(whoami) -o "%.8i %.9P %.16j %.8T %.8M %.8l %R" 2>/dev/null')
    
    local running=$(echo "$data" | grep -c RUNNING || true)
    local pending=$(echo "$data" | grep -c PENDING || true)
    running=${running:-0}
    pending=${pending:-0}
    
    echo "${W}Jobs:${N} ${G}● Running: $running${N}  ${Y}○ Pending: $pending${N}" >> "$TMP"
    echo >> "$TMP"
    
    # Running jobs table
    if [ "$running" -gt 0 ] 2>/dev/null; then
        echo "${G}▶ RUNNING${N}" >> "$TMP"
        echo "$data" | head -1 >> "$TMP"
        echo "$data" | grep RUNNING | while read line; do
            echo "${G}$line${N}"
        done >> "$TMP"
        echo >> "$TMP"
    fi
    
    # Pending jobs
    if [ "$pending" -gt 0 ] 2>/dev/null; then
        echo "${Y}◷ PENDING${N}" >> "$TMP"
        echo "$data" | grep PENDING | while read line; do
            echo "${Y}$line${N}"
        done >> "$TMP"
        echo >> "$TMP"
    fi
    
    # Recent completions (separate quick call)
    local acct=$(slurm 'sacct -u $(whoami) -S $(date -d "1 hour ago" +%Y-%m-%dT%H:%M 2>/dev/null || date -v-1H +%Y-%m-%dT%H:%M) --format=JobID%8,JobName%16,State%10,Elapsed --noheader 2>/dev/null | grep -v -E "\.(batch|extern)" | head -4')
    
    if [ -n "$acct" ]; then
        echo "${C}◉ COMPLETED (1h)${N}" >> "$TMP"
        echo "$acct" | while read line; do
            if echo "$line" | grep -q COMPLETED; then
                echo "${G}$line${N}"
            elif echo "$line" | grep -qE "FAIL|TIME|OUT_OF"; then
                echo "${R}$line${N}"
            else
                echo "$line"
            fi
        done >> "$TMP"
        echo >> "$TMP"
    fi
    
    # Local logs (fast, no SSH)
    echo "${W}◆ LOGS${N} ${D}($LOG_DIR)${N}" >> "$TMP"
    
    if [ -d "$LOG_DIR" ]; then
        for f in $(ls -t "$LOG_DIR"/*.out 2>/dev/null | head -3); do
            [ -f "$f" ] || continue
            local name=$(basename "$f")
            local age=$(( ($(date +%s) - $(stat -c %Y "$f" 2>/dev/null || stat -f %m "$f")) / 60 ))
            [ "$age" -gt 120 ] 2>/dev/null && continue
            
            echo "${C}─ ${name}${N} ${D}(${age}m)${N}" >> "$TMP"
            tail -$LINES "$f" 2>/dev/null | while IFS= read -r line; do
                line="${line:0:74}"
                if echo "$line" | grep -qiE "error|fail|exception"; then
                    echo "  ${R}$line${N}"
                elif echo "$line" | grep -qiE "iter.*loss|step.*loss|saving|epoch"; then
                    echo "  ${G}$line${N}"
                else
                    echo "  ${D}$line${N}"
                fi
            done >> "$TMP"
        done
    fi
    
    echo >> "$TMP"
    echo "${D}──────────────────────────────────────────────────────────────────────────────${N}" >> "$TMP"
    echo "${D}Commands: scancel <id>  │  tail -f logs/*<id>.out  │  Ctrl+C to exit${N}" >> "$TMP"
    
    # Clear and show all at once (smooth)
    clear
    cat "$TMP"
}

# Main loop
while true; do
    show
    $ONCE && exit 0
    
    # Countdown
    for ((i=REFRESH; i>0; i--)); do
        printf "\r${D}Refresh in %2ds...${N} " $i
        sleep 1 || break
    done
done
