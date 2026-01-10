#!/bin/bash
# NoPE Experiment Slurm Monitor
# Usage: ./slurm_jobs/monitor.sh
#
# This script provides a real-time dashboard for monitoring Slurm jobs.
# Press Ctrl+C to exit.

REFRESH_INTERVAL=5
PROJECT_DIR="/home/nlp/matan_avitan/git/nopos_locating_new"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

clear_screen() {
    clear
}

print_header() {
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC}              ${YELLOW}NoPE Analysis - Slurm Job Monitor${NC}                              ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}              Last updated: $(date '+%Y-%m-%d %H:%M:%S')                            ${CYAN}║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_job_status() {
    echo -e "${BLUE}═══ JOB QUEUE ══════════════════════════════════════════════════════════════════${NC}"
    
    # Get job info
    JOBS=$(squeue -u $USER -o "%.10i %.30j %.8T %.10M %.10l %.6D %R" 2>/dev/null)
    
    if [ -z "$JOBS" ] || [ $(echo "$JOBS" | wc -l) -le 1 ]; then
        echo -e "${YELLOW}No jobs currently in queue${NC}"
    else
        # Print header
        echo -e "${GREEN}$(echo "$JOBS" | head -1)${NC}"
        # Print jobs with color coding
        echo "$JOBS" | tail -n +2 | while read line; do
            if echo "$line" | grep -q "RUNNING"; then
                echo -e "${GREEN}$line${NC}"
            elif echo "$line" | grep -q "PENDING"; then
                echo -e "${YELLOW}$line${NC}"
            elif echo "$line" | grep -q "FAILED\|CANCELLED"; then
                echo -e "${RED}$line${NC}"
            else
                echo "$line"
            fi
        done
    fi
    echo ""
}

print_completed_jobs() {
    echo -e "${BLUE}═══ RECENTLY COMPLETED (last 1 hour) ═══════════════════════════════════════════${NC}"
    
    # Get recently completed jobs
    COMPLETED=$(sacct -u $USER --starttime=$(date -d '1 hour ago' '+%Y-%m-%dT%H:%M:%S') \
                --format=JobID,JobName%30,State,ExitCode,Elapsed,End \
                --noheader 2>/dev/null | grep -v "\.batch" | head -10)
    
    if [ -z "$COMPLETED" ]; then
        echo -e "${YELLOW}No recently completed jobs${NC}"
    else
        echo "$COMPLETED" | while read line; do
            if echo "$line" | grep -q "COMPLETED"; then
                echo -e "${GREEN}$line${NC}"
            elif echo "$line" | grep -q "FAILED"; then
                echo -e "${RED}$line${NC}"
            elif echo "$line" | grep -q "RUNNING"; then
                echo -e "${CYAN}$line${NC}"
            else
                echo "$line"
            fi
        done
    fi
    echo ""
}

print_log_tails() {
    echo -e "${BLUE}═══ RECENT LOG OUTPUT ══════════════════════════════════════════════════════════${NC}"
    
    cd "$PROJECT_DIR"
    
    # Find most recently modified log files
    LOGS=$(ls -t logs/slurm_*.out 2>/dev/null | head -3)
    
    if [ -z "$LOGS" ]; then
        echo -e "${YELLOW}No Slurm log files found yet${NC}"
    else
        for log in $LOGS; do
            if [ -f "$log" ]; then
                echo -e "${CYAN}--- ${log} ---${NC}"
                tail -n 3 "$log" 2>/dev/null || echo "(empty)"
                echo ""
            fi
        done
    fi
}

print_results_status() {
    echo -e "${BLUE}═══ RESULTS STATUS ═════════════════════════════════════════════════════════════${NC}"
    
    cd "$PROJECT_DIR"
    
    declare -A EXPERIMENTS=(
        ["token_position_correlation"]="Token-Position Correlation"
        ["comprehensive_probe_analysis"]="Comprehensive Probe Analysis"
        ["higher_order_statistics"]="Higher-Order Statistics"
        ["decoding_vector_experiments"]="Decoding Vector Experiments"
        ["causal_interventions"]="Causal Interventions"
        ["training_dynamics"]="Training Dynamics"
    )
    
    for dir in "${!EXPERIMENTS[@]}"; do
        name="${EXPERIMENTS[$dir]}"
        result_dir="results/$dir"
        
        if [ -d "$result_dir" ]; then
            file_count=$(find "$result_dir" -type f 2>/dev/null | wc -l)
            if [ "$file_count" -gt 0 ]; then
                echo -e "  ${GREEN}✓${NC} $name ($file_count files)"
            else
                echo -e "  ${YELLOW}○${NC} $name (empty)"
            fi
        else
            echo -e "  ${RED}✗${NC} $name (not started)"
        fi
    done
    echo ""
}

print_gpu_info() {
    echo -e "${BLUE}═══ CLUSTER GPU AVAILABILITY ═══════════════════════════════════════════════════${NC}"
    
    # This only works if sinfo has GPU info
    sinfo -o "%P %a %D %G" 2>/dev/null | head -10 || echo "GPU info not available"
    echo ""
}

print_help() {
    echo -e "${BLUE}═══ COMMANDS ═══════════════════════════════════════════════════════════════════${NC}"
    echo "  scancel <job_id>     Cancel a job"
    echo "  tail -f logs/slurm_<name>_<id>.out   Follow specific job output"
    echo "  scontrol show job <id>               Detailed job info"
    echo ""
    echo -e "  ${YELLOW}Refreshing every ${REFRESH_INTERVAL}s... Press Ctrl+C to exit${NC}"
}

# Main loop
main() {
    while true; do
        clear_screen
        print_header
        print_job_status
        print_completed_jobs
        print_results_status
        print_log_tails
        print_help
        sleep $REFRESH_INTERVAL
    done
}

# Run with error handling
trap "echo -e '\n${YELLOW}Monitor stopped.${NC}'; exit 0" INT TERM
main
