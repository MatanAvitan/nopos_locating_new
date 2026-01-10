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
MAGENTA='\033[0;35m'
WHITE='\033[1;37m'
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

print_job_summary() {
    echo -e "${BLUE}═══ JOB SUMMARY ════════════════════════════════════════════════════════════════${NC}"
    
    # Count jobs by state
    RUNNING_COUNT=$(squeue -u $USER -h -t RUNNING 2>/dev/null | wc -l)
    PENDING_COUNT=$(squeue -u $USER -h -t PENDING 2>/dev/null | wc -l)
    TOTAL_COUNT=$((RUNNING_COUNT + PENDING_COUNT))
    
    echo -e "  ${WHITE}Total Jobs:${NC} $TOTAL_COUNT    ${GREEN}Running:${NC} $RUNNING_COUNT    ${YELLOW}Pending:${NC} $PENDING_COUNT"
    echo ""
}

print_job_details() {
    echo -e "${BLUE}═══ RUNNING JOBS ═══════════════════════════════════════════════════════════════${NC}"
    
    # Get running jobs with detailed info
    RUNNING=$(squeue -u $USER -h -t RUNNING -o "%.10i %.25j %.10P %.8T %.10M %.10l %.6m %.4C %N" 2>/dev/null)
    
    if [ -z "$RUNNING" ]; then
        echo -e "  ${YELLOW}No running jobs${NC}"
    else
        echo -e "  ${WHITE}JOBID      NAME                      PARTITION  STATE    TIME       LIMIT      MEM    CPU  NODE${NC}"
        echo "$RUNNING" | while read line; do
            echo -e "  ${GREEN}$line${NC}"
        done
    fi
    echo ""
    
    echo -e "${BLUE}═══ PENDING JOBS ═══════════════════════════════════════════════════════════════${NC}"
    
    # Get pending jobs
    PENDING=$(squeue -u $USER -h -t PENDING -o "%.10i %.25j %.10P %.8T %.10M %.10l %.6m %.4C %R" 2>/dev/null)
    
    if [ -z "$PENDING" ]; then
        echo -e "  ${YELLOW}No pending jobs${NC}"
    else
        echo -e "  ${WHITE}JOBID      NAME                      PARTITION  STATE    TIME       LIMIT      MEM    CPU  REASON${NC}"
        echo "$PENDING" | while read line; do
            echo -e "  ${YELLOW}$line${NC}"
        done
    fi
    echo ""
}

print_resource_usage() {
    echo -e "${BLUE}═══ RESOURCE USAGE BY PARTITION ════════════════════════════════════════════════${NC}"
    
    # Get partitions being used by user's jobs
    PARTITIONS=$(squeue -u $USER -h -o "%P" 2>/dev/null | sort -u)
    
    if [ -z "$PARTITIONS" ]; then
        echo -e "  ${YELLOW}No active jobs - no resources in use${NC}"
    else
        for part in $PARTITIONS; do
            # Get partition info
            PART_INFO=$(sinfo -p $part -h -o "%P %a %D %C %G %m" 2>/dev/null | head -1)
            PART_NAME=$(echo "$PART_INFO" | awk '{print $1}')
            PART_STATE=$(echo "$PART_INFO" | awk '{print $2}')
            PART_NODES=$(echo "$PART_INFO" | awk '{print $3}')
            PART_CPUS=$(echo "$PART_INFO" | awk '{print $4}')  # A/I/O/T format
            PART_GPUS=$(echo "$PART_INFO" | awk '{print $5}')
            PART_MEM=$(echo "$PART_INFO" | awk '{print $6}')
            
            # Count user's jobs on this partition
            USER_JOBS=$(squeue -u $USER -p $part -h 2>/dev/null | wc -l)
            
            # Get total memory requested by user on this partition
            USER_MEM=$(squeue -u $USER -p $part -h -o "%m" 2>/dev/null | awk '{sum+=$1} END {print sum}')
            USER_MEM=${USER_MEM:-0}
            
            # Get total CPUs requested by user
            USER_CPUS=$(squeue -u $USER -p $part -h -o "%C" 2>/dev/null | awk '{sum+=$1} END {print sum}')
            USER_CPUS=${USER_CPUS:-0}
            
            # Get total GPUs requested by user
            USER_GPUS=$(squeue -u $USER -p $part -h -o "%b" 2>/dev/null | grep -oP '\d+' | awk '{sum+=$1} END {print sum}')
            USER_GPUS=${USER_GPUS:-0}
            
            echo -e "  ${MAGENTA}Partition: ${WHITE}$part${NC}"
            echo -e "    State: $PART_STATE | Nodes: $PART_NODES | Available GPUs: $PART_GPUS"
            echo -e "    ${CYAN}Your usage:${NC} $USER_JOBS jobs | ${USER_MEM}MB memory | $USER_CPUS CPUs | $USER_GPUS GPUs"
            echo ""
        done
    fi
    
    # Show preferred partitions info
    echo -e "  ${WHITE}Recommended Partitions:${NC}"
    echo -e "    ${GREEN}H200-4h${NC}  - H200 GPUs, 4h limit (node: hpc8h200-01)"
    echo -e "    ${GREEN}H200-12h${NC} - H200 GPUs, 12h limit (node: hpc8h200-01)"
    echo ""
}

print_completed_jobs() {
    echo -e "${BLUE}═══ RECENTLY COMPLETED (last 1 hour) ═══════════════════════════════════════════${NC}"
    
    # Get recently completed jobs
    COMPLETED=$(sacct -u $USER --starttime=$(date -d '1 hour ago' '+%Y-%m-%dT%H:%M:%S') \
                --format=JobID,JobName%25,Partition%10,State%10,ExitCode,Elapsed,MaxRSS \
                --noheader 2>/dev/null | grep -v "\.batch" | grep -v "\.extern" | head -10)
    
    if [ -z "$COMPLETED" ]; then
        echo -e "  ${YELLOW}No recently completed jobs${NC}"
    else
        echo -e "  ${WHITE}JOBID      NAME                      PARTITION  STATE      EXIT   ELAPSED  MAXMEM${NC}"
        echo "$COMPLETED" | while read line; do
            if echo "$line" | grep -q "COMPLETED"; then
                echo -e "  ${GREEN}$line${NC}"
            elif echo "$line" | grep -q "FAILED"; then
                echo -e "  ${RED}$line${NC}"
            elif echo "$line" | grep -q "CANCELLED"; then
                echo -e "  ${YELLOW}$line${NC}"
            elif echo "$line" | grep -q "RUNNING"; then
                echo -e "  ${CYAN}$line${NC}"
            else
                echo "$line"
            fi
        done
    fi
    echo ""
}

print_results_status() {
    echo -e "${BLUE}═══ EXPERIMENT RESULTS STATUS ══════════════════════════════════════════════════${NC}"
    
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
                echo -e "  ${YELLOW}○${NC} $name (in progress)"
            fi
        else
            echo -e "  ${RED}✗${NC} $name (not started)"
        fi
    done
    echo ""
}

print_log_tails() {
    echo -e "${BLUE}═══ RECENT LOG OUTPUT ══════════════════════════════════════════════════════════${NC}"
    
    cd "$PROJECT_DIR"
    
    # Find most recently modified log files
    LOGS=$(ls -t logs/slurm_*.out 2>/dev/null | head -3)
    
    if [ -z "$LOGS" ]; then
        echo -e "  ${YELLOW}No Slurm log files found yet${NC}"
    else
        for log in $LOGS; do
            if [ -f "$log" ]; then
                # Extract job name from filename
                BASENAME=$(basename "$log" .out)
                echo -e "  ${CYAN}─── ${BASENAME} ───${NC}"
                tail -n 2 "$log" 2>/dev/null | sed 's/^/    /' || echo "    (empty)"
                echo ""
            fi
        done
    fi
}

print_help() {
    echo -e "${BLUE}═══ QUICK COMMANDS ═════════════════════════════════════════════════════════════${NC}"
    echo -e "  ${WHITE}scancel <job_id>${NC}                        Cancel a job"
    echo -e "  ${WHITE}scancel -u \$USER${NC}                        Cancel all your jobs"
    echo -e "  ${WHITE}tail -f logs/slurm_<name>_<id>.out${NC}      Follow job output"
    echo -e "  ${WHITE}scontrol show job <id>${NC}                  Detailed job info"
    echo ""
    echo -e "  ${YELLOW}Refreshing every ${REFRESH_INTERVAL}s... Press Ctrl+C to exit${NC}"
}

# Main loop
main() {
    while true; do
        clear_screen
        print_header
        print_job_summary
        print_job_details
        print_resource_usage
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
