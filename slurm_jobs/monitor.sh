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
BOLD='\033[1m'

print_header() {
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC}                    ${YELLOW}${BOLD}NoPE Analysis - Slurm Job Monitor${NC}                              ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}                    Last updated: $(date '+%Y-%m-%d %H:%M:%S')                                ${CYAN}║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_job_summary() {
    echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}                                    JOB SUMMARY${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════════════${NC}"
    
    # Count jobs by state
    RUNNING_COUNT=$(squeue -u $USER -h -t RUNNING 2>/dev/null | wc -l)
    PENDING_COUNT=$(squeue -u $USER -h -t PENDING 2>/dev/null | wc -l)
    TOTAL_COUNT=$((RUNNING_COUNT + PENDING_COUNT))
    
    echo ""
    echo -e "  ${WHITE}Total Jobs:${NC} ${BOLD}$TOTAL_COUNT${NC}        ${GREEN}● Running:${NC} ${BOLD}$RUNNING_COUNT${NC}        ${YELLOW}○ Pending:${NC} ${BOLD}$PENDING_COUNT${NC}"
    echo ""
}

print_running_jobs() {
    echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}${BOLD}                                  RUNNING JOBS${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════════════${NC}"
    echo ""
    
    # Get running jobs
    RUNNING=$(squeue -u $USER -h -t RUNNING -o "%.10i|%.25j|%.10P|%.10M|%.10l|%.7m|%.4C|%N" 2>/dev/null)
    
    if [ -z "$RUNNING" ]; then
        echo -e "  ${YELLOW}No running jobs${NC}"
    else
        # Print header
        printf "  ${WHITE}%-10s %-25s %-12s %-10s %-10s %-8s %-5s %-20s${NC}\n" \
               "JOBID" "NAME" "PARTITION" "TIME" "LIMIT" "MEM" "CPUs" "NODE"
        echo -e "  ${WHITE}─────────────────────────────────────────────────────────────────────────────────────${NC}"
        
        # Print each job
        echo "$RUNNING" | while IFS='|' read jobid name partition time limit mem cpus node; do
            # Trim whitespace
            jobid=$(echo "$jobid" | xargs)
            name=$(echo "$name" | xargs)
            partition=$(echo "$partition" | xargs)
            time=$(echo "$time" | xargs)
            limit=$(echo "$limit" | xargs)
            mem=$(echo "$mem" | xargs)
            cpus=$(echo "$cpus" | xargs)
            node=$(echo "$node" | xargs)
            
            printf "  ${GREEN}%-10s %-25s %-12s %-10s %-10s %-8s %-5s %-20s${NC}\n" \
                   "$jobid" "$name" "$partition" "$time" "$limit" "$mem" "$cpus" "$node"
        done
    fi
    echo ""
}

print_pending_jobs() {
    echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${YELLOW}${BOLD}                                  PENDING JOBS${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════════════${NC}"
    echo ""
    
    # Get pending jobs
    PENDING=$(squeue -u $USER -h -t PENDING -o "%.10i|%.25j|%.10P|%.10l|%.7m|%.4C|%R" 2>/dev/null)
    
    if [ -z "$PENDING" ]; then
        echo -e "  ${YELLOW}No pending jobs${NC}"
    else
        # Print header
        printf "  ${WHITE}%-10s %-25s %-12s %-10s %-8s %-5s %-25s${NC}\n" \
               "JOBID" "NAME" "PARTITION" "LIMIT" "MEM" "CPUs" "REASON"
        echo -e "  ${WHITE}─────────────────────────────────────────────────────────────────────────────────────${NC}"
        
        # Print each job
        echo "$PENDING" | while IFS='|' read jobid name partition limit mem cpus reason; do
            jobid=$(echo "$jobid" | xargs)
            name=$(echo "$name" | xargs)
            partition=$(echo "$partition" | xargs)
            limit=$(echo "$limit" | xargs)
            mem=$(echo "$mem" | xargs)
            cpus=$(echo "$cpus" | xargs)
            reason=$(echo "$reason" | xargs)
            
            printf "  ${YELLOW}%-10s %-25s %-12s %-10s %-8s %-5s %-25s${NC}\n" \
                   "$jobid" "$name" "$partition" "$limit" "$mem" "$cpus" "$reason"
        done
    fi
    echo ""
}

print_resource_usage() {
    echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${MAGENTA}${BOLD}                              RESOURCE USAGE BY PARTITION${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════════════${NC}"
    echo ""
    
    # Get partitions being used by user's jobs
    PARTITIONS=$(squeue -u $USER -h -o "%P" 2>/dev/null | sort -u)
    
    if [ -z "$PARTITIONS" ]; then
        echo -e "  ${YELLOW}No active jobs - no resources in use${NC}"
    else
        for part in $PARTITIONS; do
            # Get partition info
            PART_INFO=$(sinfo -p $part -h -o "%P|%a|%D|%C|%G|%m" 2>/dev/null | head -1)
            PART_STATE=$(echo "$PART_INFO" | cut -d'|' -f2)
            PART_NODES=$(echo "$PART_INFO" | cut -d'|' -f3)
            PART_CPUS=$(echo "$PART_INFO" | cut -d'|' -f4)
            PART_GPUS=$(echo "$PART_INFO" | cut -d'|' -f5)
            PART_MEM=$(echo "$PART_INFO" | cut -d'|' -f6)
            
            # Count user's jobs on this partition
            USER_JOBS=$(squeue -u $USER -p $part -h 2>/dev/null | wc -l)
            
            # Get total memory requested by user on this partition (in MB)
            USER_MEM=$(squeue -u $USER -p $part -h -o "%m" 2>/dev/null | sed 's/G/*1024/g; s/M//g' | bc 2>/dev/null | awk '{sum+=$1} END {printf "%.0f", sum/1024}')
            USER_MEM=${USER_MEM:-0}
            
            # Get total CPUs requested by user
            USER_CPUS=$(squeue -u $USER -p $part -h -o "%C" 2>/dev/null | awk '{sum+=$1} END {print sum}')
            USER_CPUS=${USER_CPUS:-0}
            
            # Get total GPUs requested by user
            USER_GPUS=$(squeue -u $USER -p $part -h -o "%b" 2>/dev/null | grep -oE '[0-9]+' | awk '{sum+=$1} END {print sum}')
            USER_GPUS=${USER_GPUS:-0}
            
            echo -e "  ${WHITE}┌─ Partition: ${MAGENTA}${BOLD}$part${NC}"
            echo -e "  ${WHITE}│  ${NC}State: $PART_STATE │ Nodes: $PART_NODES │ Cluster GPUs: $PART_GPUS │ Node Memory: $PART_MEM"
            echo -e "  ${WHITE}│  ${CYAN}Your usage: ${NC}${BOLD}$USER_JOBS${NC} jobs │ ${BOLD}${USER_MEM}GB${NC} memory │ ${BOLD}$USER_CPUS${NC} CPUs │ ${BOLD}$USER_GPUS${NC} GPUs"
            echo -e "  ${WHITE}└────────────────────────────────────────────────────────────────${NC}"
            echo ""
        done
    fi
    
    # Show recommended partitions
    echo -e "  ${WHITE}Recommended: ${GREEN}H200-4h${NC} or ${GREEN}H200-12h${NC} (node: hpc8h200-01) for best performance${NC}"
    echo ""
}

print_completed_jobs() {
    echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}${BOLD}                            RECENTLY COMPLETED (last hour)${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════════════${NC}"
    echo ""
    
    # Get recently completed jobs
    COMPLETED=$(sacct -u $USER --starttime=$(date -d '1 hour ago' '+%Y-%m-%dT%H:%M:%S') \
                --format=JobID,JobName%20,Partition%10,State%12,ExitCode,Elapsed \
                --noheader 2>/dev/null | grep -v "\.batch" | grep -v "\.extern" | head -8)
    
    if [ -z "$COMPLETED" ]; then
        echo -e "  ${YELLOW}No recently completed jobs${NC}"
    else
        printf "  ${WHITE}%-12s %-20s %-12s %-12s %-8s %-10s${NC}\n" \
               "JOBID" "NAME" "PARTITION" "STATE" "EXIT" "ELAPSED"
        echo -e "  ${WHITE}─────────────────────────────────────────────────────────────────────────────────────${NC}"
        echo "$COMPLETED" | while read line; do
            if echo "$line" | grep -q "COMPLETED"; then
                echo -e "  ${GREEN}$line${NC}"
            elif echo "$line" | grep -q "FAILED\|OUT_OF"; then
                echo -e "  ${RED}$line${NC}"
            elif echo "$line" | grep -q "CANCELLED"; then
                echo -e "  ${YELLOW}$line${NC}"
            elif echo "$line" | grep -q "RUNNING"; then
                echo -e "  ${CYAN}$line${NC}"
            else
                echo "  $line"
            fi
        done
    fi
    echo ""
}

print_results_status() {
    echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${WHITE}${BOLD}                              EXPERIMENT RESULTS STATUS${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════════════${NC}"
    echo ""
    
    cd "$PROJECT_DIR" 2>/dev/null || return
    
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
                printf "  ${GREEN}✓${NC} %-35s ${GREEN}%d files${NC}\n" "$name" "$file_count"
            else
                printf "  ${YELLOW}○${NC} %-35s ${YELLOW}in progress${NC}\n" "$name"
            fi
        else
            printf "  ${RED}✗${NC} %-35s ${RED}not started${NC}\n" "$name"
        fi
    done
    echo ""
}

print_log_tails() {
    echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${WHITE}${BOLD}                                 RECENT LOG OUTPUT${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════════════${NC}"
    echo ""
    
    cd "$PROJECT_DIR" 2>/dev/null || return
    
    # Find most recently modified log files
    LOGS=$(ls -t logs/slurm_*.out 2>/dev/null | head -3)
    
    if [ -z "$LOGS" ]; then
        echo -e "  ${YELLOW}No Slurm log files found${NC}"
    else
        for log in $LOGS; do
            if [ -f "$log" ]; then
                BASENAME=$(basename "$log" .out)
                echo -e "  ${CYAN}┌─── ${BASENAME} ───${NC}"
                tail -n 2 "$log" 2>/dev/null | sed 's/^/  │ /' || echo "  │ (empty)"
                echo -e "  ${CYAN}└────────────────────────────────────────────────────────${NC}"
                echo ""
            fi
        done
    fi
}

print_help() {
    echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${WHITE}${BOLD}                                  QUICK COMMANDS${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "  ${WHITE}scancel <job_id>${NC}                          Cancel a specific job"
    echo -e "  ${WHITE}scancel -u \$USER${NC}                          Cancel all your jobs"
    echo -e "  ${WHITE}tail -f logs/slurm_<name>_<id>.out${NC}        Follow job output live"
    echo -e "  ${WHITE}scontrol show job <id>${NC}                    Show detailed job info"
    echo ""
    echo -e "  ${YELLOW}Refreshing every ${REFRESH_INTERVAL}s... Press Ctrl+C to exit${NC}"
    echo ""
}

# Main loop
main() {
    while true; do
        clear
        print_header
        print_job_summary
        print_running_jobs
        print_pending_jobs
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
