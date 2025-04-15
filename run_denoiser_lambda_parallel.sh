#!/bin/bash

# Number of threads for each process
N_THREADS=64

# Log file to store PIDs and their corresponding output files
PID_LOG="denoiser_lambda_pids.log"
echo "Starting denoiser processes with different lambda values..." > $PID_LOG

# --- Run denoiser with lambda_reg=1e-4 ---
LOG_DENOISER_1E4="log_denoiser_1e-4.out"
echo "Starting ukappa_inverse_problem_denoiser_1e-4.sh..."
nohup bash -c 'export OMP_NUM_THREADS=$0; bash ukappa_inverse_problem_denoiser_1e-4.sh' "$N_THREADS" > "$LOG_DENOISER_1E4" 2>&1 &
pid_DENOISER_1E4=$!
echo "Denoiser (lambda=1e-4) process started with PID: $pid_DENOISER_1E4. Output logged to $LOG_DENOISER_1E4"
echo "PID_DENOISER_1E4: $pid_DENOISER_1E4, Log: $LOG_DENOISER_1E4" >> $PID_LOG

# --- Run denoiser with lambda_reg=1e-6 ---
LOG_DENOISER_1E6="log_denoiser_1e-6.out"
echo "Starting ukappa_inverse_problem_denoiser_1e-6.sh..."
nohup bash -c 'export OMP_NUM_THREADS=$0; bash ukappa_inverse_problem_denoiser_1e-6.sh' "$N_THREADS" > "$LOG_DENOISER_1E6" 2>&1 &
pid_DENOISER_1E6=$!
echo "Denoiser (lambda=1e-6) process started with PID: $pid_DENOISER_1E6. Output logged to $LOG_DENOISER_1E6"
echo "PID_DENOISER_1E6: $pid_DENOISER_1E6, Log: $LOG_DENOISER_1E6" >> $PID_LOG

# --- Print summary ---
echo "---"
echo "All denoiser scripts launched in the background using nohup."
echo "Check $LOG_DENOISER_1E4 and $LOG_DENOISER_1E6 for output."
echo "Check $PID_LOG for process IDs and log file mapping."
echo "You can monitor the processes using 'ps -p $pid_DENOISER_1E4,$pid_DENOISER_1E6' or 'htop'."
echo "To terminate processes, use 'kill $pid_DENOISER_1E4' or 'kill $pid_DENOISER_1E6'." 