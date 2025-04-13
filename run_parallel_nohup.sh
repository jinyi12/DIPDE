#!/bin/bash

# Number of threads for each process (3/4 of the total 128)
N_THREADS=96

# Log file to store PIDs and their corresponding output files
PID_LOG="pids.log"
echo "Starting background processes..." > $PID_LOG # Overwrite or create the PID log

# --- Run L2 script ---
LOG_L2="log_L2.out"
echo "Starting ukappa_inverse_problem_L2.sh..."
# Execute in a subshell to set OMP_NUM_THREADS, redirect output, run in background
nohup bash -c 'export OMP_NUM_THREADS=$0; bash ukappa_inverse_problem_L2.sh' "$N_THREADS" > "$LOG_L2" 2>&1 &
pid_L2=$!
echo "L2 process started with PID: $pid_L2. Output logged to $LOG_L2"
echo "PID_L2: $pid_L2, Log: $LOG_L2" >> $PID_LOG

# --- Run TV script ---
LOG_TV="log_TV.out"
echo "Starting ukappa_inverse_problem_TV.sh..."
# Execute in a subshell to set OMP_NUM_THREADS, redirect output, run in background
nohup bash -c 'export OMP_NUM_THREADS=$0; bash ukappa_inverse_problem_TV.sh' "$N_THREADS" > "$LOG_TV" 2>&1 &
pid_TV=$!
echo "TV process started with PID: $pid_TV. Output logged to $LOG_TV"
echo "PID_TV: $pid_TV, Log: $LOG_TV" >> $PID_LOG

# --- Run denoiser script ---
LOG_DENOISER="log_denoiser.out"
echo "Starting ukappa_inverse_problem_denoiser.sh..."
# Execute in a subshell to set OMP_NUM_THREADS, redirect output, run in background
nohup bash -c 'export OMP_NUM_THREADS=$0; bash ukappa_inverse_problem_denoiser.sh' "$N_THREADS" > "$LOG_DENOISER" 2>&1 &
pid_DENOISER=$!
echo "Denoiser process started with PID: $pid_DENOISER. Output logged to $LOG_DENOISER"
echo "PID_DENOISER: $pid_DENOISER, Log: $LOG_DENOISER" >> $PID_LOG

# --- Print summary ---
echo "---"
echo "All scripts launched in the background using nohup."
echo "Check $LOG_L2 and $LOG_TV and $LOG_DENOISER for output."
echo "Check $PID_LOG for process IDs and log file mapping."
echo "You can monitor the processes using 'ps -p $pid_L2,$pid_TV,$pid_DENOISER' or 'htop'."
echo "To terminate a process, use 'kill $pid_L2' or 'kill $pid_TV' or 'kill $pid_DENOISER'."
