#!/bin/bash

# Directory containing our test scripts
TEST_DIR="./test_simulations/"
# File to store the results
RESULTS_FILE="runtests_summary.log"
# Temporary file to store unsorted results
TEMP_FILE="temp_results.log"
# Clear previous results
> "$TEMP_FILE"
> "$RESULTS_FILE"
> "runtests_errors.log"

# remove all previous results of the test_cases
rm -r ./simulation_results/runtests/*

# Loop over each test case in the directory
cnt=0
for test_script in "$TEST_DIR"/*.yaml; do
  # Set a unique output log file for each test script
  output_log="output_$cnt.log"

  echo "Running simulation $cnt with calibration $test_script..."
  
  # Execute the test, redirect output to a unique file, and save the exit code
  (
    python "run_simulation.py" "test_$cnt" -f "$test_script" > "$output_log" 2>&1
    exit_code=$?

    # Append test result to the temporary log file
    if [ $exit_code -eq 0 ]; then
      echo "Run #$cnt: $test_script PASSED" | tee -a "$TEMP_FILE"
    else
      echo "Run #$cnt: $test_script FAILED" | tee -a "$TEMP_FILE"
      echo "Run #$cnt: $test_script FAILED" >> "runtests_errors.log"
      echo "Output:" >> "runtests_errors.log"
      cat "$output_log" >> "runtests_errors.log"
      echo -e "\n---\n" >> "runtests_errors.log"
    fi
  ) &
  

  # Increment the counter
  ((cnt++))
done

wait

# Sort the results summary by the test number and save to RESULTS_FILE
sort -n -k2,2 "$TEMP_FILE" > "$RESULTS_FILE"

# Display a summary
echo "Test Results Summary:"
cat "$RESULTS_FILE"

# Clean up: Delete all individual output log files and the temporary file
rm output_*.log "$TEMP_FILE"