#!/bin/bash

# Directory containing your test scripts
TEST_DIR="./test_simulations/"
# File to store the results
RESULTS_FILE="runtests.log"
# Clear previous results
> "$RESULTS_FILE"
# remove all previous results for test_cases
rm -r ./simulation_results/runtests/*

# Loop over each test script in the directory
cnt=0
for test_script in "$TEST_DIR"/*.yaml; do
  echo "Running simulation $cnt with calibration $test_script..."
  
  # Execute the test and capture the output and exit code
  python "run_simulation.py" "test_$cnt" -f "$test_script" > output.log 2>&1
  exit_code=$?

  # Append test result to the log file with a success or failure message
  if [ $exit_code -eq 0 ]; then
    echo "Run $cnt: $test_script PASSED" | tee -a "$RESULTS_FILE"
  else
    echo "Run $cnt: $test_script FAILED" | tee -a "$RESULTS_FILE"
    echo "Output:" >> "$RESULTS_FILE"
    cat output.log >> "$RESULTS_FILE"
    echo -e "\n---\n" >> "$RESULTS_FILE"
  fi

  # Increment the counter
  ((cnt++))
done

# Optional: Display a summary
echo "Test Results Summary:"
cat "$RESULTS_FILE"

# remove ouput.log
rm output.log