#!/bin/bash

# Exit the script immediately if any initialization command fails
set -e

# Define the experiment setups you want Hydra to sweep through for each code
EXPERIMENT_SETUPS="'7_7','14_14','60_30','60_60','182_182','365_182'"
MODELS_PATH="../quantized_models"

echo "🔍 Fetching target codes using your Python function..."

# 1. Load codes safely, filtering out TensorFlow/GPU log noise using a prefix
mapfile -t CODES_ARRAY < <(python3 -c "
import sys
import os

try:
    from src.utils.experiments_utils import load_inference_codes_list
    
    codes = load_inference_codes_list('$MODELS_PATH')
    
    # Diagnostic check: alert user if the function returned nothing
    if not codes:
        print('⚠️  Python warning: load_codes() returned an empty list!', file=sys.stderr)
    
    # CRITICAL: Prefix valid codes so Bash can filter out background log noise
    for code in codes:
        print(f'VALID_CODE:{code}')
        
except Exception as e:
    print(f'❌ Error executing Python function: {e}', file=sys.stderr)
    sys.exit(1)
" | grep "^VALID_CODE:" | sed 's/^VALID_CODE://')

# Verify we actually loaded codes
if [ ${#CODES_ARRAY[@]} -eq 0 ]; then
    echo "❌ Error: No target codes retrieved from Python function."
    echo "💡 Double-check if '$MODELS_PATH' exists relative to where you are running this script."
    exit 1
fi

echo "🚀 Loaded ${#CODES_ARRAY[@]} codes. Starting sequential process isolation pipeline..."
echo "------------------------------------------------------------------------"

# 2. Iterate through each code sequentially
for CODE in "${CODES_ARRAY[@]}"; do
    echo "========================================================================"
    echo "⏱️  STARTING NEW ISOLATED PROCESS"
    echo "🎯 Target Code: $CODE"
    echo "========================================================================"
    
    # Wrapping in an 'if' statement prevents 'set -e' from exiting on failure.
    # It catches the return status of the Python process dynamically.
    if python3 main_inference.py -m \
        model.target_code="$CODE" \
        experiment_setup="$EXPERIMENT_SETUPS"; then
        
        echo "========================================================================"
        echo "✅ PROCESS TERMINATED: Memory fully reclaimed by OS for code: $CODE"
        echo "------------------------------------------------------------------------"
    else
        echo "========================================================================"
        echo "❌ ERROR: Execution failed for code: $CODE"
        echo "⚠️  Skipping to the next code in the sequence..."
        echo "------------------------------------------------------------------------"
        # Optional: Log the failed code to a file for later tracking
        echo "$CODE" >> failed_codes.log
        continue
    fi
done

echo "🎉 Isolated process loop finished processing all ${#CODES_ARRAY[@]} codes!"