#!/bin/bash

# Exit the script immediately if any command fails
set -e

# Define the experiment setups you want Hydra to sweep through for each code
EXPERIMENT_SETUPS="'7_7','14_14','60_30','60_60','182_182','365_182'"
EXCEL_PATH="../data/FINAL_DB/target_codes_qualud.json"

echo "🔍 Fetching target codes using your Python function..."

# 1. Load codes safely, filtering out TensorFlow/GPU log noise using a prefix
mapfile -t CODES_ARRAY < <(python3 -c "
import sys
import os

try:
    from src.utils.experiments_utils import load_codes
    
    codes = load_codes('$EXCEL_PATH')
    
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
    echo "💡 Double-check if '$EXCEL_PATH' exists relative to where you are running this script."
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
    
    # Run Hydra multirun (-m) for the current code across all experiment setups
    python3 main_train_quantization.py -m \
        model.target_code="$CODE" \
        experiment_setup="$EXPERIMENT_SETUPS"
        
    echo "========================================================================"
    echo "✅ PROCESS TERMINATED: Memory fully reclaimed by OS for code: $CODE"
    echo "------------------------------------------------------------------------"
done

echo "🎉 All ${#CODES_ARRAY[@]} codes executed successfully with zero memory accumulation!"