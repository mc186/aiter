#!/bin/bash

# Simple benchmark runner that generates JSON files
# Usage: ./run_benchmarks.sh [output_dir] [parallel_jobs]

OUTPUT_DIR=${1:-"json_outputs"}
PARALLEL_JOBS=${2:-1}

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Define your configs here
declare -a CONFIGS=(
    # Format: "batch,hq,hk,sq,sk,d_head,layout,extra_flags,output_name"
    #"1,32,32,2048,2048,128,bshd,,config_1"
    "1,8,8,128,128,128,bshd,,config_test_1_8_128"
    "1,8,8,8192,8192,128,bshd,,config_1_8_8192"
    "2,8,8,8192,8192,128,bshd,,config_2_8_8192"
)

# Function to run a single benchmark
run_benchmark() {
    local config=$1
    local output_dir=$2
    
    IFS=',' read -r batch hq hk sq sk d_head layout extra_flags output_name <<< "$config"
    
    local json_file="${output_dir}/${output_name}.json"
    
    echo "Running: $output_name"
    
    # Build the command
    local cmd="omniprobe -a MessageLogger -v -i -c ~/.triton/cache -t json -l $json_file -d 1 -- python single_test_mha.py"
    cmd="$cmd -b $batch -hq $hq -hk $hk -sq $sq -sk $sk -d $d_head -layout $layout"
    
    # Add extra flags if any
    if [[ -n "$extra_flags" ]]; then
        cmd="$cmd $extra_flags"
    fi
    
    echo "Command: $cmd"
    
    # Execute and capture output
    if output=$(eval "$cmd" 2>&1); then
        echo "✓ Completed: $output_name"
        echo "Output:"
        echo "$output"
    else
        echo "✗ Failed: $output_name"
        echo "Error output:"
        echo "$output"
    fi
    echo "----------------------------------------"
}

export -f run_benchmark

echo "Running ${#CONFIGS[@]} configurations with $PARALLEL_JOBS parallel jobs"
echo "Output directory: $OUTPUT_DIR"

# Run benchmarks
if [[ $PARALLEL_JOBS -eq 1 ]]; then
    # Sequential
    for config in "${CONFIGS[@]}"; do
        run_benchmark "$config" "$OUTPUT_DIR"
    done
else
    # Parallel using GNU parallel or xargs
    if command -v parallel > /dev/null; then
        printf '%s\n' "${CONFIGS[@]}" | parallel -j "$PARALLEL_JOBS" run_benchmark {} "$OUTPUT_DIR"
    else
        printf '%s\n' "${CONFIGS[@]}" | xargs -I {} -P "$PARALLEL_JOBS" bash -c "run_benchmark '{}' '$OUTPUT_DIR'"
    fi
fi

echo "All benchmarks completed. JSON files saved in $OUTPUT_DIR/"  
