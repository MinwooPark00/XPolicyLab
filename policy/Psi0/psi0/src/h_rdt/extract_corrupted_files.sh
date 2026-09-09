#!/bin/bash

# Script to extract corrupted HDF5 files from zip archives
# Reads file paths from corrupted_hdf5_files.bak.txt and extracts them

CORRUPTED_LIST="$(dirname "$0")/corrupted_hdf5_files.txt"
ZIP_DIR="/hfm/data/egodex"  # Directory containing zip files (e.g., part1.zip, part2.zip)
OUTPUT_DIR="/hfm/data/egodex"  # Output directory (same as data root)

if [ ! -f "$CORRUPTED_LIST" ]; then
    echo "Error: Corrupted files list not found: $CORRUPTED_LIST"
    exit 1
fi

total_files=$(wc -l < "$CORRUPTED_LIST")
current=0
success=0
failed=0

echo "Starting extraction of $total_files corrupted files..."
echo "ZIP directory: $ZIP_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "=============================================="

while IFS= read -r filepath || [ -n "$filepath" ]; do
    ((current++))
    
    # Skip empty lines
    [ -z "$filepath" ] && continue
    
    # Extract part name and relative path
    # e.g., /hfm/data/egodex/part2/basic_fold/4081.hdf5 -> part2, part2/basic_fold/4081.hdf5
    # Get the part (part1, part2, etc.) from the path
    part_name=$(echo "$filepath" | grep -oP 'part\d+|extra|test')
    
    if [ -z "$part_name" ]; then
        echo "[$current/$total_files] WARNING: Cannot determine part from: $filepath"
        ((failed++))
        continue
    fi
    
    # Build the relative path within the zip (e.g., part2/basic_fold/4081.hdf5)
    relative_path=$(echo "$filepath" | grep -oP '(part\d+|extra|test)/.*')
    
    # Build zip file path
    # For part1, part2, part3, use /home/songlin; otherwise use /hfm/data/egodex
    if [[ "$part_name" =~ ^part[123]$ ]]; then
        zip_file="/home/songlin/${part_name}.zip"
    else
        continue
        # zip_file="${ZIP_DIR}/${part_name}.zip"
    fi
    
    if [ ! -f "$zip_file" ]; then
        echo "[$current/$total_files] WARNING: Zip file not found: $zip_file"
        ((failed++))
        continue
    fi
    
    # Build output directory (e.g., /hfm/data/egodex/part2/basic_fold)
    output_subdir=$(dirname "$filepath")
    
    # Create output directory if not exists
    mkdir -p "$output_subdir"
    
    # Extract the file
    echo "[$current/$total_files] Extracting: $relative_path"
    if unzip -o -j "$zip_file" "$relative_path" -d "$output_subdir" > /dev/null 2>&1; then
        ((success++))
    else
        echo "[$current/$total_files] ERROR: Failed to extract $relative_path from $zip_file"
        ((failed++))
    fi
    
done < "$CORRUPTED_LIST"

echo "=============================================="
echo "Extraction completed!"
echo "  Success: $success"
echo "  Failed: $failed"
echo "  Total: $total_files"
