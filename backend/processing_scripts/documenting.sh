#!/bin/bash

OUTPUT_MD=$1

echo "# Транскрипция" > "$OUTPUT_MD"
echo "" >> "$OUTPUT_MD"

for file in $(ls clip-*.txt | sort); do
    speaker=$(basename "$file" | awk -F "-" '{print($3)}' | awk -F "." '{print $1}')
    
    echo -n "$speaker: " >> "$OUTPUT_MD"

    cat "$file" | tr -s ' ' | tr '\n' ' '  >> "$OUTPUT_MD"
    echo "" >> "$OUTPUT_MD"
    echo "" >> "$OUTPUT_MD"
done
