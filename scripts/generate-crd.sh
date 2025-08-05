#!/bin/bash

CRD_DIR="./charts/tensor-fusion/crds"
OUTPUT_FILE="./tmp.tensor-fusion-crds.yaml"

echo "Generating combined CRD file..."
> "$OUTPUT_FILE"
for file in "$CRD_DIR"/*.yaml; do
    [ -s "$OUTPUT_FILE" ]
    cat "$file" >> "$OUTPUT_FILE"
done
echo "Generated: $OUTPUT_FILE"
