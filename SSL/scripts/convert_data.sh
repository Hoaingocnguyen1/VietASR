#!/bin/bash

DATASET_NAME="doof-ferb/vlsp2020_vinai_100h"
OUTPUT_DIR="download"
MODE="streaming"  # sample, full, streaming
SPLITS="train test dev" 
MAX_SAMPLES=""

python convert_dataset.py \
    --mode $MODE \
    --dataset $DATASET_NAME \
    --output $OUTPUT_DIR \
    --splits $SPLITS \
    $MAX_SAMPLES