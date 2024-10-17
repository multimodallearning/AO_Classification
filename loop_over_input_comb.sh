#!/bin/bash

# Loop over all the input combinations
DIR="./configs/input_comb"

for FILE in $DIR/*; do
  echo "Using input combination: $FILE"
  python -m train_classifier fit --trainer configs/base_trainer.yaml --model $FILE
done