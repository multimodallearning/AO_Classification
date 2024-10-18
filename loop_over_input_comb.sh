#!/bin/bash

DEVICE=$1
FOLD=$2
# Loop over all the input combinations
DIR="./configs/input_comb"

for FILE in $DIR/*; do
  echo "Using input combination: $FILE"
  python -m train_classifier fit --trainer configs/base_trainer.yaml --trainer.devices [$DEVICE] --model $FILE --data.fold $FOLD
done