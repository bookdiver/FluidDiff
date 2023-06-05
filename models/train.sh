#!/bin/bash

# Define the arguments you want to pass to the Python script
args=(
    "--gamma 1.0"
    "--gamma 0.5"
    "--gamma 0.2"
)

# Loop through the arguments and run the Python script with each set of arguments
for arg in "${args[@]}"
do
    echo "Running script with arguments: $arg"
    python3 train.py --experiment burgers --device-no 0 --train-batch-size 8 --test-batch-size 4 --train-lr 0.0002 --train-epochs 300 $arg
    wait
done