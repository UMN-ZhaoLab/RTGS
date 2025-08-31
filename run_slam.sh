#!/bin/bash

# Check if correct number of arguments are provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <version> <config_file>"
    echo "  version: 'baseline' or 'monortgs'"
    echo "  config_file: path to the config file"
    echo ""
    echo "Examples:"
    echo "  $0 baseline configs/rgbd/tum/fr1_desk.yaml"
    echo "  $0 monortgs configs/rgbd/tum/fr1_desk.yaml"
    exit 1
fi

VERSION=$1
CONFIG_FILE=$2

# Convert version to lowercase for case-insensitive comparison
VERSION_LOWER=$(echo "$VERSION" | tr '[:upper:]' '[:lower:]')

# Check which version to run and execute the corresponding command
if [ "$VERSION_LOWER" = "baseline" ]; then
    echo "Running Baseline version with config: $CONFIG_FILE"
    cd Baseline
    python slam.py --config "$CONFIG_FILE" --eval
elif [ "$VERSION_LOWER" = "monortgs" ]; then
    echo "Running MonoRTGS version with config: $CONFIG_FILE"
    cd MonoRTGS
    python slam.py --config "$CONFIG_FILE" --eval
else
    echo "Error: Invalid version '$VERSION'"
    echo "Please use 'baseline' or 'monortgs'"
    exit 1
fi
