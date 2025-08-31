#!/bin/bash

# Hardware Speedup Simulator Evaluation Script
# This script runs the RTGS hardware speedup evaluation

# Configuration - Change this path as needed
POINT_CLOUD_FILE="/home/aeuser/MonoRTGS_ONX/hardware_speedup_simulator/point_cloud.json"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Starting RTGS Hardware Speedup Evaluation...${NC}"
echo "================================================================"

# Check if point cloud file exists
if [ ! -f "$POINT_CLOUD_FILE" ]; then
    echo -e "${RED}❌ Error: Point cloud file not found at: $POINT_CLOUD_FILE${NC}"
    echo "Please update the POINT_CLOUD_FILE variable in this script with the correct path."
    exit 1
fi

echo -e "${GREEN}✅ Found point cloud file: $POINT_CLOUD_FILE${NC}"

# Activate Python virtual environment
echo -e "${YELLOW}🔧 Activating Python virtual environment...${NC}"
source ~/venvs/jp6torch/bin/activate

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Failed to activate virtual environment!${NC}"
    echo "Please check if the virtual environment exists at ~/venvs/jp6torch/bin/activate"
    exit 1
fi

echo -e "${GREEN}✅ Virtual environment activated successfully${NC}"

# Change to hardware_speedup_simulator directory
echo -e "${YELLOW}📁 Changing to hardware_speedup_simulator directory...${NC}"
cd hardware_speedup_simulator

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Failed to change directory!${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Changed to hardware_speedup_simulator directory${NC}"

# Run the evaluation
echo -e "${YELLOW}🚀 Running evaluation with point cloud file...${NC}"
echo "================================================================"

python eval.py "$POINT_CLOUD_FILE"

# Check if evaluation was successful
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Evaluation completed successfully!${NC}"
else
    echo -e "${RED}❌ Evaluation failed!${NC}"
    exit 1
fi

echo "================================================================"
echo -e "${GREEN}🎯 Hardware evaluation script finished!${NC}"
