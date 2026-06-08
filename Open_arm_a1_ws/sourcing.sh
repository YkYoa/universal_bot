#!/bin/bash
# Sourcing this script sets up the CycloneDDS environment for the laptop.
# Usage: source sourcing.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="file://${SCRIPT_DIR}/cyclonedds.xml"

echo "[OpenArm] Sourced CycloneDDS configuration:"
echo "  ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
echo "  RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION"
echo "  CYCLONEDDS_URI=$CYCLONEDDS_URI"
