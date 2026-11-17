#!/usr/bin/env bash

export PROJECT="/home/vishnu/eswar/drone_telemetry"

# OpenMVS
export OPENMVS_BIN="/home/vishnu/eswar/apps/openmvs/bin/OpenMVS"
export PATH="$OPENMVS_BIN:$PATH"
export LD_LIBRARY_PATH="$OPENMVS_BIN:${LD_LIBRARY_PATH:-}"

echo "PROJECT=$PROJECT"
echo "OPENMVS_BIN=$OPENMVS_BIN"
