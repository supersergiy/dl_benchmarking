#!/bin/bash
docker build -t seunglab/unet_timing:tflow_haswell_optimized -f Dockerfile.tflow.haswell_optimized .
docker push seunglab/unet_timing:tflow_haswell_optimized
