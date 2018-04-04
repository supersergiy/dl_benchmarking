#!/bin/bash
docker build -t seunglab/unet_timing:tflow_haswell -f Dockerfile.tflow.haswell .
docker push seunglab/unet_timing:tflow_haswell
