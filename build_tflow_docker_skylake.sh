#!/bin/bash
docker build -t seunglab/unet_timing:tflow_skylake -f Dockerfile.tflow.skylake .
docker push seunglab/unet_timing:tflow_skylake
