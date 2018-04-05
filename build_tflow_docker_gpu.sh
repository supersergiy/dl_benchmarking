#!/bin/bash
docker build -t seunglab/unet_timing:tflow -f Dockerfile.tflow.gpu .
docker push seunglab/unet_timing:tflow
