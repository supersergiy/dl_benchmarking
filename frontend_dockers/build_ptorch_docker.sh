#!/bin/bash
docker build -t seunglab/unet_timing:ptorch -f Dockerfile.ptorch .
docker push seunglab/unet_timing:ptorch
