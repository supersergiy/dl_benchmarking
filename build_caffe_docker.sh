#!/bin/bash
docker build -t seunglab/unet_timing:caffe -f Dockerfile.caffe .
docker push seunglab/unet_timing:caffe
