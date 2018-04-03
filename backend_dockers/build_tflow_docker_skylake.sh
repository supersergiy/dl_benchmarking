#!/bin/bash
docker build -t seunglab/tensorflow:skylake -f Dockerfile.tensorflow.skylake .
docker push seunglab/tensorflow:skylake
