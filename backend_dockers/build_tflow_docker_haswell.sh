#!/bin/bash
docker build -t seunglab/tensorflow:haswell -f Dockerfile.tensorflow.haswell .
docker push seunglab/tensorflow:haswell
