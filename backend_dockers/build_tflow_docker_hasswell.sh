#!/bin/bash
docker build -t seunglab/tensorflow:hasswell -f Dockerfile.tensorflow.hasswell .
docker push seunglab/tensorflow:hasswell
