#!/bin/bash
docker pull seunglab/unet_timing:tflow_hasswell
docker run -it --rm --runtime=nvidia  seunglab/unet_timing:tflow_hasswell bash
