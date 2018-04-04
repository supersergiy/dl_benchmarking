#!/bin/bash
docker pull seunglab/unet_timing:tflow_haswell
docker run -it --rm --runtime=nvidia  seunglab/unet_timing:tflow_haswell bash
