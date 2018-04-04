#!/bin/bash
docker pull seunglab/unet_timing:tflow_skylake
docker run -it --rm --runtime=nvidia  seunglab/unet_timing:tflow_skylake bash
