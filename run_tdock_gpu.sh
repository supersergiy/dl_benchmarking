#!/bin/bash
docker pull seunglab/unet_timing:tflow
docker run -it --rm --runtime=nvidia  seunglab/unet_timing:tflow bash
