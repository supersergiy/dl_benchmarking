#!/bin/bash
docker pull seunglab/unet_timing:ptorch
docker run -it --rm --runtime=nvidia  seunglab/unet_timing:ptorch bash
