#!/bin/bash
docker pull seunglab/unet_timing
docker run -it --rm --runtime=nvidia  seunglab/unet_timing bash
