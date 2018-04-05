#!/bin/bash
docker pull seunglab/unet_timing:tflow_haswell_optimized
docker run -it --rm seunglab/unet_timing:tflow_haswell_optimized bash
