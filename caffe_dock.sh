#!/bin/bash
sudo docker run -v ~/dl_benchmarking/caffe:/workspace -v ~/tests:/tests -it bvlc/caffe:cpu /bin/bash 
