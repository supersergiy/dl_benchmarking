#!/bin/sh
mkdir dest
SLP=5
(python client.py --gpu --model block --batchnorm --activ elu --in_features 8  --batch_size 32 | tail -1) >> output.txt; sleep $SLP
(python client.py --gpu --model block --batchnorm --activ elu --in_features 16 --batch_size 32 | tail -1) >> output.txt; sleep $SLP
(python client.py --gpu --model block --batchnorm --activ elu --in_features 24 --batch_size 32 | tail -1) >> output.txt; sleep $SLP
(python client.py --gpu --model block --batchnorm --activ elu --in_features 32 --batch_size 32 | tail -1) >> output.txt; sleep $SLP
(python client.py --gpu --model block --batchnorm --activ elu --in_features 64 --batch_size 32 | tail -1) >> output.txt; sleep $SLP
(python client.py --gpu --model block --batchnorm --activ elu --in_features 96 --batch_size 8 | tail -1) >> output.txt; sleep $SLP
(python client.py --gpu --model block --batchnorm --activ elu --in_features 128 --batch_size 8 | tail -1) >> output.txt; sleep $SLP
