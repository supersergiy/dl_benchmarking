#!/bin/sh
mkdir dest
SLP=5
(python client.py --gpu --model block --batchnorm --activ elu --in_features 8  | tail -1) >> output.txt; sleep $SLP
(python client.py --gpu --model block --batchnorm --activ elu --in_features 16 | tail -1) >> output.txt; sleep $SLP
(python client.py --gpu --model block --batchnorm --activ elu --in_features 24 | tail -1) >> output.txt; sleep $SLP
(python client.py --gpu --model block --batchnorm --activ elu --in_features 32 | tail -1) >> output.txt; sleep $SLP
(python client.py --gpu --model block --batchnorm --activ elu --in_features 64 | tail -1) >> output.txt; sleep $SLP
(python client.py --gpu --model block --batchnorm --activ elu --in_features 96 | tail -1) >> output.txt; sleep $SLP
(python client.py --gpu --model block --batchnorm --activ elu --in_features 128 | tail -1) >> output.txt; sleep $SLP
