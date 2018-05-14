#!/bin/sh
mkdir dest
SLP=5
(python client.py --gpu --model block --batchnorm --activ elu --in_features 8  --batch_size 32 | tail -1) >> output.txt; sleep $SLP
(python client.py --gpu --model block --batchnorm --activ elu --in_features 16 --batch_size 32 | tail -1) >> output.txt; sleep $SLP
(python client.py --gpu --model block --batchnorm --activ elu --in_features 24 --batch_size 32 | tail -1) >> output.txt; sleep $SLP
(python client.py --gpu --model block --batchnorm --activ elu --in_features 32 --batch_size 32 | tail -1) >> output.txt; sleep $SLP
(python client.py --gpu --model block --batchnorm --activ elu --in_features 40 --batch_size 32 | tail -1) >> output.txt; sleep $SLP
(python client.py --gpu --model block --batchnorm --activ elu --in_features 48 --batch_size 32 | tail -1) >> output.txt; sleep $SLP
(python client.py --gpu --model block --batchnorm --activ elu --in_features 56 --batch_size 32 | tail -1) >> output.txt; sleep $SLP
(python client.py --gpu --model block --batchnorm --activ elu --in_features 64 --batch_size 32 | tail -1) >> output.txt; sleep $SLP
