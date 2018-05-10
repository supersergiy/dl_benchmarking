#!/bin/sh
mkdir dest
SLP=5
(python client.py --gpu --model block --optimize --batchnorm --activ elu --features 8  | tail -1) >> output.txt; sleep $SLP
(python client.py --gpu --model block --optimize --batchnorm --activ elu --features 16 | tail -1) >> output.txt; sleep $SLP
(python client.py --gpu --model block --optimize --batchnorm --activ elu --features 24 | tail -1) >> output.txt; sleep $SLP
(python client.py --gpu --model block --optimize --batchnorm --activ elu --features 32 | tail -1) >> output.txt; sleep $SLP
(python client.py --gpu --model block --optimize --batchnorm --activ elu --features 64 | tail -1) >> output.txt; sleep $SLP
(python client.py --gpu --model block --optimize --batchnorm --activ elu --features 96 | tail -1) >> output.txt; sleep $SLP
(python client.py --gpu --model block --optimize --batchnorm --activ elu --features 128 | tail -1) >> output.txt; sleep $SLP
