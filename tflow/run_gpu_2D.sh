#!/bin/sh
mkdir dest
SLP=5
(python client.py --gpu --dim 2 --model symmetric --batchnorm --activ elu --features 8  --batch_size 128 | tail -1) >> output.txt; sleep $SLP
(python client.py --gpu --dim 2 --model symmetric --batchnorm --activ elu --features 16 --batch_size 128 | tail -1) >> output.txt; sleep $SLP
(python client.py --gpu --dim 2 --model symmetric --batchnorm --activ elu --features 24 --batch_size 128 | tail -1) >> output.txt; sleep $SLP
(python client.py --gpu --dim 2 --model symmetric --batchnorm --activ elu --features 32 --batch_size 128 | tail -1) >> output.txt; sleep $SLP
(python client.py --gpu --dim 2 --model symmetric --batchnorm --activ elu --features 64 --batch_size 128 | tail -1) >> output.txt; sleep $SLP
(python client.py --gpu --dim 2 --model symmetric --batchnorm --activ elu --features 96 --batch_size 128 | tail -1) >> output.txt; sleep $SLP
(python client.py --gpu --dim 2 --model symmetric --batchnorm --activ elu --features 128 --batch_size 128 | tail -1) >> output.txt; sleep $SLP
