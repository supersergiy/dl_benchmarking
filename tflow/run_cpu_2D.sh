#!/bin/sh
mkdir dest
SLP=5
(python3 client.py --dim 2 --model symmetric --batchnorm --activ elu --features 8  --batch_size 2 | tail -1) >> output.txt; sleep $SLP
(python3 client.py --dim 2 --model symmetric --batchnorm --activ elu --features 16 --batch_size 2 | tail -1) >> output.txt; sleep $SLP
(python3 client.py --dim 2 --model symmetric --batchnorm --activ elu --features 24 --batch_size 2 | tail -1) >> output.txt; sleep $SLP
(python3 client.py --dim 2 --model symmetric --batchnorm --activ elu --features 32 --batch_size 2 | tail -1) >> output.txt; sleep $SLP
(python3 client.py --dim 2 --model symmetric --batchnorm --activ elu --features 40 --batch_size 2 | tail -1) >> output.txt; sleep $SLP
(python3 client.py --dim 2 --model symmetric --batchnorm --activ elu --features 48 --batch_size 2 | tail -1) >> output.txt; sleep $SLP
(python3 client.py --dim 2 --model symmetric --batchnorm --activ elu --features 56 --batch_size 2 | tail -1) >> output.txt; sleep $SLP
(python3 client.py --dim 2 --model symmetric --batchnorm --activ elu --features 64 --batch_size 2 | tail -1) >> output.txt; sleep $SLP
