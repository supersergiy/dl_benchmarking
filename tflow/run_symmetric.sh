#!/bin/sh
mkdir dest
SLP=5
(python3 client.py --model symmetric | tail -1) >> output.txt; sleep $SLP
(python3 client.py --model symmetric --activ elu | tail -1) >> output.txt; sleep $SLP
(python3 client.py --model symmetric --batchnorm | tail -1) >> output.txt; sleep $SLP
(python3 client.py --model symmetric --batchnorm --activ elu | tail -1) >> output.txt; sleep $SLP

(python3 client.py --model symmetric --optimize | tail -1) >> output.txt; sleep $SLP
(python3 client.py --model symmetric --optimize --activ elu | tail -1) >> output.txt; sleep $SLP
(python3 client.py --model symmetric --optimize --batchnorm | tail -1) >> output.txt; sleep $SLP
(python3 client.py --model symmetric --optimize --batchnorm --activ elu| tail -1) >> output.txt; sleep $SLP
