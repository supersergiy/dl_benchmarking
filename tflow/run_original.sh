#!/bin/sh
mkdir dest
SLP=5
(python3 client.py | tail -1) >> output.txt; sleep $SLP
(python3 client.py --activ elu | tail -1) >> output.txt; sleep $SLP
(python3 client.py --batchnorm | tail -1) >> output.txt; sleep $SLP
(python3 client.py --batchnorm --activ elu | tail -1) >> output.txt; sleep $SLP

(python3 client.py --optimize | tail -1) >> output.txt; sleep $SLP
(python3 client.py --optimize --activ elu | tail -1) >> output.txt; sleep $SLP
(python3 client.py --optimize --batchnorm | tail -1) >> output.txt; sleep $SLP
(python3 client.py --optimize --batchnorm --activ elu| tail -1) >> output.txt; sleep $SLP
