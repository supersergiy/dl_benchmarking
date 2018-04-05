#!/bin/sh
mkdir dest
SLP=5
(python client.py | tail -1) >> output.txt; sleep $SLP
(python client.py --activ elu | tail -1) >> output.txt; sleep $SLP
(python client.py --batchnorm | tail -1) >> output.txt; sleep $SLP
(python client.py --batchnorm --activ elu | tail -1) >> output.txt; sleep $SLP

(python client.py --optimize | tail -1) >> output.txt; sleep $SLP
(python client.py --optimize --activ elu | tail -1) >> output.txt; sleep $SLP
(python client.py --optimize --batchnorm | tail -1) >> output.txt; sleep $SLP
(python client.py --optimize --batchnorm --activ elu| tail -1) >> output.txt; sleep $SLP
