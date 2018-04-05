#!/bin/sh
mkdir dest
SLP=5
(python client.py --model residual | tail -1) >> output.txt; sleep $SLP
(python client.py --model residual --activ elu | tail -1) >> output.txt; sleep $SLP
(python client.py --model residual --batchnorm | tail -1) >> output.txt; sleep $SLP
(python client.py --model residual --batchnorm --activ elu §| tail -1) >> output.txt; sleep $SLP

(python client.py --model residual --optimize | tail -1) >> output.txt; sleep $SLP
(python client.py --model residual --optimize --activ elu | tail -1) >> output.txt; sleep $SLP
(python client.py --model residual --optimize --batchnorm | tail -1) >> output.txt; sleep $SLP
(python client.py --model residual --optimize --batchnorm --activ elu| tail -1) >> output.txt; sleep $SLP
