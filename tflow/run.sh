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

(python client.py --model symmetric | tail -1) >> output.txt; sleep $SLP
(python client.py --model symmetric --activ elu | tail -1) >> output.txt; sleep $SLP
(python client.py --model symmetric --batchnorm | tail -1) >> output.txt; sleep $SLP
(python client.py --model symmetric --batchnorm --activ elu | tail -1) >> output.txt; sleep $SLP

(python client.py --model symmetric --optimize | tail -1) >> output.txt; sleep $SLP
(python client.py --model symmetric --optimize --activ elu | tail -1) >> output.txt; sleep $SLP
(python client.py --model symmetric --optimize --batchnorm | tail -1) >> output.txt; sleep $SLP
(python client.py --model symmetric --optimize --batchnorm --activ elu| tail -1) >> output.txt; sleep $SLP

(python client.py --model residual | tail -1) >> output.txt; sleep $SLP
(python client.py --model residual --activ elu | tail -1) >> output.txt; sleep $SLP
(python client.py --model residual --batchnorm | tail -1) >> output.txt; sleep $SLP
(python client.py --model residual --batchnorm --activ elu §| tail -1) >> output.txt; sleep $SLP

(python client.py --model residual --optimize | tail -1) >> output.txt; sleep $SLP
(python client.py --model residual --optimize --activ elu | tail -1) >> output.txt; sleep $SLP
(python client.py --model residual --optimize --batchnorm | tail -1) >> output.txt; sleep $SLP
(python client.py --model residual --optimize --batchnorm --activ elu| tail -1) >> output.txt; sleep $SLP
