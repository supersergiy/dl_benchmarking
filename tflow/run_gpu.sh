#!/bin/sh
mkdir dest
SLP=5
(python client.py --gpu | tail -1) >> output.txt; sleep $SLP
(python client.py --gpu --activ elu | tail -1) >> output.txt; sleep $SLP
(python client.py --gpu --batchnorm | tail -1) >> output.txt; sleep $SLP
(python client.py --gpu --batchnorm --activ elu | tail -1) >> output.txt; sleep $SLP

(python client.py --gpu --optimize | tail -1) >> output.txt; sleep $SLP
(python client.py --gpu --optimize --activ elu | tail -1) >> output.txt; sleep $SLP
(python client.py --gpu --optimize --batchnorm | tail -1) >> output.txt; sleep $SLP
(python client.py --gpu --optimize --batchnorm --activ elu| tail -1) >> output.txt; sleep $SLP

(python client.py --gpu --model symmetric | tail -1) >> output.txt; sleep $SLP
(python client.py --gpu --model symmetric --activ elu | tail -1) >> output.txt; sleep $SLP
(python client.py --gpu --model symmetric --batchnorm | tail -1) >> output.txt; sleep $SLP
(python client.py --gpu --model symmetric --batchnorm --activ elu | tail -1) >> output.txt; sleep $SLP

(python client.py --gpu --model symmetric --optimize | tail -1) >> output.txt; sleep $SLP
(python client.py --gpu --model symmetric --optimize --activ elu | tail -1) >> output.txt; sleep $SLP
(python client.py --gpu --model symmetric --optimize --batchnorm | tail -1) >> output.txt; sleep $SLP
(python client.py --gpu --model symmetric --optimize --batchnorm --activ elu| tail -1) >> output.txt; sleep $SLP

(python client.py --gpu --model residual | tail -1) >> output.txt; sleep $SLP
(python client.py --gpu --model residual --activ elu | tail -1) >> output.txt; sleep $SLP
(python client.py --gpu --model residual --batchnorm | tail -1) >> output.txt; sleep $SLP
(python client.py --gpu --model residual --batchnorm --activ elu | tail -1) >> output.txt; sleep $SLP

(python client.py --gpu --model residual --optimize | tail -1) >> output.txt; sleep $SLP
(python client.py --gpu --model residual --optimize --activ elu | tail -1) >> output.txt; sleep $SLP
(python client.py --gpu --model residual --optimize --batchnorm | tail -1) >> output.txt; sleep $SLP
(python client.py --gpu --model residual --optimize --batchnorm --activ elu| tail -1) >> output.txt; sleep $SLP
