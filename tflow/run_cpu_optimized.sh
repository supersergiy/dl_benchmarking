SLP=5

(python3 client.py --name opt/1 --optimize | tail -1) >> output.txt; sleep $SLP
(python3 client.py --name opt/2 --optimize --activ elu | tail -1) >> output.txt; sleep $SLP
(python3 client.py --name opt/3 --optimize --batchnorm | tail -1) >> output.txt; sleep $SLP
(python3 client.py --name opt/4 --optimize --batchnorm --activ elu| tail -1) >> output.txt; sleep $SLP

(python3 client.py --name opt/5 --model symmetric --optimize | tail -1) >> output.txt; sleep $SLP
(python3 client.py --name opt/6 --model symmetric --optimize --activ elu | tail -1) >> output.txt; sleep $SLP
(python3 client.py --name opt/7 --model symmetric --optimize --batchnorm | tail -1) >> output.txt; sleep $SLP
(python3 client.py --name opt/8 --model symmetric --optimize --batchnorm --activ elu| tail -1) >> output.txt; sleep $SLP
