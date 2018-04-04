mkdir dest
(python client.py --gpu --box_size 108 | tail -1) >> output.txt; sleep 5
(python client.py --gpu --box_size 108 --activ elu | tail -1) >> output.txt; sleep 5
(python client.py --gpu --box_size 108 --batchnorm | tail -1) >> output.txt; sleep 5
(python client.py --gpu --box_size 108 --batchnorm | tail -1) >> output.txt; sleep 5

(python client.py --gpu --box_size 108 --optimize | tail -1) >> output.txt; sleep 5
(python client.py --gpu --box_size 108 --optimize --activ elu | tail -1) >> output.txt; sleep 5
(python client.py --gpu --box_size 108 --optimize --batchnorm | tail -1) >> output.txt; sleep 5
(python client.py --gpu --box_size 108 --optimize --batchnorm --activ elu| tail -1) >> output.txt; sleep 5

(python client.py --gpu --sym_unet | tail -1) >> output.txt; sleep 5
(python client.py --gpu --sym_unet --activ elu | tail -1) >> output.txt; sleep 5
(python client.py --gpu --sym_unet --batchnorm | tail -1) >> output.txt; sleep 5
(python client.py --gpu --sym_unet --batchnorm | tail -1) >> output.txt; sleep 5

(python client.py --gpu --sym_unet --optimize | tail -1) >> output.txt; sleep 5
(python client.py --gpu --sym_unet --optimize --activ elu | tail -1) >> output.txt; sleep 5
(python client.py --gpu --sym_unet --optimize --batchnorm | tail -1) >> output.txt; sleep 5
(python client.py --gpu --sym_unet --optimize --batchnorm --activ elu| tail -1) >> output.txt; sleep 5
