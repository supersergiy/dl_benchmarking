#!/usr/bin/python
import caffe
import h5py
import numpy as np
import sys
import os
from time import time

caffe.set_mode_cpu()

test_folder  = sys.argv[1]
model_path   = os.path.join(test_folder, "net.prototxt")
weights_path = os.path.join(test_folder, "weights.h5")
in_path      = os.path.join(test_folder, "in.h5")

net = caffe.Net(model_path, 1, weights=weights_path)
in_data = h5py.File(inp_path)["main"][:]
net.blobs["input"].data[:] = in_data

w = 5
t = 10

for i in range(w + t):
	s = time()
	net.forward()
	e = time()
	r = e - c
	print (r)
	if (i > w):
		l.append(r)

average = sum(l) / len(l)
print ("average: {}".format(average))	
