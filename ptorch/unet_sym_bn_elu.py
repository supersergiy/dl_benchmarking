from __future__ import print_function
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.backends.cudnn.benchmark = True

import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
from time import time 
import sys

class Identity(nn.Module):
    def forward(self, x):
        return x

class Unet(nn.Module):
    def __init__(self, merge = True, #Otherwise sum
                       symmetric = True,
                       activation = nn.ELU,
                       kernel_size = 3,
                       batchnorm = True,
                       upsample = True,
                       kernel_shape =  [[3,32],
                                        [32,64],
                                        [64,128],
                                        [128,256]]):
        super(Unet, self).__init__()
        self.blocks = []
        self.merge = merge
        kshp = kernel_shape
        self.levels = len(kernel_shape)

        if merge:
            factor = 3
        else:
            factor = 1

        if symmetric:
            padding = int(kernel_size/2)
        else:
            padding = 0

        print('~~~~ Down ~~~~~')
        self.downsamples = nn.ModuleList()
        for i in range(self.levels-1):
            block = self.block( inp = kshp[i][0],
                                mid = int(kshp[i][1]/2),
                                out = kshp[i][1],
                                padding = padding,
                                activation = activation,
                                kernel_size = kernel_size,
                                batchnorm = batchnorm)
            self.downsamples.append(block)

        self.mid = self.block(inp = kernel_shape[-1][0],
                              mid = kernel_shape[-1][0],
                              out = kernel_shape[-1][1],
                              padding = padding,
                              activation = activation,
                              kernel_size = kernel_size,
                              batchnorm = batchnorm)


        print('~~~~ Up ~~~~~')
        self.upsamples = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        for i in range(1, self.levels):
            j = self.levels-i
            self.upconvs.append(self.upconv(inp = kernel_shape[j][1],
                                            out = min(factor,2)*kernel_shape[j][0],
                                            upsample = upsample,
                                            activation = activation,
                                            kernel_size = kernel_size))

            self.upsamples.append(self.block(inp = factor*kernel_shape[j][0],
                                             mid = kernel_shape[j][0],
                                             out = kernel_shape[j][0],
                                             padding=padding,
                                             activation=activation,
                                             kernel_size=kernel_size,
                                             batchnorm = batchnorm))

        self.final = nn.Conv3d(kernel_shape[0][1], kernel_shape[0][0],
                               kernel_size=1, padding=0)

        #for m in self.modules():
        #    if isinstance(m, nn.Conv2d):
        #        torch.nn.init.xavier_normal(m.weight)
        #self.final.weight.data *= eps
        #self.final.bias.data *= eps
        self.pool = nn.MaxPool3d(2)

    def upconv(self, inp, out,kernel_size=3, stride=2, padding=1,
                output_padding=1,upsample=False, activation=nn.ReLU):

        if upsample:
            print('Upsample', inp, out)
            layer = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv3d(inp, out, 1, padding=0))
        else:
            print('ConvTranpose', inp, out)
            layer = nn.Sequential(
                nn.ConvTranspose3d(inp, out, 2, stride=2,
                                   padding=padding, output_padding=output_padding))
        return layer

    def block(self, inp, mid, out, kernel_size=3, padding=1,
                up=False, activation=nn.ReLU, batchnorm=False):
        print('Conv', inp, mid), print('Conv', mid, out)
        batch = lambda x: Identity()
        if batchnorm:
            print('BatchNorm3d')
            batch = nn.BatchNorm3d
        return nn.Sequential(
                nn.Conv3d(inp, mid, kernel_size=kernel_size, padding=padding),
                batch(mid),
                activation(),
                nn.Conv3d(mid, out, kernel_size=kernel_size, padding=padding),
                batch(out),
                activation(),
                )

    def forward(self, x):
        layers = [x]

        for i in range(self.levels-1):
            x = self.downsamples[i](x)
            layers.append(x)
            x = self.pool(x)

        x = self.mid(x)
        layers.append(x)

        for i in range(1, self.levels):
            x = self.upconvs[i-1](x)
            x_enc = layers[(self.levels-1)-(i-1)]

            start = int((x_enc.shape[2]-x.shape[2])/2)
            if self.merge:
                x = torch.cat((x_enc[:,:,start:start+x.shape[2],
                                         start:start+x.shape[2],
                                         start:start+x.shape[2]], x),dim=1)
            else:
                x = x+x_enc[:,:,start:start+x.shape[2],
                                start:start+x.shape[2],
                                start:start+x.shape[2]]
            x = self.upsamples[i-1](x)
        x = self.final(x)

        return x


#Simple test
if __name__ == "__main__":
    torch.set_num_threads(2)
    if sys.argv[1] == "cpu":
       cpu_mode = True
    else:
       cpu_mode = False
     
    if cpu_mode:
	    unet = Unet()
    else:
	    unet = Unet().cuda()
    s = np.ones((1,3,128,128,128), dtype=np.float32)
    if cpu_mode:
	    x = torch.from_numpy(s)
    else:
	    x = torch.from_numpy(s).cuda()
    xs = torch.autograd.Variable(x, requires_grad=False)
    durations = []
    num_iterations = 10
    num_warmups = 5
    for i in range(num_iterations + num_warmups):
	if not cpu_mode:
		torch.cuda.synchronize()
        t1 = time()
        unet(xs)
	if not cpu_mode:
		torch.cuda.synchronize()
        t2 = time()
        if i >= num_warmups:
           durations.append(t2 - t1) 
    average = sum(durations) / len(durations)
    print ("bn elu: {} sec".format(average))
    
