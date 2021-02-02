#!/usr/bin/python
#encoding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
import pdb

class BatchDHCN(nn.Module):
    """docstring for BatchDHCN"""
    def __init__(self, embed_size=512, output_size=512 , num_channel=2, conv_size=3,batch_norm=True):
        super(BatchDHCN, self).__init__()

        self.batch_norm = batch_norm
        self.embed_size = embed_size
        self.output_size = output_size
        self.num_channel = num_channel

        self.padding = nn.ZeroPad2d((0, conv_size - 1, conv_size - 1, 0))
        self.conv_1 = nn.Conv2d(self.num_channel, self.output_size,(conv_size,conv_size))
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):

        # x = self.dropout(x)
        x_conv_1 = self.conv_1(self.padding(x))
        x_conv_1 = F.relu(x_conv_1)
        
        return x_conv_1

class DHCN(nn.Module):
    def __init__(self, input_size = 200, embed_size=200, densenet_layer = 2, output_size=16 + 1, conv_size=3, batch_norm = False):
        super(DHCN, self).__init__()

        self.densenet_layer = densenet_layer
        self.batch_norm = batch_norm
        self.input_size = input_size
        self.embed_size = embed_size
        self.output_size = output_size

        if self.densenet_layer > 0:
            DHCNs = []
            ConvLinears = []
            for k_layer in range(self.densenet_layer):
                tcn = BatchDHCN(embed_size = self.embed_size, output_size = self.embed_size, num_channel = self.embed_size * (k_layer + 1 ), conv_size=3, batch_norm = self.batch_norm)
                DHCNs.append((str(k_layer),tcn))
                ConvLinears.append((str(k_layer), nn.Conv2d(self.embed_size, self.output_size,(conv_size,conv_size))))
            ConvLinears.append((str(self.densenet_layer), nn.Conv2d(self.embed_size, self.output_size,(conv_size,conv_size))))

            self.DHCNs = nn.Sequential(OrderedDict(DHCNs))
            self.ConvLinears = nn.Sequential(OrderedDict(ConvLinears))

        self.padding = nn.ZeroPad2d((0, conv_size - 1, conv_size - 1, 0))
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):  

        scores = []
        prediction = self.ConvLinears[-1](self.dropout(self.padding(x)))
        # prediction = self.ConvLinears[-1](self.padding(x))
        scores.append(prediction)

        for k_layer in range(self.densenet_layer):

            if k_layer > 0 :
                x = torch.cat((x,x_conv_1),1)
            x_conv_1 = self.DHCNs[k_layer](x)
            prediction = self.ConvLinears[k_layer](self.dropout(self.padding(x_conv_1)))
            # prediction = self.ConvLinears[k_layer](self.padding(x_conv_1))

            scores.append(prediction)

        return scores, []