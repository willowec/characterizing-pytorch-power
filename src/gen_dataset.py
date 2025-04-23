'''
Generate the dataset for Conv2d, Linear, and MaxPool layers
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import numpy as np

# suppress pyRAPL warnings?
import warnings
warnings.filterwarnings("ignore", category=Warning)
import pyRAPL


class ConvLayer(nn.Module):
    def __init__(self, in_chan, out_chan, k_size, stride):
        super(ConvLayer, self).__init__()
        self.layer = nn.Conv2d(in_chan, out_chan, k_size, stride=stride)


    def forward(self, x):
        return self.layer(x)


class LinearLayer(nn.Module):
    def __init__(self, in_size, out_size):
        super(LinearLayer, self).__init__()
        self.layer = nn.Linear(in_size, out_size)


    def forward(self, x):
        return self.layer(x)


class PoolLayer(nn.Module):
    def __init__(self, k_size, stride):
        super(PoolLayer, self).__init__()
        self.layer = nn.MaxPool2d(k_size, stride=stride)


    def forward(self, x):
        return self.layer(x)
    

def train_one_epoch(model, train_loader, optimizer):
    '''
    simulates training a the model and returns per-batch forward energy/time and backward energy/time
    '''


    forward_energy = []
    forward_time = []
    backward_energy = []
    backward_time = []

    for batch_idx, (data, label) in enumerate(train_loader):
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()

        # stat energy of forward pass
        meter = pyRAPL.Measurement('bar')
        meter.begin()

        output = model(data)

        meter.end()
        forward_energy.append(meter.result.pkg)
        forward_time.append(meter.result.duration)

        # stat energy of backward pass
        meter = pyRAPL.Measurement('bar')
        meter.begin()

        loss = F.mse_loss(output, label)
        loss.backward()
        optimizer.step()

        meter.end()
        backward_energy.append(meter.result.pkg)
        backward_time.append(meter.result.duration)

    return np.mean(forward_energy), np.mean(forward_time), np.mean(backward_energy), np.mean(backward_time)


if __name__ == "__main__":

    pyRAPL.setup()

    device = 'cpu'

    net = ConvLayer(10, 10, 3, 2)

    in_size = (10, 32, 32)
    out_size = (10, 15, 15) # out chan, in_size/stride - (k_size-1)/2
    data_size = 1024
    batch_size = 512

    train_loader = DataLoader(
            TensorDataset(torch.rand(data_size, *in_size), 
                          torch.rand(data_size, *out_size)), 
            batch_size=batch_size, shuffle=True
    )    
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    print(train_one_epoch(net, train_loader, optimizer))

