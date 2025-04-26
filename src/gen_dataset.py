'''
Generate the dataset for Conv2d, Linear, and MaxPool layers
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import argparse
import csv, os, datetime
from pathlib import Path

import pyRAPL


# data size is batch_size * BATCH_MULTIPLIER
BATCH_MULTIPLIER=5
DEVICE = 'cpu'

# conv vals: batch size, in chan, out chan, im side length, k side length, stride
CONV_MIN = torch.Tensor([32, 1, 1, 8, 3, 1])
CONV_MAX = torch.Tensor([2048, 16, 64, 64, 9, 4])

# linear vals: batch_size, in_size, out_size
LINEAR_MIN = torch.Tensor([32, 1, 1])
LINEAR_MAX = torch.Tensor([2048, 512, 512])

# pool vals: batch_size, k side length, stride, in_chan, im side length
POOL_MIN = torch.Tensor([32, 3, 1, 3, 8])
POOL_MAX = torch.Tensor([2048, 15, 8, 64, 64])


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
        data, label = data.to(DEVICE), label.to(DEVICE)
        
        if optimizer is not None:
            optimizer.zero_grad()

        # stat energy of forward pass
        meter = pyRAPL.Measurement('bar')
        meter.begin()

        output = model(data)

        meter.end()
        forward_energy.append(meter.result.pkg)
        forward_time.append(meter.result.duration)

        # stat energy of backward pass
        if optimizer is not None:
            meter = pyRAPL.Measurement('bar')
            meter.begin()

            loss = F.mse_loss(output, label)
            loss.backward()
            optimizer.step()

            meter.end()
            backward_energy.append(meter.result.pkg)
            backward_time.append(meter.result.duration)
        else:
            backward_energy.append(0)
            backward_time.append(0)

    forward_energy = np.asarray(forward_energy)
    forward_time = np.asarray(forward_time)
    backward_energy = np.asarray(backward_energy)
    backward_time = np.asarray(backward_time)

    return np.mean(forward_energy), np.mean(forward_time), np.mean(backward_energy), np.mean(backward_time)


def gather_epoch(net, batch_size, in_size, out_size):
    '''
    generates random data for a net, returns forward/backward energy/time
    '''
    data_size = batch_size*BATCH_MULTIPLIER

    train_loader = DataLoader(
            TensorDataset(torch.rand(data_size, *in_size), 
                          torch.rand(data_size, *out_size)), 
            batch_size=batch_size, shuffle=True
    )

    # handle layers with no backwards pass, like pool
    if len(list(net.parameters())) > 0:
        optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    else:
        optimizer = None

    return train_one_epoch(net, train_loader, optimizer)


def gather_conv(batch_size, in_chan, out_chan, side_len, k_size, stride):
    in_size = (in_chan, side_len, side_len)
    out_size = (out_chan, # calculated based on eqns at https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
                int(((side_len - (k_size-1) - 1) / stride) + 1),
                int(((side_len - (k_size-1) - 1) / stride) + 1)
                )

    net = ConvLayer(in_chan, out_chan, k_size, stride)
    return gather_epoch(net, batch_size, in_size, out_size)


def gather_linear(batch_size, in_size, out_size):
    net = LinearLayer(in_size, out_size)
    return gather_epoch(net, batch_size, [in_size], [out_size])


def gather_pool(batch_size, k_size, stride, chan, side_len):
    net = PoolLayer(k_size, stride)

    in_size = (chan, side_len, side_len)
    out_size = (
        int((side_len - (k_size-1) - 1) / stride + 1),
        int((side_len - (k_size-1) - 1) / stride + 1),
    )

    return gather_epoch(net, batch_size, in_size, out_size)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-nc', '--n_conv', default=10, type=int, help='number of data points to generate for conv layer')
    parser.add_argument('-np', '--n_pool', default=10, type=int, help='number of data points to generate for pooling layer')
    parser.add_argument('-nl', '--n_linear', default=10, type=int, help='number of data points to generate for linear layer')
    parser.add_argument('-na', '--n_avg', default=10, type=int, help='number of iteratios to average over per data point')

    args = parser.parse_args()

    pyRAPL.setup()

    out_dir = Path(f"out/{os.uname().nodename}-{datetime.datetime.now().isoformat().replace(':', '-').split('.')[0]}")
    out_dir.mkdir(exist_ok=True, parents=True)

    with open(out_dir.joinpath('conv.csv'), 'w+') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['batch_size', 'in_chan', 'out_chan', 'side_len', 'k_size', 'stride', 'for_energy', 'for_time', 'back_energy', 'back_time'])

        # conv loop
        for i in range(args.n_conv):
            # generate the conv args
            conv_args = None
            while conv_args is None:
                conv_args = ((torch.rand(6) * (CONV_MAX - CONV_MIN)) + CONV_MIN).type(torch.int32)

                if conv_args[4] > conv_args[3]:
                    conv_args = None # regen if k size > im size

            print(f'conv {i}, args={conv_args.tolist()}:\t', end='', flush=True)    

            # repeatedly run and average
            res = [[gather_conv(*conv_args.tolist())] for j in range(args.n_avg)]
            means = np.mean(res, axis=0).tolist()[0]

            # store results
            print(f'{means[0]/1_000_000:.4f}J, {means[1]/1_000_000:.4f}s, {means[2]/1_000_000:.4f}J, {means[3]/1_000_000:.4f}s')
            writer.writerow(conv_args.tolist() + means) 


    with open(out_dir.joinpath('linear.csv'), 'w+') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['batch_size', 'in_size', 'out_size', 'for_energy', 'for_time', 'back_energy', 'back_time'])

        # linear loop
        for i in range(args.n_linear):
            # generate the linear args
            linear_args = ((torch.rand(3) * (LINEAR_MAX - LINEAR_MIN)) + LINEAR_MIN).type(torch.int32)

            print(f'linear {i}, args={linear_args.tolist()}:\t', end='', flush=True)    

            # repeatedly run and average
            res = [[gather_linear(*linear_args.tolist())] for j in range(args.n_avg)]
            means = np.mean(res, axis=0).tolist()[0]

            # store results
            print(f'{means[0]/1_000_000:.4f}J, {means[1]/1_000_000:.4f}s, {means[2]/1_000_000:.4f}J, {means[3]/1_000_000:.4f}s')
            writer.writerow(linear_args.tolist() + means) 


    with open(out_dir.joinpath('pool.csv'), 'w+') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['batch_size', 'in_size', 'out_size', 'for_energy', 'for_time', 'back_energy', 'back_time'])

        # pool loop
        for i in range(args.n_pool):
            # generate the pool args
            pool_args = ((torch.rand(len(POOL_MAX)) * (POOL_MAX - POOL_MIN)) + POOL_MIN).type(torch.int32)

            print(f'pool {i}, args={pool_args.tolist()}:\t', end='', flush=True)    

            # repeatedly run and average
            res = [[gather_pool(*pool_args.tolist())] for j in range(args.n_avg)]
            means = np.mean(res, axis=0).tolist()[0]

            # store results
            print(f'{means[0]/1_000_000:.4f}J, {means[1]/1_000_000:.4f}s, {means[2]/1_000_000:.4f}J, {means[3]/1_000_000:.4f}s')
            writer.writerow(pool_args.tolist() + means) 
