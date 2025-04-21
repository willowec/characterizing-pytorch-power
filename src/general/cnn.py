import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader

import argparse
import time

class Net(nn.Module):
    def __init__(self, layers, channels, k_size, stride):
        super(Net,self).__init__()
        
        self.layers = nn.ModuleList([nn.Conv2d(channels, channels, k_size, stride=stride, padding='same')])
        if layers > 1:
            for i in range(1, layers-1):
                self.layers.append(nn.Conv2d(channels, channels, k_size, stride=stride, padding='same'))

        self.layers.append(nn.Conv2d(channels, channels, k_size, stride=stride, padding='same'))


    def forward(self,x):
    
        out = self.layers[0](x)
        if len(self.layers) > 0:
            for layer in self.layers[1:-1]:
                out = layer(out)

        out = self.layers[-1](out)

        # get to size 1 for label
        out = torch.mean(out, (1, 2, 3))

        return out


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), torch.squeeze(target.to(device))
        optimizer.zero_grad()
        output = torch.squeeze(model(data))

        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()


def simulate_training(epochs, ds, x_size, y_size, n_layers, channels, k_size, stride, bs, verbose=False):
    
    device = "cpu"

    model = Net(channels, channels, k_size, stride)
    
    if verbose:
        for layer in model.children():
            print(layer)

    imsize = (channels, x_size, y_size)
    training_X = torch.rand((ds, *imsize))
    training_y = torch.rand((ds, 1))
    train_dataloader = DataLoader(TensorDataset(training_X, training_y), batch_size=bs, shuffle=True)

    optimizer = optim.Adadelta(model.parameters(), lr=1e-3)

    if verbose:
        print("Training")
    
    start = time.time()
    for epoch in range(epochs):
        if verbose:
            print(".", end='', flush=True)
        train(model, device, train_dataloader, optimizer, epoch)

    if verbose:
        print(f"\nDone!. Took {time.time()-start:.4f}s.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-ds", "--data_size", type=int, default=100, help="size of the randomzied dataset")
    parser.add_argument("-nl", "--n_layers", type=int, default=4, help="number of hidden layers")
    parser.add_argument("-bs", "--batch_size", type=int, default=64, help="batch_size")
    parser.add_argument("-e", "--epochs", type=int, default=5, help="number of epochs")
    parser.add_argument("-cs", "--channels", type=int, default=3, help="number of channels")
    parser.add_argument("-ks", "--k_size", type=int, default=3, help="kernel size")
    parser.add_argument("-st", "--stride", type=int, default=1, help="stride")
    parser.add_argument("-sl", "--side_length", type=int, default=32, help="image side length")
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose")

    args = parser.parse_args()

    simulate_training(args.epochs, args.data_size, 
                      args.side_length, args.side_length, args.n_layers, 
                      args.channels, args.k_size, args.stride, 
                      args.batch_size, verbose=args.verbose)
