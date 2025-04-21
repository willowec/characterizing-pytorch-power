import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader

import argparse
import time

class Net(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(Net,self).__init__()
        
        self.layers = nn.ModuleList([nn.Linear(input_size, output_size)])

    def forward(self,x):
    
        for layer in self.layers:
            x = layer(x)

        return x


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()


def simulate_training(ds, input_size, output_size, e, bs, verbose=False):
    
    device = "cpu"

    model = Net(input_size, output_size)
    
    if verbose:
        for layer in model.children():
            print(layer)

    training_X = torch.rand((ds, input_size))
    training_y = torch.rand((ds, output_size))
    train_dataloader = DataLoader(TensorDataset(training_X, training_y), batch_size=bs, shuffle=True)

    optimizer = optim.Adadelta(model.parameters(), lr=1e-3)

    if verbose:
        print("Training")
    
    start = time.time()
    for epoch in range(e):
        if verbose:
            print(".", end='', flush=True)
        train(model, device, train_dataloader, optimizer, epoch)

    if verbose:
        print(f"\nDone!. Took {time.time()-start:.4f}s.")

    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-ds", "--data_size", type=int, default=100, help="size of the randomzied dataset")
    parser.add_argument("-is", "--input_size", type=int, default=128, help="input size of the linear layer")
    parser.add_argument("-os", "--output_size", type=int, default=128, help="output size of the linear layer")
    parser.add_argument("-bs", "--batch_size", type=int, default=64, help="batch_size")
    parser.add_argument("-e", "--epochs", type=int, default=5, help="number of epochs")
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose")

    args = parser.parse_args()

    params = simulate_training(args.data_size, args.input_size, args.output_size, args.epochs, args.batch_size, verbose=args.verbose)
    print(params) # print params to pass back to trainer
