import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader

import argparse
import time

class Net(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: list, num_classes: int):
        super(Net,self).__init__()
        
        self.layers = nn.ModuleList([nn.Linear(input_size, hidden_sizes[0])])
        self.layers.append(nn.ReLU())
        if len(hidden_sizes) > 1:
            for i in range(1, len(hidden_sizes)-1):
                self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
                self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(hidden_sizes[-1], num_classes))


    def forward(self,x):
    
        out = self.layers[0](x)
        if len(self.layers) > 0:
            for layer in self.layers[1:-1]:
                out = layer(out)

        out = self.layers[-1](out)

        return out


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()


def simulate_training(ds, nl, hs, e, bs, print_layers=False):
    
    device = "cpu"
    input_size=32

    model = Net(input_size, [hs for i in range(nl)], 1)
    
    if print_layers:
        for layer in model.children():
            print(layer)

    training_X = torch.rand((ds, input_size))
    training_y = torch.rand((ds, 1))
    train_dataloader = DataLoader(TensorDataset(training_X, training_y), batch_size=bs, shuffle=True)

    optimizer = optim.Adadelta(model.parameters(), lr=1e-3)

    print("Training")
    start = time.time()
    for epoch in range(e):
        print(".", end='', flush=True)
        train(model, device, train_dataloader, optimizer, epoch)

    print(f"\nDone!. Took {time.time()-start:.4f}s.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-ds", "--data_size", type=int, default=100, help="size of the randomzied dataset")
    parser.add_argument("-hs", "--hidden_size", type=int, default=128, help="size of the hidden layer")
    parser.add_argument("-nl", "--n_layers", type=int, default=4, help="number of hidden layers")
    parser.add_argument("-bs", "--batch_size", type=int, default=64, help="batch_size")
    parser.add_argument("-e", "--epochs", type=int, default=5, help="number of epochs")
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose")

    args = parser.parse_args()

    simulate_training(args.data_size, args.n_layers, args.hidden_size, args.epochs, args.batch_size, print_layers=args.verbose)
