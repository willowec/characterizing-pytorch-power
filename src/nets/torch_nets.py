'''
Gather energy data for some legitimate CNNs on CIFAR-10
'''

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision

import argparse
from pathlib import Path

import pyRAPL
import numpy as np
import json, os

HUB_NAME = 'pytorch/vision:release/0.10'
NUM_CLASSES = 10

BATCH_SIZE = 512
IM_SIDE_LENGTH = 64

DEVICE = 'cpu'

verbose = False

def flatten_modules(modules):
	mods = []
	for module in modules:
		if not isinstance(module, nn.Sequential):
			mods.append(module)
		else:
			mods += flatten_modules(module)

	return mods



def train_one_epoch(model, train_loader, optimizer):
	'''
	shamelessly copy pasted from ../gen_dataset.py cause i aint got time to do imports
	'''
	forward_energy = []
	forward_time = []
	backward_energy = []
	backward_time = []

	criterion = nn.CrossEntropyLoss()

	for batch_idx, (data, label) in enumerate(train_loader):
		if verbose:
			print(f'batch {batch_idx}/{len(train_loader)}...')

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
		meter = pyRAPL.Measurement('bar')
		meter.begin()

		loss = criterion(output, label)
		loss.backward()
		optimizer.step()

		meter.end()
		backward_energy.append(meter.result.pkg)
		backward_time.append(meter.result.duration)


	forward_energy = np.asarray(forward_energy)
	forward_time = np.asarray(forward_time)
	backward_energy = np.asarray(backward_energy)
	backward_time = np.asarray(backward_time)

	return np.mean(forward_energy), np.mean(forward_time), np.mean(backward_energy), np.mean(backward_time)



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--n_passes', type=int, default=1, help='number of forward and backward passes to collect for each net')
	parser.add_argument('--nets', type=list, default=['alexnet', 'resnet18', 'mobilenet_v3_small'], help=f'the names of the networks to use (available in {HUB_NAME})')
	parser.add_argument('--image_sizes', type=list, default=[()])
	parser.add_argument('--list_avail', action='store_true', help='lists available models in hub')
	parser.add_argument('--verbose', action='store_true')
	
	args = parser.parse_args()
	verbose = args.verbose
	
	if args.list_avail:
		param_counts = []
		model_names = torch.hub.list(HUB_NAME)
		for model_name in model_names:
			model = torch.hub.load(HUB_NAME, model_name, verbose=False)
			param_counts.append(sum(p.numel() for p in model.parameters()))

		idxs = np.argsort(np.asarray(param_counts))
		for idx in idxs:
			print(f'{model_names[idx]}:\t{param_counts[idx]}')

		quit()

	# make pyrapl work
	pyRAPL.setup()

	# load up the dataset
	trainset = torchvision.datasets.CIFAR10(Path('dataset'), download=True, train=True)

	# transform it comfortably
	transform = torchvision.transforms.Compose([
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Resize(IM_SIDE_LENGTH)
	])
	trainset.transform = transform

	# now create data loader
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
												shuffle=True)

	# and set up the output directory
	out_dir = Path('out')
	out_dir.mkdir(exist_ok=True, parents=True)

	# for each model, see how much it costs!
	for model_name in args.nets:
		print(f'{model_name}...')

		# 1. download the model
		model = torch.hub.load(HUB_NAME, model_name)

		# 2. modify the model to fit CIFAR10 by adding one more linear layer
		layers = flatten_modules(model.children())
		model = nn.Sequential(model, torch.nn.Linear(in_features=layers[-1].out_features, out_features=NUM_CLASSES))

		optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

		# 3. get a bunch of forward and backward passes
		data = {
			'net': model_name,
			'batch_size': BATCH_SIZE,
			'im_side_len': IM_SIDE_LENGTH,
			'forward_energy': [],
			'forward_time': [],
			'backward_energy': [],
			'backward_time': [],
		}
		for i in range(args.n_passes):
			fe, ft, be, bt = train_one_epoch(model, trainloader, optimizer)
			data['forward_energy'].append(fe)
			data['forward_time'].append(ft)
			data['backward_energy'].append(be)
			data['backward_time'].append(bt)

		# 5. write results
		if args.verbose:
			print(f'completed {model_name}!\n{json.dumps(data, indent=4)}')

		out_path = out_dir.joinpath(f'{os.uname().nodename}_{model_name}.json')
		out_path.write_text(json.dumps(data))
		