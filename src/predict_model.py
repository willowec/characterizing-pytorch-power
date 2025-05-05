'''
script for predicting the energy/time for a given cnn using a generated model
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import argparse
import pickle

from pathlib import Path

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, LassoCV, Lasso
from sklearn.metrics import root_mean_squared_error


def flatten_modules(modules):
	mods = []
	for module in modules:
		if not isinstance(module, nn.Sequential):
			mods.append(module)
		else:
			mods += flatten_modules(module)

	return mods


def load_energy_model(model_path: Path) -> tuple:
	'''
	loads a single energy model at path
	'''
	with open(model_path, 'rb') as handle:
		model = pickle.load(handle)

	return model


def load_energy_models(models_dir: Path) -> tuple[tuple, tuple, tuple]:
	'''
	loads the conv, linear, and pool models from the given directory
	'''
	conv = None
	linear = None
	pool = None

	for path in models_dir.iterdir():
		if 'conv' in path.name: 
			conv = load_energy_model(path)
		elif 'linear' in path.name:
			linear = load_energy_model(path)
		elif 'pool' in path.name:
			pool = load_energy_model(path)

	return conv, linear, pool


def predict_conv_layer(conv_model: tuple, layer: torch.nn.Module) -> tuple:
	'''
	predict forward/backward energy consumption & time for one Conv2d layer 
	'''
	# in: batch size, in chan, out chan, im side length, k side length, stride


def predict_linear_layer(conv_model: tuple, layer: torch.nn.Module) -> tuple:
	'''
	predict forward/backward energy consumption & time for one linear layer 
	'''
	# in: batch_size, in_size, out_size


def predict_pool_layer(conv_model: tuple, layer: torch.nn.Module) -> tuple:
	'''
	predict forward/backward energy consumption & time for one pool layer 
	'''
	# in: batch_size, k side length, stride, in_chan, im side length


def predict_cnn(energy_model: tuple, cnn_model: torch.nn.Module):
	'''
	predict the energy consumption and execution time for forward and
	backward passes of the 'cnn_model' using the three energy models (conv, linear, pool) in 'energy_model'
	'''
	conv, linear, pool = energy_model

	layers = flatten_modules(cnn_model.children())
	for layer in layers:
		if isinstance(layer, torch.nn.Conv2d):
			print('conv')
		if isinstance(layer, torch.nn.Linear):
			print('linear')
		if isinstance(layer, torch.nn.MaxPool2d):
			print('pool')



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('models_dir', type=Path, help='path to the directory of the energy prediction models for the target machine')
	parser.add_argument('model_name', type=str, help='name of the CNN model to load from the hub and predict. e.x. "alexnet"')
	parser.add_argument('--hub', type=str, default='pytorch/vision:release/0.10', help='name of the pytorch hub to download the model from')
	args = parser.parse_args()

	# load energy models
	energy_model = load_energy_models(args.models_dir)

	# load CNN to estimate
	model = torch.hub.load(args.hub, args.model_name)

	prediction = predict_cnn(energy_model, model)
	print(prediction)
