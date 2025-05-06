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

import numpy as np

def flatten_modules(modules):
	mods = []
	for module in modules:
		if not isinstance(module, nn.Sequential):
			mods.append(module)
		else:
			mods += flatten_modules(module)

	return mods


def get_layer_data_size(modules, start_im_size: tuple):
	'''
	run an image through each of the modules and get the input image size
	for each module
	'''
	modules = flatten_modules(modules)

	start_image = torch.Tensor(np.ones(start_im_size))
	out = start_image

	input_sizes = []
	for module in modules:
		# ensure size fits for linear
		if isinstance(module, nn.Linear):
			out = out.reshape(np.prod(out.shape))


		input_sizes.append(out.shape)
		#print(f'{out.shape}\t->\t{module}\t->', end='\t')
		out = module(out)
		#print(out.shape)

	return input_sizes


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


def predict_conv_layer(conv_model: tuple, batch_size: int, in_size: tuple, layer: torch.nn.Conv2d) -> tuple:
	'''
	predict forward/backward energy consumption & time for one Conv2d layer 
	'''
	lin, degree, poly, scaler, test_err = conv_model

	# in: batch size, in chan, out chan, im side length, k side length, stride
	X = torch.Tensor([batch_size, 
				   layer.in_channels, 
				   layer.out_channels,
				   in_size[-1],
				   layer.kernel_size[-1],
				   layer.stride[-1]
				   ]).reshape(1, -1)

	print(X)

	X_scaled = scaler.transform(X)
	X_poly = poly.transform(X_scaled)
	pred = lin.predict(X_poly)

	return pred


def predict_linear_layer(linear_model: tuple, batch_size: int, layer: torch.nn.Linear) -> tuple:
	'''
	predict forward/backward energy consumption & time for one linear layer 
	'''
	lin, degree, poly, scaler, test_err = linear_model

	# in: batch_size, in_size, out_size
	X = torch.Tensor([batch_size, 
				   layer.in_features, 
				   layer.out_features,
				   ]).reshape(1, -1)
		
	X_scaled = scaler.transform(X)
	X_poly = poly.transform(X_scaled)
	pred = lin.predict(X_poly)

	return pred


def predict_pool_layer(pool_model: tuple, batch_size: int, in_size: tuple, layer: torch.nn.MaxPool2d) -> tuple:
	'''
	predict forward/backward energy consumption & time for one pool layer 
	'''
	lin, degree, poly, scaler, test_err = pool_model

	# in: batch_size, k side length, stride, in_chan, im side length
	X = torch.Tensor([batch_size, 
				   layer.kernel_size[-1] if isinstance(layer.kernel_size, tuple) else layer.kernel_size, 
					layer.stride[-1] if isinstance(layer.stride, tuple) else layer.stride,
				   in_size[0],
				   in_size[-1],
				   ]).reshape(1, -1)
		
	X_scaled = scaler.transform(X)
	X_poly = poly.transform(X_scaled)
	pred = lin.predict(X_poly)

	return pred


def predict_cnn(energy_model: tuple, batch_size: int, input_size: tuple, cnn_model: torch.nn.Module):
	'''
	predict the energy consumption and execution time for forward and
	backward passes of the 'cnn_model' using the three energy models (conv, linear, pool) in 'energy_model'
	'''
	conv, linear, pool = energy_model

	predicted_vals = []
	predicted_layers = []

	layers = flatten_modules(cnn_model.children())
	in_sizes = get_layer_data_size(layers, input_size)

	for layer, in_size in zip(layers, in_sizes):
		#print(f'predicting {layer} with input size {in_size}')

		if isinstance(layer, torch.nn.Conv2d):
			predicted_vals.append(predict_conv_layer(conv, batch_size, in_size, layer))
			predicted_layers.append(layer)

		if isinstance(layer, torch.nn.Linear):
			predicted_vals.append(predict_linear_layer(linear, batch_size, layer))
			predicted_layers.append(layer)

		if isinstance(layer, torch.nn.MaxPool2d):
			predicted_vals.append(predict_pool_layer(pool, batch_size, in_size, layer))
			predicted_layers.append(layer)

	return predicted_vals, predicted_layers


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('models_dir', type=Path, help='path to the directory of the energy prediction models for the target machine')
	parser.add_argument('model_name', type=str, help='name of the CNN model to load from the hub and predict. e.x. "alexnet"')
	parser.add_argument('--hub', type=str, default='pytorch/vision:release/0.10', help='name of the pytorch hub to download the model from')
	parser.add_argument('--batch_size', type=int, default=512, help='the batch size of the model we are predicting')
	parser.add_argument('--input_size', type=tuple, default=(3, 64, 64), help='the size of the input image we are predicting')

	args = parser.parse_args()

	# load energy models
	energy_model = load_energy_models(args.models_dir)

	# load CNN to estimate
	model = torch.hub.load(args.hub, args.model_name)

	vals, layers = predict_cnn(energy_model, args.batch_size, args.input_size, model)
	
	for val, layer in zip(vals, layers):
		print(f'{layer}:\t{val}')

	print(f'Total result: {np.sum(vals)}')
