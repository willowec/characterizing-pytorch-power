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


def load_model(model_path: Path) -> tuple:
	'''
	loads a single model at path
	'''
	with open(model_path, 'rb') as handle:
		model = pickle.load(handle)

	return model


def load_models(models_dir: Path) -> tuple[tuple, tuple, tuple]:
	'''
	loads the conv, linear, and pool models from the given directory
	'''
	conv = None
	linear = None
	pool = None

	for path in models_dir.iterdir():
		if 'conv' in path.name: 
			conv = load_model(path)
		elif 'linear' in path.name:
			linear = load_model(path)
		elif 'pool' in path.name:
			pool = load_model(path)

	return conv, linear, pool


def predict_cnn(model: tuple, cnn_model: torch.nn.Module):
	'''
	predict the energy consumption and execution time for forward and
	backward passes of the 'cnn_model' using the linear 'model'
	'''


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('models_dir', type=Path, help='path to the directory of models for the target machine')
	args = parser.parse_args()

	conv, linear, pool = load_models(args.models_dir)
	print(conv, linear, pool)
