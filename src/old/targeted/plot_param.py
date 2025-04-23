# plot a param

import matplotlib.pyplot as plt
import numpy as np
import json, argparse
from pathlib import Path


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('file', type=Path, help='path to the json file you want to plot')

	args = parser.parse_args()

	layer = args.file.name.split('.')[0].split('-')[0]
	param = args.file.name.split('.')[0].split('-')[1]
	data = json.loads(args.file.read_text())
	
	f, ax = plt.subplots(1, 2, figsize=(12, 5))
	# plot energy
	ax[0].scatter(data['param_vals'], data['energies'], color='orange')
	ax[0].set_xlabel(param)
	ax[0].set_ylabel('energy (J)')
	# add line of best fit
	coeffs = np.polyfit(data['param_vals'], data['energies'], 1)
	poly = np.poly1d(coeffs)
	ax[0].plot(data['param_vals'], poly(data['param_vals']), color='red')
	print(coeffs)

	ax[1].scatter(data['param_vals'], data['model_n_params'])
	ax[1].set_xlabel(param)
	ax[1].set_ylabel('model parameter count')

	# build title
	title = ', '.join([f'{k.replace("-", "")}={v["fixed"]}' for k, v in data['fixed'].items()])
	title = f'{layer} sweeping {param}:\n' + title
	f.suptitle(title)

	plt.tight_layout()
	plt.show()