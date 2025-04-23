# Related Works

## To Read

- [MIT: Deep Neural Network Energy Estimation Tool](https://energyestimation.mit.edu/)
- [A Method to Estimate the Energy Consumption of Deep Neural Networks](https://eems.mit.edu/wp-content/uploads/2017/12/2017_asilomar_tool.pdf)
- [Measuring the Energy Consumption and Efficiency of Deep Neural Networks: An Empirical Analysis and Design Recommendations](https://arxiv.org/html/2403.08151v1)
- [Estimating energy consumption of neural networks with joint Structure–Device encoding](https://www.sciencedirect.com/science/article/pii/S2210537924001070)
- [Energy Consumption of Neural Networks on NVIDIA Edge Boards: an Empirical Model](https://arxiv.org/pdf/2210.01625)
- [Profiling Energy Consumption of Deep Neural Networks on NVIDIA Jetson Nano](https://ieeexplore.ieee.org/document/9290876)

## Have Read

### [Estimation of energy consumption in machine learning](https://www.sciencedirect.com/science/article/pii/S0743731518308773)

Review of different approaches to energy estimation, focusing on CPU-based and DRAM-based energy estimations

Approaches to estimate system power (we use RAPL)
 - Performance counters using regression or correlation techniques on PMCs

Section 6 talks about energy estimation in machine learning
 - MACC counting is one way (discussed in class)
 - NeuralPower uses regression to model energy consumption of max pool, conv, and fully connected, estimates energy per layer

### [NeuralPower: Predict and Deploy Energy-Efficient Convolutional Neural Networks](https://arxiv.org/abs/1710.05420)
 - Designed to predict energy consumption of forward phase of a CNN before even training it
 - Splits problem into predicting energy consumption of FC, Conv, and MaxPool layers
 - Propose a "learning-based polynomial regression model"
 - By layer:
   - Conv: model based on batch size, the input tensor size, the kernel size, the stride size, the padding size, and the output tensor size
   - Fully connected: model based on batch size, the input tensor size, and the output tensor siz
   - Pooling: model based on  input tensor size, the stride size, the kernel size, and the output tensor size
 - Claim to acquire training data in 10 minutes and train the NeuralPower model in under 20 on an NVIDIA TITAN
 - "We separate the training data points into groups based on their layer types. In this paper, the training data include 858 convolution layer samples, 216 pooling layer samples, and 116 fully connected layer samples. The statistics can change if one needs any form of customization"