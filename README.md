# LinearNetworkDynamics

## Overview

The repository contains the code and materials for my Master Thesis titled "Learning Dynamics of Linear Neural Networks". This research aims to better understand how weight initialisation and network structure influence learning properties in neural networks. To do so, we study $\lambda$-Balanced linear networks. The study provides exact solutions for aligned and unaligned $\lambda$-Balanced networks and explores their applications to continual learning. 

## Contents

- `continual_learning_test.py`: Contains experiments related to continual learning.
- `diagonalisation_check.py`: Includes scripts to check diagonalization properties of matrices in the context of learning dynamics.
- `illustrative_example.ipynb`: Jupyter notebook demonstrating illustrative examples of the concepts discussed in the thesis.
- `linear_network.py`: Implementation of a linear neural network.
- `qqt_dynamics_figure.ipynb`: Jupyter notebook for generating figures related to QQT dynamics.
- `qqt_lambda_balanced.py`: Module for QQT lambda balanced calculations. Please note that some lines are commented out/not used because this code is a work in progress for the unequal input output case. 
- `qqt_test.py`: Test scripts for QQT dynamics.
- `relu_network.py`: Implementation of a ReLU-based neural network.
- `tanh_network.py`: Implementation of a Tanh-based neural network.
- `tools.py`: Utility functions and classes used across various scripts.

## Requirements

To install the necessary dependencies, run:

```bash
pip install -r requirements.txt
