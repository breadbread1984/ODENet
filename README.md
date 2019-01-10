# ODENet
This project implements a simple demo of [ODENet](https://arxiv.org/abs/1806.07366) with tensorflow

## Introduction
ODENet models the outputs of NN layers with a function f(x). In tradition network, layer index is within integer domain. ODENet extends the index into real number domain. The f(x) is defined by an ordinary differential equation df(x)/dx = F(f(x),x,theta). This demo use lorenz function as the f(x). The demo learns the parameter with ODENet algorithm.

## Note
The demo currently crash at odeint function due to its restriction on input time point. it requires the time points are provided in ascending order. I already filed for feature improvement [here](https://github.com/tensorflow/tensorflow/issues/24823). The demo is supposed to work after the odeint accepts unordered timepoints.

