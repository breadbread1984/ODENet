# ODENet
This project implements a simple demo of [ODENet](https://arxiv.org/abs/1806.07366) with tensorflow

## Introduction
ODENet models the outputs of NN layers with a function f(x). In tradition network, layer index is within integer domain. ODENet extends the index into real number domain. The f(x) is defined by an ordinary differential equation df(x)/dx = F(f(x),x,theta). This demo use lorenz function as the f(x). The demo learns the parameter with ODENet algorithm.

