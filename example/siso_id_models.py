# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 09:15:39 2026
@author: Lara Colognese de Almeida
"""
"""
Testing the OCI method on a SISO example with different model identification structures:
  OE (Output-Error), 
  ARX (AutoRegressive with eXogenous input), 
  ARMAX (AutoRegressive Moving Average),
  BJ (Box-Jenkins) models
"""

#%% Header: importing python libraries

import numpy as np  # important package for scientific computing
from scipy import signal  # signal processing library
import matplotlib.pyplot as plt  # library to plot graphics
import oci  # oci package
import vrft  # vrft package

#%% Simulating the open loop system to obtain the data for OCI

# declaration of the SISO transfer function of the process G(z)
G = signal.TransferFunction([1], [1, -0.9], dt=1)
# IMPORTANT: if the numerator of the transfer function is 1, define it as num=[1], instead of num=[0,1]
# num=[0,1] produces a warning!

# number of samples
N = 100

# step signal
u = np.ones((N, 1))
u[0] = 0
# IMPORTANT: in our package, signals are organized as (N, n) matrices
# N = number of data samples, n = number of inputs and outputs

# calculating the output of the system without noise
y = vrft.filter(G, u)

#%% Control - OCI parameters: reference model Td(z), filter L(z), and fixed controller structure Cf(z)

# declaration of the transfer function of the reference model Td(z)
Td = signal.TransferFunction([0.2], [1, -0.8], dt=1)

# choosing the OCI method filter
L = signal.TransferFunction([0.25], [1, -0.75], dt=1)

# defining the fixed controller structure that will be used in the method
Cf = signal.TransferFunction([1], [1, -1], dt=1)

#%% Design the controller using the OCI method and the OE model structure

# OE (Output-Error) model structure:
# y(t) = [B(z)/F(z)]u(t) + e(t)

nb = 0      # order of the polynomial B(z)
nf = 1      # order of the polynomial F(z)
nk = 1      # input delay (number of samples)

C = oci.design(u, y, Td, Cf, 'OE', [[[nb, nf, nk]]], L)
print("Controller C(z), with OE identification:", C)

#%% Design the controller using the OCI method and the ARX model structure

# ARX (AutoRegressive with eXogenous input) model structure:
# A(z)y(t) = B(z)u(t) + e(t)

na = 1      # order of the polynomial A(z)
nb = 0      # order of the polynomial B(z)
nk = 1      # input delay (number of samples)

C = oci.design(u, y, Td, Cf, 'ARX', [[[na, nb, nk]]], L)
print("Controller C(z), with ARX identification:", C)

#%% Design the controller using the OCI method and the ARMAX model structure

# ARMAX (AutoRegressive Moving Average with eXogenous input) model structure:
#A(z)y(t) = B(z)u(t) + C(z)e(t)

na = 1      # order of the polynomial A(z)
nb = 0      # order of the polynomial B(z)
nc = 0      # order of the polynomial C(z)
nk = 1      # input delay (number of samples)

C = oci.design(u, y, Td, Cf, 'ARMAX', [[[na, nb, nc, nk]]], L)
print("Controller C(z), with ARMAX identification:", C)

#%% Design the controller using the OCI method and the BJ model structure

# BJ (Box-Jenkins) model structure:
# y(t) = [B(z)/F(z)]u(t) + [C(z)/D(z)]e(t)

nb = 0      # order of the polynomial B(z)
nc = 0      # order of the polynomial C(z)
nd = 1      # order of the polynomial D(z)
nf = 1      # order of the polynomial F(z)
nk = 1      # input delay (number of samples)

C = oci.design(u, y, Td, Cf, 'BJ', [[[nb, nc, nd, nf, nk]]], L)
print("Controller C(z), with BJ identification:", C)
# %%
