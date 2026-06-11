# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 09:15:39 2026
@author: Lara Colognese de Almeida
"""
"""
Testing the OCI method with PI controller structure on a SISO system
Demonstrates controller design using PI structure on SISO system
"""

#%% Header: importing python libraries

import numpy as np  # important package for scientific computing
import matplotlib.pyplot as plt  # library to plot graphics
from scipy import signal  # signal processing library
import oci  # oci package
import vrft  # vrft package

#%% Simulating the open loop system to obtain data for controller design

# sampling time
Ts = 1

# number of samples
N = 50

# step signal
u = np.ones((N, 1))
u[0] = 0  # zero initial condition
# IMPORTANT: signals are organized as (N, n) matrices

# time vector
t = np.arange(len(u))

# declaration of the SISO transfer function of the process G(z)
G = signal.TransferFunction([0.5], [1, -0.9], dt=1)

# calculating the output of the system
y = vrft.filter(G, u)
y = np.array(y)

#%% Control - OCI parameters: reference model Td(z), filter L(z), fixed controller structure Cf(z)

# desired closed-loop pole location parameter
a = 0.6

# declaration of the transfer function of the reference model Td(z)
Td = signal.TransferFunction([1 - a], [1, -a], dt=1)

# defining the fixed controller structure (integrator-based structure only)
Cf = signal.TransferFunction([1], [1, -1], dt=1)

# choosing the OCI method filter
L = signal.TransferFunction([1], [1], dt=1)

#%% Design controller using OE model structure

# orders for OE model:
nb = 0      # order of the polynomial B
nf = 1      # order of the polynomial F
nk = 1      # time delay (number of samples)

C = oci.design(u, y, Td, Cf, 'OE', [[[nb, nf, nk]]], L)
print("Controller C(z):", C)

#%% Adding noise to output for robustness analysis

# variance of the white noise signal
sigma2_e1 = 0.01

# creating noise vector
w = np.random.normal(0, np.sqrt(sigma2_e1), N)

# pushing the dimensions to match our signals
w.shape = (N, 1)

# real (measured) output
y_noise = y + w

# designing controller with noisy output
C = oci.design(u, y_noise, Td, Cf, 'OE', [[[nb, nf, nk]]], L)
print("Controller C(z) with noise:", C)
# %%
