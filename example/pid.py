# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 09:15:39 2026
@author: Lara Colognese de Almeida
"""
"""
Testing the OCI method with a PID controller structure on a SISO system
Demonstrates various controller configurations on second-order SISO system
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
# this is a second-order system with zero at z = 0
G = signal.TransferFunction([1, 0], [1, -1.7, 0.72], dt=1)

# calculating the output of the system
y = vrft.filter(G, u)

#%% Control - OCI parameters: reference model Td(z), filter L(z), fixed controller structures

# desired closed-loop pole location parameter
a = 0.4

# declaration of the transfer function of the reference model Td(z)
Td = signal.TransferFunction([1 - a], [1, -a], dt=1)

# defining the fixed controller structure (integrator-based structure only)
Cf_integrator = signal.TransferFunction([1], [1, -1], dt=1)  # fixes only the integrator

# alternative: full denominator constraint 
Cf_full = signal.TransferFunction([1], [1, -1, 0], dt=1)  # fixes the complete denominator

# choosing the OCI method filter
L = signal.TransferFunction([1], [1], dt=1)

#%% Design controller using OE model structure - Configuration 1 (Fixes only the integrator)

# orders for OE model:
nb = 1      # order of the polynomial B
nf = 2      # order of the polynomial F
nk = 1      # time delay (number of samples)

C = oci.design(u, y, Td, Cf_integrator, 'OE', [[[nb, nf, nk]]], L)
print("Controller C(z):", C)

#%% Design controller using OE model structure - Configuration 2 (Fixes the complete denominator)

# orders for OE model:
nb2 = 0      # order of the polynomial B
nf2 = 2      # order of the polynomial F
nk2 = 2      # time delay (number of samples)

C = oci.design(u, y, Td, Cf_full, 'OE', [[[nb2, nf2, nk2]]], L)
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
C = oci.design(u, y_noise, Td, Cf_full, 'OE', [[[nb2, nf2, nk2]]], L)
print("Controller C(z) with noise:", C)

# %%
