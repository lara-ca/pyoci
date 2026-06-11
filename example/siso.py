# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 09:15:39 2026
@author: Lara Colognese de Almeida
"""
"""
Testing the OCI on a SISO example
"""
#%% Header: importing python libraries

import numpy as np  # important package for scientific computing
from scipy import signal  # signal processing library
import matplotlib.pyplot as plt  # library to plot graphics
import vrft # vrft package
import oci  # oci package

#%% Simulating the open loop system to obtain the data for the OCI

# declaration of the SISO transfer fuction of the process G(z)
G = signal.TransferFunction([1], [1, -0.9], dt=1)
# IMPORTANT: if the numerator of the transfer function is 1, for example, define it as num=[1], instead of num=[0,1]
# num=[0,1] produces a warning!

# number of samples
N = 100

# step signal
u = np.ones((N, 1))
u[0] = 0
# IMPORTANT: in our package, we decided to organize the input and output signals as a matrix (N,n)
# N=number of data samples, n=number of inputs and outputs

# calculating the output of the system
yu = vrft.filter(G, u)

# add noise to the output
# variance of the whie noise signal
sigma2_e1 = 0.1
# creating noise vector
w = np.random.normal(0, np.sqrt(sigma2_e1), N)
# pushing the dimensions to match our signals
w.shape = (N, 1)

# real (measured) output
y = yu + w

#%% Graphics

lw=1.5 # linewidth

# plot input signal
plt.figure()
plt.plot(u, "b", drawstyle="steps", linewidth=lw, label="u(t)")
plt.grid(True)
plt.xlabel("time (samples)")
plt.ylabel("u(t)")
plt.xlim(left=-2, right=N)
plt.show()

# plot output signal
plt.figure()
plt.plot(y, "b", drawstyle="steps", linewidth=lw, label="u(t)")
plt.grid(True)
plt.xlabel("time (samples)")
plt.ylabel("y(t)")
plt.xlim(left=-2, right=N)
plt.show()

#%% Control - OCI parameters: reference model Td(z), filter L(z), fixed controller structure Cf(z) and the model parameters

# declaration of the transfer fuction of the reference model Td(z)
Td = signal.TransferFunction([0.2], [1, -0.8], dt=1)

# choosing the OCI method filter
L = signal.TransferFunction([1], [1], dt=1)

# defining the fixed controller that will be used in the method
Cf = signal.TransferFunction([1], [1, -1], dt=1) 

# Orders of the inverse of the free controller
# In the OCI method, the free controller is denoted as C_i(z)
# Instead of identifying C_i(z) directly, we identify its inverse:
#
#     C_til(z) = 1 / C_i(z)
#
# For the Output-Error ('OE') model, this inverse is parameterized as:
#
#     C_til(z) = B(z) / F(z) * z^(-nk)

model = 'OE' 

nb = 0   # order of the polynomial B(z)
nf = 1   # order of the polynomial F(z)     
nk = 1   # input delay (number of samples) 

#%% Design the controller using the OCI method - without noise
C = oci.design(u, yu, Td, Cf, model, [[[nb, nf, nk]]], L)
print("Ideal Controller, C(z) = ", C)

# %% Adding noise to the output 
C = oci.design(u, y, Td, Cf, model, [[[nb, nf, nk]]], L)
print("Controller with Noise, C(z) = ", C)

# %% Changing the filter 
L = signal.TransferFunction([0.25], [1, -0.75], dt=1)

C = oci.design(u, y, Td, Cf, model, [[[nb, nf, nk]]], L)
print("Controller with Noise and Changed Filter, C(z) = ", C)

# %%
