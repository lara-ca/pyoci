# -*- coding: utf-8 -*-
"""
Created on Wed May 20 09:15:39 2026
@author: Lara Colognese de Almeida
"""
"""
Testing the OCI method and analyzing the variance of the controller coefficients with noisy output
"""

#%% Header: importing python libraries

import numpy as np  # important package for scientific computing
from scipy import signal  # signal processing library
import matplotlib.pyplot as plt  # library to plot graphics
import oci  # oci package
import vrft  # vrft package

#%% Simulating the open loop system to obtain data for controller design

# sampling time
Ts = 1

# number of samples
N = 100

# step signal
u = np.random.normal(0, np.sqrt(10), (N, 1))
u[0] = 0  # zero initial condition

# IMPORTANT: signals are organized as (N, n) matrices

# time vector for plotting
t = np.arange(len(u))

# declaration of the SISO transfer function of the process G(z)
# Second-order system with zeros
G = signal.TransferFunction([1, 0], [1, -1.7, 0.72], dt=1)

# calculating the output of the system
y = vrft.filter(G, u)
y = np.array(y)

#%% Control - OCI parameters: reference model Td(z), filter L(z), fixed controller structures

# desired closed-loop pole location parameter
a = 0.4

# declaration of the transfer function of the reference model Td(z)
Td = signal.TransferFunction([1 - a], [1, -a], dt=1)

# defining the fixed controller structure (first-order)
Cf = signal.TransferFunction([1], [1, -1, 0], dt=1)

# choosing the OCI method filter
L = signal.TransferFunction([1], [1], dt=1)

#%% Design the ideal controller (without noise)

# orders for OE model:
nb = 0      # order of the polynomial B(z)
nf = 2      # order of the polynomial F(z)     
nk = 2      # input delay (number of samples)

C_ideal = oci.design(u, y, Td, Cf, 'OE', [[[nb, nf, nk]]], L)

num_ideal = C_ideal[0][0].num

a0_ideal = num_ideal[0]
a1_ideal = num_ideal[1]
a2_ideal = num_ideal[2]

# gain
K_ideal = a0_ideal

# zeros
zeros_ideal = np.roots([a0_ideal, a1_ideal, a2_ideal])

z1_ideal = zeros_ideal[0]
z2_ideal = zeros_ideal[1]

print("Ideal Controller C(z) = ", C_ideal)

#%% Monte Carlo simulation: adding noise and analyzing controller variance

# variance of the white noise signal
sigma2_e1 = 0.1

# lists to store controller coefficients
K_list, z1_list, z2_list = [], [], []

# create initial noisy output for visualization
w = np.random.normal(0, np.sqrt(sigma2_e1), N)
w.shape = (N, 1)
y_noise = y + w

# visualization of signals
plt.figure()
plt.plot(y, label="y (ideal output)")
plt.plot(y_noise, label="y_noise (measured output)")
plt.legend()
plt.title("Output Signals: Ideal, Noise, and Noisy Measurement")
plt.show()

#%% Repetitions: design controller for multiple noisy realizations

# number of Monte Carlo repetitions
n_reps = 200

for i in range(n_reps):

    w = np.random.normal(0, np.sqrt(sigma2_e1), (N,1))
    y_noise = y + w

    C = oci.design(u, y_noise, Td, Cf, 'OE', [[[nb, nf, nk]]], L)

    num = C[0][0].num

    a0 = num[0]
    a1 = num[1]
    a2 = num[2]

    # gain
    K = a0

    # zeros
    zeros = np.roots([a0, a1, a2])

    z1 = zeros[0]
    z2 = zeros[1]

    K_list.append(K)
    z1_list.append(z1)
    z2_list.append(z2)

#%% Visualization: cloud plot of controller zeros

fig, ax = plt.subplots(figsize=(7,5))

ax.scatter(np.real(z1_list), np.real(z2_list), color='steelblue', alpha=0.7, zorder=3, label='With noise')

ax.scatter(np.real(z1_ideal), np.real(z2_ideal), color='crimson', s=50, zorder=5, marker='o', label='Ideal')

ax.set_xlabel('$z_1$ — First zero')
ax.set_ylabel('$z_2$ — Second zero')

ax.set_title(f'OCI controller zeros ($\\sigma^2 = {sigma2_e1}$) - {n_reps} repetitions')

ax.legend()
ax.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.show()

#%% Visualization: gain and first zero

fig, ax = plt.subplots(figsize=(7,5))

ax.scatter(np.real(z1_list), np.real(K_list), color='steelblue', alpha=0.7, zorder=3, label='With noise')

ax.scatter(np.real(z1_ideal), np.real(K_ideal), color='crimson', s=50, zorder=5, marker='o', label='Ideal')

ax.set_xlabel('$z_1$ — First zero')
ax.set_ylabel('$K$ — controller gain')

ax.set_title(f'OCI controller $z_1$ and K($\\sigma^2 = {sigma2_e1}$) - {n_reps} repetitions')

ax.legend()
ax.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.show()
# %%
