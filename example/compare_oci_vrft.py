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
N = 50

# random input signal (binary: -1 or 1)
u = np.random.choice([-1, 1], size=(N, 1))

# time vector for plotting
t = np.arange(len(u))

# declaration of the SISO transfer function of the process G(z)
G = signal.TransferFunction([0.5], [1, -0.9], dt=1)

# calculating the output of the system by filtering input through G(z)
y = vrft.filter(G, u)

#%% Control - OCI parameters: reference model Td(z), filter L(z), fixed controller structure Cf(z)

# desired closed-loop pole location parameter
a = 0.6

# declaration of the transfer function of the reference model Td(z)
Td = signal.TransferFunction([1 - a], [1, -a], dt=1)

# defining the fixed controller structure (PI structure with integrator)
Cf = signal.TransferFunction([1], [1, -1], dt=1) 

# choosing the OCI method filter
L = signal.TransferFunction([1], [1], dt=1)

#%% Design the ideal controller (without noise)

# orders for OE model:
nb = 0      # order of the polynomial B(z)
nf = 1      # order of the polynomial F(z)     
nk = 1      # input delay (number of samples) 

C_ideal = oci.design(u, y, Td, Cf, 'OE', [[[nb, nf, nk]]], L)

# extract controller coefficients
b0 = C_ideal[0][0].num[0]
b1 = C_ideal[0][0].num[1]

# calculate controller gain and zero
K_ideal_oci = b0  # gain of the ideal controller
z_ideal_oci = -b1 / b0  # zero of the ideal controller

print("Ideal Controller (without noise), C(z) = ", C_ideal)

#%% VRFT design for comparison 
C_vrft = [
    [signal.TransferFunction([1, 0], [1, -1], dt=1)],
    [signal.TransferFunction([1], [1, -1], dt=1)],
]  # PI controller structure

# design controller using VRFT 
p = vrft.design(u, y, y, Td, C_vrft, L)

# calculate gain and zero for VRFT-designed controller
K_ideal_vrft = p[0][0]
z_ideal_vrft = - p[1][0] / p[0][0]

print ("Coeficients designed by VRFT, C_vrft(z) = ", p)

#%% Adding noise and analyzing controller performance through multiple repetitions

# variance of the white noise signal
sigma2_e1 = 0.1

# lists to store controller parameters across repetitions
K_list_oci, z_list_oci, K_list_vrft, z_list_vrft, K_list_vrft_iv, z_list_vrft_iv = [], [], [], [], [], []

# create two noise vector and add to output
w = np.random.normal(0, np.sqrt(sigma2_e1), N)
w2 = np.random.normal(0, np.sqrt(sigma2_e1), N) # for instrumental variable in VRFT
w.shape = (N, 1)
w2.shape = (N, 1)
y_noise = y + w
y_noise_2 = y + w2 # second noisy output for instrumental variable in VRFT

# plot showing noise, output, and noisy output
plt.figure()
plt.plot(y, label="y (ideal output)")
plt.plot(y_noise, label="y_noise (measured output)")
plt.plot(y_noise_2, label="y_noise_2 (measured output 2)")
plt.legend()
plt.title("Ideal Output vs Measured Output")
plt.show()

#%% Monte Carlo simulation: design controller for multiple noisy realizations

# number of Monte Carlo repetitions
n_reps = 200

for i in range(n_reps):
    # generate new noise realization
    w = np.random.normal(0, np.sqrt(sigma2_e1), (N, 1))
    y_noise = y + w

    # design controller on noisy output
    C = oci.design(u, y_noise, Td, Cf, 'OE', [[[nb, nf, nk]]], L)

    # extract numerator and denominator
    num = C[0][0].num
    den = C[0][0].den
    b0 = num[0]
    b1 = num[1]
    
    # calculate gain and zero
    K = b0
    z0 = -b1 / b0

    # append to lists
    K_list_oci.append(K)
    z_list_oci.append(z0)

    # design controller using VRFT with noisy output
    p = vrft.design(u, y_noise, y_noise, Td, C_vrft, L)

    # calculate gain and zero for VRFT-designed controller
    K = p[0][0]
    z0 = - p[1][0] / p[0][0]

    # append to lists
    K_list_vrft.append(K)
    z_list_vrft.append(z0)

    # design controller using VRFT with noisy output and instrumental variable (y_noise_2)
    p = vrft.design(u, y_noise, y_noise_2, Td, C_vrft, L)

    # calculate gain and zero for VRFT-designed controller with instrumental variables
    K = p[0][0]
    z0 = - p[1][0] / p[0][0]

    # append to lists 
    K_list_vrft_iv.append(K)
    z_list_vrft_iv.append(z0)

#%% Visualization: cloud plot of controller parameters (gain vs zero) - OCI
fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(K_list_oci, z_list_oci, color='steelblue', alpha=0.6, zorder=3, label=f'OCI with noise')
ax.scatter(K_ideal_oci, z_ideal_oci, color='crimson', s=50, zorder=5, marker='o', label='Ideal')

ax.set_xlabel('$K$ — controller gain')
ax.set_ylabel('$z_0$ — controller zero')
ax.set_title(f'OCI — controller zero and gain ($\\sigma^2 = {sigma2_e1}$) - {n_reps} reps')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

#%% Visualization: cloud plot of controller parameters (gain vs zero) - VRFT
fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(K_list_vrft, z_list_vrft, color='seagreen', alpha=0.6, zorder=3, label=f'VRFT with noise')
ax.scatter(K_ideal_vrft, z_ideal_vrft, color='crimson', s=50, zorder=5, marker='o', label='Ideal')

ax.set_xlabel('$K$ — controller gain')
ax.set_ylabel('$z_0$ — controller zero')
ax.set_title(f'VRFT — controller zero and gain ($\\sigma^2 = {sigma2_e1}$) - {n_reps} reps')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

#%% Visualization: cloud plot of controller parameters (gain vs zero) - VRFT with instrumental variables
fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(K_list_vrft_iv, z_list_vrft_iv, color='gold', alpha=0.6, zorder=3, label=f'VRFT with noise + IV')
ax.scatter(K_ideal_vrft, z_ideal_vrft, color='crimson', s=50, zorder=5, marker='o', label='Ideal')

ax.set_xlabel('$K$ — controller gain')
ax.set_ylabel('$z_0$ — controller zero')
ax.set_title(f'VRFT — controller zero and gain ($\\sigma^2 = {sigma2_e1}$) - {n_reps} reps')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

#%% Visualization: cloud plot of controller parameters (gain vs zero) - comparing OCI, VRFT, and VRFT with instrumental variables
fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(K_list_oci, z_list_oci, color='steelblue', alpha=0.6, zorder=3, label=f'OCI with noise')
ax.scatter(K_list_vrft, z_list_vrft, color='seagreen', alpha=0.6, zorder=3, label=f'VRFT with noise')
ax.scatter(K_list_vrft_iv, z_list_vrft_iv, color='gold', alpha=0.6, zorder=3, label=f'VRFT with noise + IV')
ax.scatter(K_ideal_oci, z_ideal_oci, color='crimson', s=50, zorder=5, marker='o', label='Ideal')

ax.set_xlabel('$K$ — controller gain')
ax.set_ylabel('$z_0$ — controller zero')
ax.set_title(f'Comparison OCI vs VRFT — controller zero and gain ($\\sigma^2 = {sigma2_e1}$) - {n_reps} reps')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

# %%
