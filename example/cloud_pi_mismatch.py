# -*- coding: utf-8 -*-
"""
Created on Fri May 22 10:25:17 2026
@author: Lara Colognese de Almeida
"""
"""
Testing the OCI method with PI controller structure on a SISO system (mismatch case)
Analyzes the variance of the closed-loop frequency response through a Bode diagram
cloud obtained via Monte Carlo simulation with noisy output
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

# random input signal (binary: -2 or 2)
u = np.random.choice([-2, 2], size=(N, 1))

# time vector for plotting
t = np.arange(len(u))

# declaration of the SISO transfer function of the process G(z)
# mismatch case: G has a zero at z = 0.5, making the ideal controller outside the PI class
G = signal.TransferFunction([1, -0.5], [1, -0.9], dt=1)

# calculating the output of the system by filtering input through G(z)
y = vrft.filter(G, u)

#%% Control - OCI parameters: reference model Td(z), filter L(z), fixed controller structure Cf(z)

# desired closed-loop pole location parameter
a = 0.4

# declaration of the transfer function of the reference model Td(z)
Td = signal.TransferFunction([1 - a], [1, -a], dt=1)

# defining the fixed controller structure (integrator only)
Cf = signal.TransferFunction([1], [1, -1], dt=1)

# choosing the OCI method filter
L = signal.TransferFunction([1], [1], dt=1)

# orders for OE model:
nb = 0      # order of the polynomial B(z)
nf = 1      # order of the polynomial F(z)
nk = 1      # input delay (number of samples)

#%% Monte Carlo simulation: design controller for multiple noisy realizations

# variance of the white noise signal
sigma2_e = 0.5

# number of Monte Carlo repetitions
n_reps = 500

# reference frequency vector from Td(z)
w_ref, _, _ = signal.dbode(Td)

# lists to store frequency response data for Bode cloud
mag_list = []
phase_list = []

for i in range(n_reps):
    # generate new noise realization
    w = np.random.normal(0, np.sqrt(sigma2_e), (N, 1))
    y_noise = y + w

    # design controller using OCI with noisy output
    C = oci.design(u, y_noise, Td, Cf, 'OE', [[[nb, nf, nk]]], L)

    # calculating the closed-loop transfer function
    T_i = oci.siso_closed_loop(C[0][0], G)

    # compute Bode response at reference frequencies
    _, mag_i, phase_i = signal.dbode(T_i, w=w_ref)
    mag_list.append(mag_i)
    phase_list.append(phase_i)

#%% Graphics - Bode cloud comparison

lw = 1.5  # linewidth

# reference Bode curve from Td(z)
w_td, mag_td, phase_td = signal.dbode(Td)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), dpi=300, sharex=True)

# magnitude plot - Bode cloud
for mag_i in mag_list:
    ax1.semilogx(w_ref, mag_i, color='steelblue', alpha=1, linewidth=0.8)
ax1.semilogx(w_td, mag_td, 'r-', linewidth=lw, label="Td(z) - reference model")
ax1.set_title(f"Bode Cloud - PI Mismatch ($\\sigma^2 = {sigma2_e}$, {n_reps} reps)")
ax1.set_ylabel("Magnitude [dB]")
ax1.grid(True, which="both", linestyle="--", alpha=0.5)
ax1.legend()

# phase plot - Bode cloud
for phase_i in phase_list:
    ax2.semilogx(w_ref, phase_i, color='steelblue', alpha=1, linewidth=0.8)
ax2.semilogx(w_td, phase_td, 'r-', linewidth=lw, label="Td(z) - reference model")
ax2.set_xlabel("Frequency [rad/s]")
ax2.set_ylabel("Phase [degrees]")
ax2.grid(True, which="both", linestyle="--", alpha=0.5)
ax2.legend()

plt.tight_layout()
plt.show()
# %%
