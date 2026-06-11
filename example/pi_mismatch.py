# -*- coding: utf-8 -*-
"""
Created on Thu May 21 08:47:14 2026
@author: Lara Colognese de Almeida
"""
"""
Testing the OCI method with PI controller structure on a SISO system (mismatch case)
Demonstrates controller design using PI structure when the ideal controller is outside
the considered controller class
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
# mismatch case: G has a zero at z = 0.5, making the ideal controller outside the PI class
G = signal.TransferFunction([1, -0.5], [1, -0.9], dt=1)

# calculating the output of the system
y = vrft.filter(G, u)
y = np.array(y)

#%% Control - OCI parameters: reference model Td(z), filter L(z), fixed controller structure Cf(z)

# desired closed-loop pole location parameter
a = 0.5

# declaration of the transfer function of the reference model Td(z)
Td = signal.TransferFunction([1 - a], [1, -a], dt=1)

# defining the fixed controller structure (integrator only)
Cf = signal.TransferFunction([1], [1, -1], dt=1)

# choosing the OCI method filter
L = signal.TransferFunction([1], [1], dt=1)

#%% Design controller using OE model structure

# orders for OE model:
nb = 0      # order of the polynomial B(z)
nf = 1      # order of the polynomial F(z)
nk = 1      # time delay (number of samples)

C = oci.design(u, y, Td, Cf, 'OE', [[[nb, nf, nk]]], L)
print("Controller C(z):", C)

# calculating the closed-loop transfer function
T = oci.siso_closed_loop(C[0][0], G)

#%% Graphics - closed-loop step response

lw = 1.5  # linewidth

y_T = vrft.filter(T, u)
y_Td = vrft.filter(Td, u)

plt.figure()
plt.plot(y_T, "b", drawstyle="steps", linewidth=lw, label="T(z) - closed-loop")
plt.plot(y_Td, "r--", drawstyle="steps", linewidth=lw, label="Td(z) - reference model")
plt.grid(True)
plt.xlabel("time (samples)")
plt.ylabel("Amplitude")
plt.xlim(left=-2, right=N)
plt.legend()
plt.show()

#%% Graphics - Bode diagram comparison between Td(z) and T(z)

w1, mag1, phase1 = signal.dbode(Td)
w2, mag2, phase2 = signal.dbode(T)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), dpi=300, sharex=True)

# magnitude plot
ax1.semilogx(w1, mag1, linewidth=lw, label="Td(z) - reference model")
ax1.semilogx(w2, mag2, linewidth=lw, linestyle="--", label="T(z) - closed-loop")
ax1.set_title("Bode Diagram - Mismatch Case")
ax1.set_ylabel("Magnitude [dB]")
ax1.grid(True, which="both", linestyle="--", alpha=0.5)
ax1.legend()

# phase plot
ax2.semilogx(w1, phase1, linewidth=lw, label="Td(z) - reference model")
ax2.semilogx(w2, phase2, linewidth=lw, linestyle="--", label="T(z) - closed-loop")
ax2.set_xlabel("Frequency [rad/s]")
ax2.set_ylabel("Phase [degrees]")
ax2.grid(True, which="both", linestyle="--", alpha=0.5)
ax2.legend()

plt.tight_layout()
plt.show()