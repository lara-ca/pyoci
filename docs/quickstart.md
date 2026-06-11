# Quick Start

This example designs a SISO PI controller using OCI with an OE identification model.

## Generate input/output data

```python
import numpy as np
from scipy import signal
import oci

G = signal.TransferFunction([1], [1, -0.9], dt=1)

N = 100
u = np.ones((N, 1))
u[0] = 0

# Simulate the plant output
_, y_arr = signal.dlsim(G, u.flatten())
y = y_arr.reshape(N, 1)
```

Signals in `pyoci` are organized as NumPy matrices shaped `(N, n)`, where `N` is the number of samples and `n` is the number of inputs or outputs.

## Define the OCI design objects

```python
Td = signal.TransferFunction([0.2], [1, -0.8], dt=1)
L  = signal.TransferFunction([0.25], [1, -0.75], dt=1)

Cf = signal.TransferFunction([1], [1, -1], dt=1)
```

Here:

- `Td` is the desired closed-loop reference model.
- `L` is the OCI pre-filter.
- `Cf` is the fixed part of the controller structure (e.g. an integrator).

## Define the identification model parameters

```python
method = 'OE'
model_params = [[ [1, 1, 1] ]]  # SISO: [[nb, nf, nk]]
```

The `model_params` structure is always a `ny × nu` matrix of lists. For a SISO system `ny = nu = 1`, so it is a 1×1 nested list. Each inner list contains the polynomial orders for the chosen identification model.

## Design the controller

```python
C = oci.design(u, y, Td, Cf, method, model_params, L)
print(C)
```

`C` is the full controller `C = Cf * Ci`, where `Ci` is the non-fixed part estimated by the OCI method.

## SciPy transfer-function note

When a transfer-function numerator is one, define it as `[1]` rather than `[0, 1]` to avoid SciPy warnings.
