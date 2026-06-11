# Basic Concepts

This page summarizes the data and model conventions used by `pyoci`.

## Signals

Input and output signals are represented as NumPy matrices shaped `(N, n)`:

- `N` is the number of samples.
- `n` is the number of inputs or outputs.

For a SISO system, use one-column arrays shaped `(N, 1)` rather than one-dimensional vectors.

## Transfer functions

`pyoci` uses SciPy discrete-time transfer functions, usually created with `dt=1`:

```python
from scipy import signal

G = signal.TransferFunction([1], [1, -0.9], dt=1)
```

When a numerator is one, use `[1]` rather than `[0, 1]` to avoid SciPy warnings.

## SISO systems

For SISO systems, all inputs (`Td`, `Cf`, `L`) can be passed directly as a transfer function. Internally, `pyoci` converts them to the same nested-list format used for MIMO systems.

```python
Td = signal.TransferFunction([0.2], [1, -0.8], dt=1)
Cf = signal.TransferFunction([1], [1, -1], dt=1)
L  = signal.TransferFunction([0.25], [1, -0.75], dt=1)
```

## MIMO transfer matrices

MIMO transfer functions are represented as nested Python lists. The first index is the output and the second index is the input.

```python
Td = [
    [Td11, Td12],
    [Td21, Td22],
]
```

Zero entries are written as the literal value `0` or as a zero transfer function `signal.TransferFunction([0], [1], dt=1)`.

## Controller structure

The OCI controller has a split structure:

```
C = Cf * Ci
```

- `Cf` is the **fixed part**, defined by the user (e.g. an integrator for steady-state tracking).
- `Ci` is the **non-fixed part**, estimated automatically by identifying `C_til = 1/Ci` from data.

This decomposition allows the user to enforce structural constraints (such as integral action) while the method estimates the remaining degrees of freedom.

## Identification models

The non-fixed part `Ci` is estimated by identifying `C_til = 1/Ci` using one of the following models:

| Model | Description | Formulation |
|-------|-------------|-------------|
| `'OE'` | Output-Error | `y(t) = [B(z)/F(z)]u(t) + e(t)` |
| `'ARX'` | Auto-Regressive with eXogenous input | `A(z)y(t) = B(z)u(t) + e(t)` |
| `'ARMAX'` | Auto-Regressive Moving Average with eXogenous input | `A(z)y(t) = B(z)u(t) + C(z)e(t)` |
| `'BJ'` | Box-Jenkins | `y(t) = [B(z)/F(z)]u(t) + [C(z)/D(z)]e(t)` |

## Model parameters

The `model_params` argument is a `ny × nu` nested list. Each element `model_params[i][j]` is a list of polynomial orders for the transfer function from input `j` to output `i`.

The required parameters depend on the chosen model:

| Model | Parameters |
|-------|------------|
| `'OE'` | `[nb, nf, nk]` |
| `'ARX'` | `[na, nb, nk]` |
| `'ARMAX'` | `[na, nb, nc, nk]` |
| `'BJ'` | `[nb, nc, nd, nf, nk]` |

Example for a SISO OE model with `nb=1, nf=1, nk=1`:

```python
model_params = [[ [1, 1, 1] ]]
```

Example for a 2×2 MIMO OE model:

```python
model_params = [
    [ [1, 1, 1], [1, 1, 1] ],
    [ [1, 1, 1], [1, 1, 1] ],
]
```

## OCI design call

The main design call is:

```python
C = oci.design(u, y, Td, Cf, method, model_params, L)
```

where:

- `Td` is the desired closed-loop reference model.
- `Cf` is the fixed part of the controller structure.
- `method` is the identification model string.
- `model_params` are the identification polynomial orders.
- `L` is the OCI pre-filter.
