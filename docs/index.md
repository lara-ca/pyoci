# pyoci - Optimal Control Identification

`pyoci` is a Python toolbox for designing feedback controllers from input/output data using the Optimal Control Identification (OCI) method.

The package supports SISO and MIMO controller design by combining system identification with the Virtual Reference Feedback Tuning framework.

## Install

```bash
pip install pyoci
```

## Quick Start

```python
import oci

C = oci.design(u, y, Td, Cf, method, model_params, L)
```

where `u` and `y` are input/output data, `Td` is the reference model, `Cf` is the fixed part of the controller structure, `method` is the identification model (e.g. `'OE'`, `'ARX'`), `model_params` are the identification parameters, and `L` is the OCI pre-filter.

Signals are represented as NumPy matrices shaped `(N, n)`, where `N` is the number of samples and `n` is the number of inputs or outputs.

## Features

- SISO and MIMO OCI controller design
- Identification model support: OE, ARX, ARMAX, BJ
- Fixed and non-fixed controller structure decomposition (C = Cf * Ci)
- Transfer function utilities: identity, subtraction, multiplication, inversion
- Model input validation with detailed error messages
- Built on top of `pyvrft` and `pysid`

## Documentation

- [Installation](installation.md): install from PyPI or from a development checkout.
- [Quick Start](quickstart.md): design a first SISO controller with OCI.
- [Basic Concepts](basic-concepts.md): signal shapes, transfer-function conventions, controller structures, and identification models.
- [Examples](examples.md): runnable SISO and MIMO examples from the repository.
- [API Reference](api.md): public functions re-exported by `oci`.
- [Citation](citation.md): citation information.

## Links

- [GitHub](https://github.com/lara-ca/pyoci)

- [PyPI](https://pypi.org/project/pyoci/)
