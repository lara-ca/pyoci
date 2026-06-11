# pyoci

[![PyPI](https://img.shields.io/pypi/v/pyoci.svg)](https://pypi.org/project/pyoci/)
[![License:MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Optimal Control Identification Toolbox

## Description

This Python toolbox provides commands to design feedback controllers using the Optimal Control Identification (OCI) method.
The toolbox implements SISO and MIMO controller design by combining system identification with the Virtual Reference Feedback Tuning framework.

## Documentation

Documentation is available at <https://pyoci.net/>.

## Install

Use PIP to install:

```bash
pip install pyoci
```

The package name on PyPI is `pyoci`, but the Python import package is `oci`.

## Use

Please check the *example* folder. Basic use:

```python
import oci

C = oci.design(u, y, Td, Cf, method, model_params, L)
```

where `u` and `y` are input/output data, `Td` is the reference model, `Cf` is the fixed part of the controller structure, `method` is the identification model, `model_params` are the identification parameters, and `L` is a pre-filter.

## Citation

For citation guidance, see <https://pyoci.net/citation/>.

## Contributors

Lara Colognese de Almeida - lara.almeida@ufrgs.br - @lara-ca
