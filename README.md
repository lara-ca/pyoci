# pyoci

Optimal Control Identification

## Description

This Python Toolbox provides commands to design feedback controllers using the Optimal Control Identification method.
The toolbox implements both SISO and MIMO controllers, combining system identification and the Virtual Reference Feedback Tuning method.

## Install

Use PIP to install:

```bash
pip install pyoci
```

## Use

Please check the *example* folder. Basic use:

```Python
C = oci.design(u, y, Td, Cf, method, model_params, L)
```
where *u* and *y* are input/output data, *Td* is the reference model, *Cf* describes the controller structure, *method* is the identification method, *model_params* are the identification parameters and *L* is a pre-filter.

## Contributors

Lara Colognese de Almeida - lara.almeida@ufrgs.com - @lara-ca