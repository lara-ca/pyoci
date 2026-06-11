# Installation

## Install from PyPI

Install `pyoci` with pip:

```bash
pip install pyoci
```

The package name on PyPI is `pyoci`, but the Python import package is `oci`.

```python
import oci
```

## Dependencies

`pyoci` depends on:

- NumPy >= 1.20, < 2.0
- SciPy >= 1.7, < 1.14
- Matplotlib >= 3.5
- pyvrft
- pysid
- IPython

These dependencies are installed automatically when installing the package with pip.

## Development installation

To install from a local checkout:

```bash
git clone https://github.com/lara-ca/pyoci
cd pyoci
pip install .
```

## Verify the installation

Check that the package can be imported:

```bash
python -c "import oci; print('pyoci installed successfully')"
```

If you are working from the repository, run the test suite from the repository root:

```bash
pytest
```
