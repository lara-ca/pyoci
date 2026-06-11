# API Reference

The package is installed from PyPI as `pyoci`, but the Python import package is `oci`.

```python
import oci
```

The public API is re-exported from `oci/__init__.py`.

## Main function

### `oci.design(u, y, Td, Cf, method, model_params, L)`

Designs a feedback controller using the OCI method.

Parameters:

- `u`: input data matrix shaped `(N, nu)`.
- `y`: output data matrix shaped `(N, ny)`.
- `Td`: desired closed-loop reference model. SISO transfer function or MIMO nested list.
- `Cf`: fixed part of the controller structure. SISO transfer function or MIMO nested list.
- `method`: identification model string. One of `'OE'`, `'ARX'`, `'ARMAX'`, `'BJ'`.
- `model_params`: polynomial orders for the identification model. Nested list of shape `ny × nu`. See [Basic Concepts](basic-concepts.md) for the format per model.
- `L`: OCI pre-filter. SISO transfer function or MIMO nested list.

Returns:

- `C`: the full controller `C = Cf * Ci`, as a MIMO nested list of transfer functions.

Raises:

- `RuntimeError`: if the stable inversion of `Cf`, `(1 - Td)`, or `C_til` fails.
- `ValueError`: if `model_params` dimensions or parameter values are invalid.

---

## Transfer function utilities

### `oci.eye_tf(p, dt=1)`

Creates a `p × p` identity transfer function matrix.

Parameters:

- `p`: matrix size.
- `dt`: sampling time (default `1`).

Returns:

- Nested list of shape `(p, p)` with `1/1` on the diagonal and `0/1` off-diagonal.

---

### `oci.subtract_tf(G1, G2)`

Subtracts two MIMO transfer function matrices element-wise: `G1 - G2`.

Parameters:

- `G1`, `G2`: nested lists of the same shape `(p, m)`.

Returns:

- Nested list `(p, m)` with the element-wise difference.

---

### `oci.multiply_tf(G1, G2)`

Multiplies two MIMO transfer function matrices: `G1 * G2`.

Parameters:

- `G1`: nested list of shape `(p, q)`.
- `G2`: nested list of shape `(q, m)`.

Returns:

- Nested list of shape `(p, m)`.

---

### `oci.invert_mtf(G, dt=1)`

Calculates the inverse of a square MIMO transfer function matrix using the adjugate-determinant formula: `G^{-1} = adj(G) / det(G)`.

Parameters:

- `G`: square nested list of shape `(n, n)`.
- `dt`: sampling time (default `1`).

Returns:

- `(G_inv, flag)` where:
  - `flag = 0`: success.
  - `flag = 1`: not invertible (determinant is zero).
  - `flag = 2`: inverse is unstable (zeros of `det(G)` are outside the unit circle).

---

### `oci.siso_closed_loop(C, G)`

Calculates the closed-loop transfer function for a SISO feedback loop.

Parameters:

- `C`: controller transfer function.
- `G`: plant transfer function.

Returns:

- Closed-loop transfer function `C*G / (1 + C*G)`.

---

### `oci.clean_coefficients(arr, threshold=1e-10)`

Sets near-zero coefficients to zero and removes leading zeros from a coefficient array.

Parameters:

- `arr`: NumPy array of polynomial coefficients.
- `threshold`: values with absolute value below this are set to zero (default `1e-10`).

Returns:

- Cleaned NumPy array.

---

## Validation utilities

### `oci.validate_model_inputs(model, model_params, ny, nu)`

Validates the `model` string and `model_params` structure against the expected format.

Raises `ValueError` or `TypeError` with a descriptive message if anything is wrong.

---

### `oci.unpack_model_params(model, model_params)`

Converts `model_params` from the `ny × nu` interface layout to separate parameter matrices expected by `pysid`.

Parameters:

- `model`: identification model string.
- `model_params`: nested list of shape `ny × nu`.

Returns:

- Dictionary mapping parameter names (e.g. `'nb'`, `'nf'`, `'nk'`) to their corresponding matrices.
