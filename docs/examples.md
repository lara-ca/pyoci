# Examples

Example scripts are available in the repository's `example/` directory.

The examples use Matplotlib and open figures when executed.

## Available examples

### `siso.py`

Designs a SISO controller using the OCI method and compares the result against VRFT.

### `siso_id_models.py`

Demonstrates SISO controller design with different identification models (OE, ARX, ARMAX, BJ) and compares their results.

### `pi.py`

Designs a PI controller using OCI. Uses an integrator as the fixed part `Cf` and estimates the proportional gain via OCI.

### `pi_mismatch.py`

Repeats the PI design scenario with a deliberate model mismatch to illustrate robustness.

### `pid.py`

Designs a PID controller using OCI with a fixed integrator structure.

### `pid_mismatch.py`

Repeats the PID design with model mismatch.

### `cloud_pi.py`

Simulates a cloud-based process and designs a PI controller using OCI.

### `cloud_pi_mismatch.py`

Cloud PI design with model mismatch.

### `cloud_pid.py`

Cloud process PID design using OCI.

### `cloud_pid_mismatch.py`

Cloud PID design with model mismatch.

### `compare_oci_vrft.py`

Side-by-side comparison of OCI and VRFT controller designs on the same dataset, showing the differences between the two methods.

## Running the examples

Install the package first:

```bash
pip install .
```

Then run the examples from the `example/` directory:

```bash
cd example
python siso.py
python pi.py
python pid.py
python compare_oci_vrft.py
```
