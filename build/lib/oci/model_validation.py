"""
Created on Wed Apr 22 17:45:33 2018
@author: Lara Colognese de Almeida
"""

def validate_model_inputs(model, model_params, ny, nu):
    """
    Validate inputs for the design function.
    Supports SISO (ny=1, nu=1) and MIMO (ny>=1, nu>=1).

    model_params is a ny x nu matrix where each element contains
    the parameters for that specific transfer function:

        OE:    model_params[i][j] = [nb, nf, nk]
        ARX:   model_params[i][j] = [na, nb, nk]
        ARMAX: model_params[i][j] = [na, nb, nc, nk]
        BJ:    model_params[i][j] = [nb, nc, nd, nf, nk]

    Raises ValueError or TypeError if something is wrong.
    """

    model_param_names = {
        "OE":    ["nb", "nf", "nk"],
        "ARX":   ["na", "nb", "nk"],
        "ARMAX": ["na", "nb", "nc", "nk"],
        "BJ":    ["nb", "nc", "nd", "nf", "nk"],
    }

    if model not in model_param_names:
        raise ValueError(
            f"Invalid model '{model}'. "
            f"Valid options are: {list(model_param_names.keys())}"
        )

    if not isinstance(ny, int) or ny <= 0:
        raise ValueError(f"ny must be a positive integer. Got {ny}.")
    if not isinstance(nu, int) or nu <= 0:
        raise ValueError(f"nu must be a positive integer. Got {nu}.")

    if not isinstance(model_params, list):
        raise TypeError("model_params must be a list.")

    # Check ny rows
    if len(model_params) != ny:
        raise ValueError(
            f"model_params must have {ny} rows (ny). Got {len(model_params)}."
        )

    names = model_param_names[model]
    expected_len = len(names)

    for i, row in enumerate(model_params):
        if not isinstance(row, list):
            raise TypeError(f"model_params[{i}] must be a list. Got {type(row).__name__}.")

        # Check nu columns
        if len(row) != nu:
            raise ValueError(
                f"model_params[{i}] must have {nu} columns (nu). Got {len(row)}."
            )

        for j, params in enumerate(row):
            if not isinstance(params, list):
                raise TypeError(
                    f"model_params[{i}][{j}] must be a list. Got {type(params).__name__}."
                )

            # Check number of parameters
            if len(params) != expected_len:
                raise ValueError(
                    f"model_params[{i}][{j}] must have {expected_len} parameters "
                    f"{names} for model '{model}'. Got {len(params)}."
                )

            # Check each parameter is a non-negative integer
            for k, (val, name) in enumerate(zip(params, names)):
                if not isinstance(val, int) or val < 0:
                    raise ValueError(
                        f"model_params[{i}][{j}][{k}] ('{name}') must be a "
                        f"non-negative integer. Got {val}."
                    )

def unpack_model_params(model, model_params):
    """
    Converts model_params from ny x nu interface layout to
    separate parameter matrices as expected by pysid.

    Returns a dict, e.g. for OE: {'nb': [...], 'nf': [...], 'nk': [...]}
    """

    model_param_names = {
        "OE":    ["nb", "nf", "nk"],
        "ARX":   ["na", "nb", "nk"],
        "ARMAX": ["na", "nb", "nc", "nk"],
        "BJ":    ["nb", "nc", "nd", "nf", "nk"],
    }

    names = model_param_names[model]

    # Initialize empty matrices for each parameter
    unpacked = {name: [] for name in names}

    for row in model_params:
        unpacked_row = {name: [] for name in names}
        for params in row:
            for name, val in zip(names, params):
                unpacked_row[name].append(val)
        for name in names:
            unpacked[name].append(unpacked_row[name])

    return unpacked