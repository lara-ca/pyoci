"""
Created on Wed Apr 22 17:45:33 2018
@author: Lara Colognese de Almeida
"""

import numpy as np
from scipy import signal

# Functions used locally to perform basic operations with transfer functions
def _subtract_tf_siso(H1, H2):
    """Subtracts two scipy.signal.TransferFunction objects (H1 - H2)."""
    if H1.dt != H2.dt:
        raise ValueError("Sampling times (dt) must match.")
    
    # 1. Find the common denominator: D1 * D2
    common_den = np.polymul(H1.den, H2.den)
    
    # 2. Find the new numerator: (N1 * D2) - (N2 * D1)
    term1 = np.polymul(H1.num, H2.den)
    term2 = np.polymul(H2.num, H1.den)
    
    # Pad shorter term with leading zeros to align for subtraction
    max_len = max(len(term1), len(term2))
    term1_padded = np.pad(term1, (max_len - len(term1), 0))
    term2_padded = np.pad(term2, (max_len - len(term2), 0))
    
    res_num = term1_padded - term2_padded
    
    return signal.TransferFunction(res_num, common_den, dt=H1.dt)

def _add_tf_siso(H1, H2):
    """Adds two scipy.signal.TransferFunction objects (H1 + H2)."""
    if H1.dt != H2.dt:
        raise ValueError("Sampling times (dt) must match.")
    
    common_den = np.polymul(H1.den, H2.den)
    
    term1 = np.polymul(H1.num, H2.den)
    term2 = np.polymul(H2.num, H1.den)
    
    max_len = max(len(term1), len(term2))
    term1_padded = np.pad(term1, (max_len - len(term1), 0))
    term2_padded = np.pad(term2, (max_len - len(term2), 0))
    
    res_num = term1_padded + term2_padded
    
    return signal.TransferFunction(res_num, common_den, dt=H1.dt)

def _multiply_tf_siso(H1, H2):
    """Multiplies two scipy.signal.TransferFunction objects (H1 * H2)."""
    if H1.dt != H2.dt:
        raise ValueError("Sampling times (dt) must match.")

    num1 = np.asarray(H1.num).flatten()
    den1 = np.asarray(H1.den).flatten()
    num2 = np.asarray(H2.num).flatten()
    den2 = np.asarray(H2.den).flatten()

    res_num = np.polymul(num1, num2)
    res_den = np.polymul(den1, den2)

    return signal.TransferFunction(res_num, res_den, dt=H1.dt)

def _negate_tf(H):
    """Negates a transfer function: -H"""
    return signal.TransferFunction(-H.num, H.den, dt=H.dt)

def _invert_tf_scalar(H):
    """Inverts a scalar transfer function: 1/H"""
    return signal.TransferFunction(clean_coefficients(H.den), 
                                   clean_coefficients(H.num), dt=H.dt)

def _det_mtf(G, dt=1):
    """Calculates the determinant of a MIMO transfer function matrix G (list of lists of TransferFunction objects)."""
    n = len(G)
    
    if n == 1:
        return G[0][0]
    
    det = signal.TransferFunction([0], [1], dt=dt)
    for j in range(n):
        minor = [[G[i][k] for k in range(n) if k != j] for i in range(1, n)]
        cofactor = _det_mtf(minor, dt=dt)
        term = _multiply_tf_siso(G[0][j], cofactor)
        if j % 2 == 0:
            det = _add_tf_siso(det, term)
        else:
            det = _subtract_tf_siso(det, term)
    
    return det

def _adjugate_mtf(G, dt=1):
    """Calculates the adjugate of a MIMO transfer function matrix G (list of lists of TransferFunction objects)."""
    n = len(G)
    
    # 1x1 case: adjugate is [[1]]
    if n == 1:
        return [[signal.TransferFunction([1], [1], dt=dt)]]
    
    adj = []
    for i in range(n):
        row = []
        for j in range(n):
            minor = [[G[r][c] for c in range(n) if c != i] for r in range(n) if r != j]
            cofactor = _det_mtf(minor, dt=dt)
            if (i + j) % 2 != 0:
                cofactor = _negate_tf(cofactor)
            row.append(cofactor)
        adj.append(row)
    return adj


# Functions to be imported in __init__.py
def eye_tf(p, dt=1):
    """Creates a p×p identity transfer function matrix."""
    return [[signal.TransferFunction([1], [1], dt=dt) if i == j 
             else signal.TransferFunction([0], [1], dt=dt) 
             for j in range(p)] for i in range(p)]

def clean_coefficients(arr, threshold=1e-10):
    """Cleans the coefficients of a transfer function by setting very small values to zero and removing trailing zeros."""
    arr = np.array(arr, dtype=float)

    # Zera valores muito pequenos
    arr[np.abs(arr) < threshold] = 0.0

    # Remove apenas zeros no início do array (leading zeros)
    while len(arr) > 1 and arr[0] == 0.0:
        arr = arr[1:]

    return arr

def subtract_tf(G1, G2):
    """Subtracts two MIMO transfer function matrices (G1 - G2).
    G1 and G2 must have the same dimensions (p x m lists of TransferFunction objects)."""
    p = len(G1)
    m = len(G1[0])
    
    return [[_subtract_tf_siso(G1[i][j], G2[i][j]) for j in range(m)] for i in range(p)]

def invert_mtf(G, dt=1):
    """Calculates the inverse of a MIMO transfer function matrix.
    G^{-1} = adj(G) / det(G)
    
    Input: G (list of lists of TransferFunctionDiscrete)
    Output: G_inv, flag
        flag=0: success
        flag=1: system not invertible (det=0)
        flag=2: inverse is unstable"""
    
    n = len(G)
    
    # calculate determinant
    det = _det_mtf(G, dt=dt)
    
    # check invertibility: num of det cannot be zero
    if np.allclose(det.num, 0):
        return None, 1
    
    # check stability of the inverse: poles of G^{-1} are zeros of det(G)
    zeros = np.roots(det.num)
    unstable = np.any(np.abs(zeros) >= 1)
    
    # calculate adjugate
    adj = _adjugate_mtf(G, dt=dt)
    
    # G^{-1} = adj / det
    G_inv = [[_multiply_tf_siso(adj[i][j], _invert_tf_scalar(det)) 
              for j in range(n)] for i in range(n)]
    
    if unstable:
        return G_inv, 2
    
    return G_inv, 0

def multiply_tf(G1, G2):
    """Multiplies two MIMO transfer function matrices G1 * G2."""
    p = len(G1)
    m = len(G2[0])
    q = len(G2)
    
    result = []
    for i in range(p):
        row = []
        for j in range(m):
            entry = _multiply_tf_siso(G1[i][0], G2[0][j])
            for k in range(1, q):
                entry = _add_tf_siso(entry, _multiply_tf_siso(G1[i][k], G2[k][j]))
            row.append(entry)
        result.append(row)
    return result

def siso_closed_loop(C, G):
    """ Calculate the closed-loop transfer function """
    
    num_CG_i = signal.convolve(C.num, G.num)
    den_CG_i = signal.convolve(C.den, G.den)

    diff = len(den_CG_i) - len(num_CG_i)
    if diff > 0:
        num_CG_i = np.concatenate([np.zeros(diff), num_CG_i])
    else:
        den_CG_i = np.concatenate([np.zeros(-diff), den_CG_i])

    result = signal.TransferFunction(num_CG_i, den_CG_i + num_CG_i, dt=1)

    return result