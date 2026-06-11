
import unittest
import numpy as np
from scipy import signal
import oci

from oci.tf_operations import _add_tf_siso, _subtract_tf_siso, _multiply_tf_siso, _negate_tf, _invert_tf_scalar, _det_mtf, _adjugate_mtf, eye_tf, clean_coefficients

# ── helpers ────────────────────────────────────────────────────────────────────

def _tf_allclose(H1, H2, atol=1e-8):
    """Check if two scalar TFs are equal (up to a common scale on num/den)."""
    # normalise both by their leading coefficient so scale doesn't matter
    n1 = np.array(H1.num, dtype=float).flatten()
    d1 = np.array(H1.den, dtype=float).flatten()
    n2 = np.array(H2.num, dtype=float).flatten()
    d2 = np.array(H2.den, dtype=float).flatten()

    # pad to same length
    def pad(a, b):
        la, lb = len(a), len(b)
        if la < lb:
            a = np.pad(a, (lb - la, 0))
        elif lb < la:
            b = np.pad(b, (la - lb, 0))
        return a, b

    n1, n2 = pad(n1, n2)
    d1, d2 = pad(d1, d2)

    # cross-multiply to compare: n1/d1 == n2/d2  =>  n1*d2 == n2*d1
    lhs = np.polymul(n1, d2)
    rhs = np.polymul(n2, d1)
    lhs, rhs = pad(lhs, rhs)
    return np.allclose(lhs, rhs, atol=atol)

def _make_tf(num, den, dt=1):
    return signal.TransferFunction(num, den, dt=dt)


class TestSubtractTfSiso(unittest.TestCase):
    def test_same(self):
        # H - H = 0
        H = _make_tf([1, 2], [1, -0.5])
        result = _subtract_tf_siso(H, H)
        self.assertTrue(np.allclose(result.num, 0, atol=1e-10))

    def test_simple(self):
        # 2/z - 1/z = 1/z
        H1 = _make_tf([2], [1, 0])
        H2 = _make_tf([1], [1, 0])
        result = _subtract_tf_siso(H1, H2)
        expected = _make_tf([1], [1, 0])
        self.assertTrue(_tf_allclose(result, expected))

    def test_dt_mismatch(self):
        H1 = _make_tf([1], [1], dt=1)
        H2 = _make_tf([1], [1], dt=2)
        with self.assertRaises(ValueError):
            _subtract_tf_siso(H1, H2)

class TestAddTfSiso(unittest.TestCase):
    def test_simple(self):
        # 1/z + 1/z = 2/z
        H1 = _make_tf([1], [1, 0])
        H2 = _make_tf([1], [1, 0])
        result = _add_tf_siso(H1, H2)
        expected = _make_tf([2], [1, 0])
        self.assertTrue(_tf_allclose(result, expected))

    def test_different_denominators(self):
        # 1/(z-1) + 1/(z+1) = 2z / (z^2-1)
        H1 = _make_tf([1], [1, -1])
        H2 = _make_tf([1], [1,  1])
        result = _add_tf_siso(H1, H2)
        expected = _make_tf([2, 0], [1, 0, -1])
        self.assertTrue(_tf_allclose(result, expected))

    def test_dt_mismatch(self):
        H1 = _make_tf([1], [1], dt=1)
        H2 = _make_tf([1], [1], dt=2)
        with self.assertRaises(ValueError):
            _add_tf_siso(H1, H2)

class TestMultiplyTfSiso(unittest.TestCase):
    def test_simple(self):
        # 1/z * 1/z = 1/z^2
        H1 = _make_tf([1], [1, 0])
        H2 = _make_tf([1], [1, 0])
        result = _multiply_tf_siso(H1, H2)
        expected = _make_tf([1], [1, 0, 0])
        self.assertTrue(_tf_allclose(result, expected))

class TestNegateTf(unittest.TestCase):
    def test_negate(self):
        H = _make_tf([1.25], [1, -0.9])
        result = _negate_tf(H)
        expected = _make_tf([-1.25], [1, -0.9])
        self.assertTrue(_tf_allclose(result, expected))

    def test_double_negate(self):
        H = _make_tf([1, 2], [1, -0.5])
        self.assertTrue(_tf_allclose(_negate_tf(_negate_tf(H)), H))

class TestInvertTfScalar(unittest.TestCase):
    def test_simple(self):
        # 1/H where H = 1.25/(z-0.9)  =>  (z-0.9)/1.25
        H = _make_tf([1.25], [1, -0.9])
        result = _invert_tf_scalar(H)
        expected = _make_tf([1, -0.9], [1.25])
        self.assertTrue(_tf_allclose(result, expected))

    def test_double_invert(self):
        H = _make_tf([1, 2], [3, -1])
        self.assertTrue(_tf_allclose(_invert_tf_scalar(_invert_tf_scalar(H)), H))

class TestDetMtf(unittest.TestCase):
    def test_1x1(self):
        G = [[_make_tf([1.25], [1, -0.9])]]
        det = _det_mtf(G)
        self.assertTrue(_tf_allclose(det, _make_tf([1.25], [1, -0.9])))

    def test_2x2_diagonal(self):
        # det([[a,0],[0,b]]) = a*b
        a = _make_tf([2], [1, -0.5])
        b = _make_tf([3], [1, -0.3])
        G = [[a, _make_tf([0], [1])],
             [_make_tf([0], [1]), b]]
        det = _det_mtf(G)
        expected = _multiply_tf_siso(a, b)
        self.assertTrue(_tf_allclose(det, expected))

    def test_2x2_identity(self):
        # det(I) = 1
        one = _make_tf([1], [1])
        zero = _make_tf([0], [1])
        I = [[one, zero], [zero, one]]
        det = _det_mtf(I)
        self.assertTrue(_tf_allclose(det, one))

class TestAdjugateMtf(unittest.TestCase):
    def test_1x1(self):
        # adjugate of 1x1 matrix is always [[1]]
        G = [[_make_tf([1.25], [1, -0.9])]]
        adj = _adjugate_mtf(G)
        expected = _make_tf([1], [1])
        self.assertTrue(_tf_allclose(adj[0][0], expected))

    def test_2x2_diagonal(self):
        # For diagonal matrix [[a, 0], [0, b]], adjugate is [[b, 0], [0, a]]
        a = _make_tf([2], [1, -0.5])
        b = _make_tf([3], [1, -0.3])
        zero = _make_tf([0], [1])
        G = [[a, zero], [zero, b]]
        adj = _adjugate_mtf(G)
        # Check diagonal elements
        self.assertTrue(_tf_allclose(adj[0][0], b))
        self.assertTrue(_tf_allclose(adj[1][1], a))
        # Check off-diagonal are zero
        self.assertTrue(np.allclose(adj[0][1].num, 0, atol=1e-8))
        self.assertTrue(np.allclose(adj[1][0].num, 0, atol=1e-8))

    def test_adjugate_property(self):
        # G * adj(G) = det(G) * I
        a = _make_tf([2], [1, -0.5])
        b = _make_tf([3], [1, -0.3])
        zero = _make_tf([0], [1])
        G = [[a, zero], [zero, b]]
        adj = _adjugate_mtf(G)
        det = _det_mtf(G)
        
        # G * adj(G)
        product = oci.multiply_tf(G, adj)
        # det(G) * I
        one = _make_tf([1], [1])
        det_I = [[_multiply_tf_siso(det, one), _multiply_tf_siso(det, zero)],
                 [_multiply_tf_siso(det, zero), _multiply_tf_siso(det, one)]]
        
        # Check if G * adj(G) == det(G) * I
        for i in range(2):
            for j in range(2):
                self.assertTrue(_tf_allclose(product[i][j], det_I[i][j]))

class TestEyeTf(unittest.TestCase):
    def test_1x1(self):
        # eye_tf(1) should return [[1/1]]
        I = eye_tf(1)
        one = _make_tf([1], [1])
        self.assertTrue(_tf_allclose(I[0][0], one))

    def test_2x2(self):
        # eye_tf(2) should return identity matrix
        I = eye_tf(2)
        one = _make_tf([1], [1])
        zero = _make_tf([0], [1])
        
        # Check diagonal
        self.assertTrue(_tf_allclose(I[0][0], one))
        self.assertTrue(_tf_allclose(I[1][1], one))
        # Check off-diagonal
        self.assertTrue(np.allclose(I[0][1].num, 0, atol=1e-8))
        self.assertTrue(np.allclose(I[1][0].num, 0, atol=1e-8))

    def test_3x3(self):
        # eye_tf(3) should return 3x3 identity matrix
        I = eye_tf(3)
        one = _make_tf([1], [1])
        
        # Check dimensions
        self.assertEqual(len(I), 3)
        self.assertEqual(len(I[0]), 3)
        
        # Check diagonal and off-diagonal
        for i in range(3):
            for j in range(3):
                if i == j:
                    self.assertTrue(_tf_allclose(I[i][j], one))
                else:
                    self.assertTrue(np.allclose(I[i][j].num, 0, atol=1e-8))

class TestCleanCoefficients(unittest.TestCase):
    def test_remove_small_values(self):
        # Values smaller than threshold should be set to zero
        arr = np.array([1.0, 1e-12, 0.5, 1e-11])
        result = clean_coefficients(arr, threshold=1e-10)
        expected = np.array([1.0, 0.0, 0.5, 0.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_remove_leading_zeros(self):
        # Leading zeros should be removed (not trailing)
        arr = np.array([0.0, 0.0, 1.0, 2.0, 3.0, 0.0])
        result = clean_coefficients(arr)
        expected = np.array([1.0, 2.0, 3.0, 0.0])
        np.testing.assert_array_equal(result, expected)

    def test_preserve_trailing_zeros(self):
        # Trailing zeros should be preserved (represent higher order terms)
        arr = np.array([0.5, 1.0, 0.0])
        result = clean_coefficients(arr)
        expected = np.array([0.5, 1.0, 0.0])
        np.testing.assert_array_equal(result, expected)

    def test_preserve_inner_zeros(self):
        # Inner zeros should be preserved
        arr = np.array([1.0, 0.0, 3.0, 0.0])
        result = clean_coefficients(arr)
        expected = np.array([1.0, 0.0, 3.0, 0.0])
        np.testing.assert_array_equal(result, expected)

    def test_combined_cleaning(self):
        # Small leading values should be zeroed, then leading zeros removed
        arr = np.array([1e-12, 0.0, 1.0, 0.5, 0.0])
        result = clean_coefficients(arr, threshold=1e-10)
        expected = np.array([1.0, 0.5, 0.0])
        np.testing.assert_array_almost_equal(result, expected)

class TestSubtractTf(unittest.TestCase):
    def test_identity_minus_td_1x1(self):
        # I - Td where Td = z^{-1} = 1/z
        # (1/1) - (1/z) = (z-1)/z
        I = [[_make_tf([1], [1])]]
        Td = [[_make_tf([1], [1, 0])]]
        result = oci.subtract_tf(I, Td)
        expected_00 = _make_tf([1, -1], [1, 0])
        self.assertTrue(_tf_allclose(result[0][0], expected_00))

    def test_2x2(self):
        # element-wise subtraction
        G1 = [[_make_tf([2], [1]), _make_tf([0], [1])],
              [_make_tf([0], [1]), _make_tf([2], [1])]]
        G2 = [[_make_tf([1], [1]), _make_tf([0], [1])],
              [_make_tf([0], [1]), _make_tf([1], [1])]]
        result = oci.subtract_tf(G1, G2)
        for i in range(2):
            self.assertTrue(_tf_allclose(result[i][i], _make_tf([1], [1])))

class TestInvertTf(unittest.TestCase):
    def test_1x1(self):
        # inv(1.25/(z-0.9)) = (z-0.9)/1.25
        G = [[_make_tf([1.25], [1, -0.9])]]
        G_inv, flag = oci.invert_mtf(G)
        self.assertEqual(flag, 0)
        expected = _make_tf([1, -0.9], [1.25])
        self.assertTrue(_tf_allclose(G_inv[0][0], expected))

    def test_1x1_roundtrip(self):
        # G * G^{-1} = I
        G = [[_make_tf([1.25], [1, -0.9])]]
        G_inv, flag = oci.invert_mtf(G)
        self.assertEqual(flag, 0)
        product = _multiply_tf_siso(G[0][0], G_inv[0][0])
        self.assertTrue(_tf_allclose(product, _make_tf([1], [1])))

    def test_2x2_diagonal_roundtrip(self):
        # diagonal MIMO: inv is also diagonal with inverted elements
        a = _make_tf([2], [1, -0.5])
        b = _make_tf([3], [1, -0.3])
        zero = _make_tf([0], [1])
        G = [[a, zero], [zero, b]]
        G_inv, flag = oci.invert_mtf(G)
        self.assertEqual(flag, 0)
        # G * G_inv should be identity
        product = oci.multiply_tf(G, G_inv)
        one = _make_tf([1], [1])
        self.assertTrue(_tf_allclose(product[0][0], one))
        self.assertTrue(_tf_allclose(product[1][1], one))
        self.assertTrue(np.allclose(product[0][1].num, 0, atol=1e-8))
        self.assertTrue(np.allclose(product[1][0].num, 0, atol=1e-8))

    def test_not_invertible(self):
        # singular matrix: both rows equal
        H = _make_tf([1], [1, -0.5])
        G = [[H, H], [H, H]]
        G_inv, flag = oci.invert_mtf(G)
        self.assertEqual(flag, 1)

    def test_unstable_inverse(self):
        # G = (z-2)/1 => G^{-1} = 1/(z-2), pole at z=2 > 1 => unstable
        G = [[_make_tf([1, -2], [1])]]
        G_inv, flag = oci.invert_mtf(G)
        self.assertEqual(flag, 2)

class TestMultiplyTf(unittest.TestCase):
    def test_1x1(self):
        # scalar case
        G1 = [[_make_tf([1], [1, -1])]]   # 1/(z-1) = Cf
        G2 = [[_make_tf([1, -0.9], [1.25])]]  # (z-0.9)/1.25 = inv_C_til
        result = oci.multiply_tf(G1, G2)
        expected = _make_tf([1, -0.9], [1.25, -1.25])
        self.assertTrue(_tf_allclose(result[0][0], expected))

    def test_identity(self):
        # G * I = G
        G = [[_make_tf([2], [1, -0.5]), _make_tf([1], [1])],
             [_make_tf([0], [1]),        _make_tf([3], [1, -0.3])]]
        one  = _make_tf([1], [1])
        zero = _make_tf([0], [1])
        I = [[one, zero], [zero, one]]
        result = oci.multiply_tf(G, I)
        for i in range(2):
            for j in range(2):
                self.assertTrue(_tf_allclose(result[i][j], G[i][j]))

class TestSisoClosedLoop(unittest.TestCase):
    def test_simple_closed_loop(self):
        C = _make_tf([1], [1])
        G = _make_tf([1], [1, 0])

        result = oci.siso_closed_loop(C, G)

        expected = _make_tf([1], [1, 1])

        self.assertTrue(_tf_allclose(result, expected))

    def test_integrator_controller(self):
        C = _make_tf([2], [1, -1])
        G = _make_tf([1], [1, 0])

        result = oci.siso_closed_loop(C, G)

        expected = _make_tf([2], [1, -1, 2])

        self.assertTrue(_tf_allclose(result, expected))

    def test_same_order_num_den(self):
        C = _make_tf([1], [1, 1])
        G = _make_tf([1, 0], [1, 2])

        result = oci.siso_closed_loop(C, G)

        expected = _make_tf(
            [1, 0],
            [1, 4, 2]
        )

        self.assertTrue(_tf_allclose(result, expected))

if __name__ == "__main__":
    unittest.main()