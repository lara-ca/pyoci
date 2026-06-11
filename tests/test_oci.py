import unittest
import oci
import vrft
import scipy.signal as signal
import numpy as np


class TestOciDesign(unittest.TestCase):
    
    def setUp(self):
        np.random.seed(42)
        self.N = 50
        self.u = np.random.choice([-1, 1], size=(self.N, 1))
        self.u[0] = 0
        
        self.G = signal.TransferFunction([1, 0], [1, -1.7, 0.72], dt=1)
        self.y = vrft.filter(self.G, self.u)
        self.y = np.array(self.y)
        
        self.a = 0.4
        self.Td = signal.TransferFunction([1 - self.a], [1, -self.a], dt=1)
        self.L = signal.TransferFunction([1], [1], dt=1)

        self.Cf_fixed_pi  = signal.TransferFunction([1], [1, -1], dt=1) 
        self.model_params_fixed_pi = [[[1, 2, 1]]]

        self.Cf_fixed_pid = signal.TransferFunction([1], [1, -1, 0], dt=1) 
        self.model_params_fixed_pid = [[[0, 2, 2]]]
        
    
    def test_oci_fixed_pi_design_numerator(self):
        C = oci.design(self.u, self.y, self.Td, self.Cf_fixed_pi, 'OE', self.model_params_fixed_pi, self.L)
        expected_num = np.array([0.6, -1.02, 0.432])
        np.testing.assert_array_almost_equal(C[0][0].num, expected_num)
    
    def test_oci_fixed_pi_design_denominator(self):
        C = oci.design(self.u, self.y, self.Td, self.Cf_fixed_pi, 'OE', self.model_params_fixed_pi, self.L)
        expected_den = np.array([1., -1., 0.])
        np.testing.assert_array_almost_equal(C[0][0].den, expected_den)

    def test_oci_fixed_pid_design_numerator(self):
        C = oci.design(self.u, self.y, self.Td, self.Cf_fixed_pid, 'OE', self.model_params_fixed_pid, self.L)
        expected_num = np.array([0.6, -1.02, 0.432])
        np.testing.assert_array_almost_equal(C[0][0].num, expected_num)
    
    def test_oci_fixed_pid_design_denominator(self):
        C = oci.design(self.u, self.y, self.Td, self.Cf_fixed_pid, 'OE', self.model_params_fixed_pid, self.L)
        expected_den = np.array([1., -1., 0.])
        np.testing.assert_array_almost_equal(C[0][0].den, expected_den)



if __name__ == "__main__":
    unittest.main()