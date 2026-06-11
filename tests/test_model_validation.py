import unittest
from oci import validate_model_inputs, unpack_model_params


class TestValidateModelInputs(unittest.TestCase):
    """Testes para a função validate_model_inputs"""

    def test_validate_valid_input(self):
        validate_model_inputs("OE", [[[1, 1, 1]]], ny=1, nu=1)

    def test_validate_invalid_model(self):
        with self.assertRaises(ValueError):
            validate_model_inputs("XPTO", [[[1, 1, 1]]], ny=1, nu=1)

    def test_validate_invalid_dimensions(self):
        with self.assertRaises(ValueError):
            validate_model_inputs("OE", [[[1, 1, 1]]], ny=2, nu=1)

    def test_validate_invalid_structure(self):
        with self.assertRaises(TypeError):
            validate_model_inputs("OE", "errado", ny=1, nu=1)


class TestUnpackModelParams(unittest.TestCase):
    """Testes para a função unpack_model_params"""

    def test_unpack_siso(self):
        result = unpack_model_params("OE", [[[1, 2, 3]]])
        self.assertEqual(result, {"nb": [[1]], "nf": [[2]], "nk": [[3]]})

    def test_unpack_mimo(self):
        params = [
            [[1, 1, 1], [2, 2, 0]],
            [[3, 3, 1], [4, 4, 2]],
        ]
        result = unpack_model_params("OE", params)
        self.assertEqual(result["nb"], [[1, 2], [3, 4]])
        self.assertEqual(result["nk"], [[1, 0], [1, 2]])


if __name__ == "__main__":
    unittest.main()