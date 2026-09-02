from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:  # The repository's standard Python need not include PyTorch.
    torch = None

if torch is not None:
    import syv_gptq_utils_adaptive as adaptive


@unittest.skipUnless(torch is not None, "PyTorch is required for numerical GPTQ tests")
class AdaptiveHessianTests(unittest.TestCase):
    def test_spd_hessian_uses_original_one_percent_damping(self) -> None:
        matrix = torch.eye(4, dtype=torch.float32)
        factorization, diagnostics = adaptive.stabilize_hessian(
            matrix, percdamp=0.01, module_name="mtp.fc"
        )
        self.assertEqual(diagnostics["chosen_damping_factor"], 1.0)
        self.assertEqual(diagnostics["attempts"], [{"factor": 1.0, "cholesky_info": 0}])
        self.assertTrue(torch.allclose(factorization @ factorization.t(), matrix))

    def test_finite_asymmetric_hessian_is_symmetrized(self) -> None:
        matrix = torch.tensor([[2.0, 3.0], [1.0, 2.0]], dtype=torch.float32)
        _factorization, diagnostics = adaptive.stabilize_hessian(
            matrix, percdamp=0.01, module_name="mtp.layers.0.mlp.down_proj"
        )
        self.assertEqual(diagnostics["asymmetry_max_abs"], 2.0)
        self.assertTrue(torch.allclose(matrix, matrix.t()))
        self.assertEqual(diagnostics["chosen_damping_factor"], 1.0)

    def test_small_negative_eigenvalue_selects_first_working_factor(self) -> None:
        matrix = torch.tensor([[1.0, 1.01], [1.01, 1.0]], dtype=torch.float32)
        _factorization, diagnostics = adaptive.stabilize_hessian(
            matrix, percdamp=0.01, module_name="mtp.layers.0.mlp.down_proj"
        )
        self.assertEqual(diagnostics["chosen_damping_factor"], 2.0)
        self.assertEqual([item["cholesky_info"] for item in diagnostics["attempts"]], [2, 0])

    def test_nan_and_inf_are_rejected_before_cholesky(self) -> None:
        for value in (float("nan"), float("inf")):
            matrix = torch.eye(2, dtype=torch.float32)
            matrix[0, 1] = value
            with self.assertRaises(adaptive.HessianError) as caught:
                adaptive.stabilize_hessian(matrix, percdamp=0.01, module_name="mtp.fc")
            self.assertEqual(caught.exception.code, "hessian_nonfinite")
            self.assertFalse(caught.exception.diagnostics["finite"])
            self.assertEqual(caught.exception.diagnostics["attempts"], [])

    def test_bounded_schedule_fails_without_rtn_fallback(self) -> None:
        matrix = torch.tensor([[1.0, 3.0], [3.0, 1.0]], dtype=torch.float32)
        with self.assertRaises(adaptive.HessianError) as caught:
            adaptive.stabilize_hessian(matrix, percdamp=0.01, module_name="mtp.fc")
        self.assertEqual(caught.exception.code, "hessian_not_positive_definite")
        self.assertEqual(
            [item["factor"] for item in caught.exception.diagnostics["attempts"]],
            list(adaptive.DAMPING_FACTORS),
        )
        self.assertTrue(all(item["cholesky_info"] != 0 for item in caught.exception.diagnostics["attempts"]))


if __name__ == "__main__":
    unittest.main()
