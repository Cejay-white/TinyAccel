"""Deterministic randomized end-to-end regression coverage."""

import unittest

import numpy as np

import tinyaccel


class RandomizedRegressionTests(unittest.TestCase):
    def test_random_matmul_and_fused_graphs_match_numpy(self) -> None:
        rng = np.random.default_rng(20260812)

        for case in range(32):
            m, k, n = (int(value) for value in rng.integers(1, 25, size=3))
            tile_m, tile_k, tile_n = (
                int(value) for value in rng.integers(1, 10, size=3)
            )
            lhs_data = rng.standard_normal((m, k), dtype=np.float32)
            rhs_data = rng.standard_normal((k, n), dtype=np.float32)
            bias_data = rng.standard_normal((n,), dtype=np.float32)

            with self.subTest(case=case, shape=(m, k, n)):
                builder = tinyaccel.GraphBuilder()
                lhs = builder.input("lhs", (m, k))
                rhs = builder.input("rhs", (k, n))
                bias = builder.input("bias", (n,))
                product = builder.matmul(lhs, rhs)
                graph = builder.build(
                    builder.relu(builder.add(product, bias), name="result")
                )
                options = tinyaccel.CompileOptions(
                    tile_m=tile_m,
                    tile_n=tile_n,
                    tile_k=tile_k,
                    optimize=bool(case % 2),
                )
                executable = tinyaccel.compile(graph, options=options)

                actual = executable.run(lhs_data, rhs_data, bias_data)
                expected = np.maximum(lhs_data @ rhs_data + bias_data, 0)

                np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
                self.assertLessEqual(
                    executable.last_report.peak_sram_bytes,
                    executable.hardware.sram_bytes,
                )
                self.assertIsNotNone(executable.program.memory_plan)


if __name__ == "__main__":
    unittest.main()
