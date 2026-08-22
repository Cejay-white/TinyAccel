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

    def test_random_tiled_conv2d_matches_reference(self) -> None:
        rng = np.random.default_rng(20260813)

        for case in range(16):
            batch = int(rng.integers(1, 3))
            input_h, input_w = (int(value) for value in rng.integers(3, 10, size=2))
            input_c = int(rng.integers(1, 4))
            output_c = int(rng.integers(1, 6))
            kernel_h = int(rng.integers(1, min(4, input_h + 1)))
            kernel_w = int(rng.integers(1, min(4, input_w + 1)))
            stride = tuple(int(value) for value in rng.integers(1, 3, size=2))
            dilation = tuple(int(value) for value in rng.integers(1, 3, size=2))
            effective_h = (kernel_h - 1) * dilation[0] + 1
            effective_w = (kernel_w - 1) * dilation[1] + 1
            padding = (
                max(0, effective_h - input_h),
                int(rng.integers(0, 3)),
                max(0, effective_w - input_w),
                int(rng.integers(0, 3)),
            )
            builder = tinyaccel.GraphBuilder()
            input_value = builder.input(
                "input", (batch, input_h, input_w, input_c), layout="NHWC"
            )
            weight = builder.input(
                "weight",
                (kernel_h, kernel_w, input_c, output_c),
                layout="HWIO",
            )
            graph = builder.build(
                builder.conv2d(
                    input_value,
                    weight,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    name="output",
                )
            )
            executable = tinyaccel.compile(
                graph,
                options=tinyaccel.CompileOptions(
                    tile_h=int(rng.integers(1, 5)),
                    tile_w=int(rng.integers(1, 5)),
                    tile_oc=int(rng.integers(1, 4)),
                    tile_ic=int(rng.integers(1, 4)),
                ),
            )
            input_data = rng.standard_normal(input_value.type.shape, dtype=np.float32)
            weight_data = rng.standard_normal(weight.type.shape, dtype=np.float32)

            with self.subTest(case=case):
                actual = executable.run(input_data, weight_data)
                expected = tinyaccel.evaluate(graph, input_data, weight_data)
                np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
                self.assertLessEqual(
                    executable.last_report.peak_sram_bytes,
                    executable.hardware.sram_bytes,
                )


if __name__ == "__main__":
    unittest.main()
