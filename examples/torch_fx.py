"""Trace a small PyTorch module into TinyAccel when PyTorch is installed."""

import numpy as np

import tinyaccel


try:
    import torch
except ModuleNotFoundError:
    torch = None


if torch is None:
    print("PyTorch is optional; run `python -m pip install -e .[torch]` first.")
else:

    class TinyMlp(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(5, 4)
            self.activation = torch.nn.ReLU()

        def forward(self, value):
            return self.activation(self.linear(value))


    class TinyConv(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.convolution = torch.nn.Conv2d(
                2,
                3,
                kernel_size=(3, 2),
                stride=(2, 1),
                padding=(1, 0),
            )
            self.activation = torch.nn.ReLU()

        def forward(self, value):
            return self.activation(self.convolution(value))


    def run_example(label, module, example) -> None:
        module = module.eval()
        graph = tinyaccel.trace_torch_module(module, example)
        executable = tinyaccel.compile(graph)

        torch_result = module(example).detach().cpu().numpy()
        tinyaccel_result = executable.run(example.detach().cpu().numpy())
        np.testing.assert_allclose(
            tinyaccel_result,
            torch_result,
            rtol=1e-5,
            atol=1e-5,
        )

        print(f"{label} TinyAccel IR")
        print("-" * (len(label) + 13))
        print(graph)
        print()
        print(executable.report())
        print()
        print(f"{label} PyTorch correctness: PASS")


    torch.manual_seed(79)
    run_example("Linear", TinyMlp(), torch.randn(3, 5))
    print()
    run_example("Conv2d", TinyConv(), torch.randn(1, 2, 5, 6))
