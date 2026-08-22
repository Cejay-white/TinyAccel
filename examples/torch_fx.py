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
            self.weight = torch.nn.Parameter(torch.randn(5, 4))
            self.bias = torch.nn.Parameter(torch.randn(4))

        def forward(self, value):
            return torch.relu(value @ self.weight + self.bias)


    torch.manual_seed(79)
    module = TinyMlp().eval()
    example = torch.randn(3, 5)
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

    print("Imported TinyAccel IR")
    print("---------------------")
    print(graph)
    print()
    print(executable.report())
    print()
    print("PyTorch correctness: PASS")
