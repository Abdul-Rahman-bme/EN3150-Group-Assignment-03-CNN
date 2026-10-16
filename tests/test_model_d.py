import unittest

import torch

from cnn_assignment.models import build_model


class TestResidualDepthwiseCNN(unittest.TestCase):

    def test_model_d_forward_shape(self):
        model = build_model("model_d")
        model.eval()

        x = torch.randn(4, 3, 64, 64)

        with torch.no_grad():
            output = model(x)

        self.assertEqual(tuple(output.shape), (4, 10))

    def test_model_d_parameter_count(self):
        model = build_model("model_d")

        parameters = sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )

        self.assertEqual(parameters, 15442)


if __name__ == "__main__":
    unittest.main()