from unittest import TestCase

import numpy as np
import torch

from model import CSAE

class TestCSAE(TestCase):
    def setUp(self) -> None:
        self.model = CSAE(num_classes=3, input_dim=(256, 256, 32))

    def test_forward_y_hat(self):
        x = torch.from_numpy(np.random.rand(1, 1, 256, 256, 32).astype(np.float32))
        y_hat, x_hat = self.model(x)
        self.assertEqual(y_hat.shape, torch.Size([1, 3]))

    def test_forward_x_hat(self):
        x = torch.from_numpy(np.random.rand(1, 1, 256, 256, 32).astype(np.float32))
        y_hat, x_hat = self.model(x)
        self.assertEqual(x_hat.shape, torch.Size([1, 1, 256, 256, 32]))
