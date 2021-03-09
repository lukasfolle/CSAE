from model import CSAE

cae = CSAE(num_classes=3, input_dim=(256, 256, 32))
import numpy as np

x = torch.from_numpy(np.random.rand(1, 1, 256, 256, 32).astype(np.float32))
y_hat, x_hat = cae(x)
print(y_hat)
print(x.shape)
print(x_hat.shape)
