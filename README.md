# CSAE
Implementation of the CSAE model for the corresponding publication xyz.

![badge](https://github.com/lukasfolle/CSAE/actions/workflows/python-package.yml/badge.svg)

## Get started
1.  `pip install -r requirements.txt`
2.  `git clone https://github.com/lukasfolle/CSAE.git && cd CSAE`
3.  Start using the code with an example below!

## Example usage

```
import torch
from model import CSAE


model = CSAE(num_classes=3, input_dim=(256, 256, 32))

x = torch.rand((1, 1, 256, 256 , 32))

y_hat, x_hat = model(x)

print(y_hat)
# >> tensor([[-0.0065,  0.0039,  0.0336]], grad_fn=<AddmmBackward>)
print(x.shape)
# >> torch.Size([1, 1, 256, 256, 32])
print(x_hat.shape)
# >> torch.Size([1, 1, 256, 256, 32])
```

## 2D/3D Input dimensionality
- To use the network for images (2D) change `DIMENSION` in model.py:19 to `DIMENSION = 2`
- To use the network for volumes (3D) change (keep) `DIMENSION` in model.py:19 to `DIMENSION = 3`
