# CSAE
Implementation of the CSAE model for the corresponding publication xyz.

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
