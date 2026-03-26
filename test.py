import torch
x = torch.randn(1, 100, 12)
if x.dim() == 3:
    if x.shape[1] == 100 and x.shape[2] == 12:
        x = x.transpose(1, 2)
print(x.shape)
