import torch

x = torch.rand(2,3, dtype=torch.float32)
xq = torch.quantize_per_tensor(x, scale = 0.5, zero_point = 8, dtype=torch.quint8)
print(xq)
print(xq.int_repr())
