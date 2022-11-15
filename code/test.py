import torch

x = torch.tensor([1])
y = torch.rand((4,1))
print(x)
print(y)
print(y.add(x).reshape(4,))