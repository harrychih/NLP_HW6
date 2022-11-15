import torch
from logsumexp_safe import logsumexp_new

x = torch.tensor([1])
y = torch.rand((5,4))
print(y)
print(logsumexp_new(y, dim=0, safe_inf=True))