import torch

a = torch.tensor([[1,2],[3,4]])
b = torch.tensor([[1, 2, 3], [3, 2, 1]])
print(a)
print(b)
i = b.reshape(-1,1)
j = b.reshape(1,-1)
print(i)
print(j)
# x = i // 3
# y = i % 3
# print(b[x,y])
# print(a)
# m = torch.argmax(a)
# print(a[m])
# print(m.div(5, rounding_mode='floor'))
# print(m.remainder(4)+torch.tensor(1))
