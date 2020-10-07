import torch

x = torch.zeros(2, 1, 4)

a = torch.squeeze(x, dim = 0)

print(a.shape)
x_list = torch.split(x, 1)
print(x_list[0].shape)

y = torch.cat(tuple(x), dim = 1)
print(y.shape)

z = y.permute(1,0)

print(z.shape)

q = torch.cat((x,x), dim = 2)
print(q.shape)

a = torch.ones(2, 1, 4)

print((x+a).shape)
