import torch
v1=torch.tensor([1,2])
v2=torch.tensor([0,1])
result=torch.inner(v1,v2)
print(result)

result=torch.outer(v1,v2)
print(result)