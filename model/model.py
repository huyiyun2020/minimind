import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self,dim:int,eps:float=1e-5):
        super().__init__()
        self.eps=eps
        self.weight=nn.Parameter(torch.ones(dim))

    def _norm(self,x):
        return x.pow(2).mean(-1,keepdim=True).add(self.eps).rsqrt()

    def forward(self,x):
        output=self._norm(x.float())*x
        return output*self.weight.type_as(x)