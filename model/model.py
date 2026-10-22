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
        return self._norm(x.float()).type_as(x)*x*self.weight.type_as(x)

# class RMSNormBad(nn.Module):
#     def __init__(self, dim: int, eps: float = 1e-5):
#         super().__init__()
#         self.eps = eps
#         self.weight = nn.Parameter(torch.ones(dim))

#     def forward(self, x):
#         # ❌ 危险版：直接在半精度里面做pow(2)，容易溢出inf/NaN
#         r = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
#         return x * r * self.weight.type_as(x)


# torch.manual_seed(0)
# dim = 8
# norm_safe = RMSNorm(dim)
# norm_bad = RMSNormBad(dim)

# # 造偏大的数，很容易触发半精度平方溢出
# x_data = torch.tensor([[300.0, -180.0, 150.0, -120.0, 90.0, -60.0, 30.0, -10.0]])

# print("===== fp16 (float16) 测试 =====")
# x_fp16 = x_data.to(torch.float16)
# y_safe_fp16 = norm_safe(x_fp16)
# y_bad_fp16 = norm_bad(x_fp16)
# print("safe 输出:\n", y_safe_fp16)
# print("bad  输出:\n", y_bad_fp16)
# print("bad 是否含NaN/Inf:", torch.any(torch.isnan(y_bad_fp16) | torch.isinf(y_bad_fp16)))

# print("\n===== bf16 (bfloat16) 测试 =====")
# x_bf16 = x_data.to(torch.bfloat16)
# y_safe_bf16 = norm_safe(x_bf16)
# y_bad_bf16 = norm_bad(x_bf16)
# print("safe 输出:\n", y_safe_bf16)
# print("bad  输出:\n", y_bad_bf16)
# print("bad 是否含NaN/Inf:", torch.any(torch.isnan(y_bad_bf16) | torch.isinf(y_bad_bf16)))

# # 高精度参考真值
# x_ref = x_data.to(torch.float64)
# rms_ref = torch.sqrt(x_ref.pow(2).mean(-1, keepdim=True) + 1e-5)
# y_ref = x_ref / rms_ref
# print("\n===== float64 参考真值 =====")
# print(y_ref)
