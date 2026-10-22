import debugpy
try:
    # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
    debugpy.listen(("localhost", 9501))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
except Exception as e:
    pass

import torch

t=torch.rsqrt(torch.tensor(4.0))
print(t)

t2=torch.ones(3,4)
print(t2)
