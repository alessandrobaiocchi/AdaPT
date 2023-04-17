from model import CrossAttention
import torch
import torch.nn.functional as F



crossatt = CrossAttention(128,64,64,3)
x = torch.randn(1, 10, 128)

output, attn = crossatt(x)

print(attn)
hard = torch.argmax(attn, dim=-1)
print(hard)
print(output.shape)

gumbatt = F.gumbel_softmax(torch.log(attn), hard=True, tau=10e-8)
print(gumbatt+attn)