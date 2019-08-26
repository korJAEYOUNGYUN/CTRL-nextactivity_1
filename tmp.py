import torch
import torch.nn as nn

loss = nn.CrossEntropyLoss()
input = torch.randn(1, 16, requires_grad=True)
print(input.shape)
target = torch.empty(1, dtype=torch.long).random_(16)
print(target.shape)

output= loss(input, target)