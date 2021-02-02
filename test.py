import torch
import torch.nn as nn
import math
n=160
criterion = nn.CrossEntropyLoss()
output = torch.randn(n, 5, requires_grad=True)
label = torch.empty(n, dtype=torch.long).random_(5)
loss = criterion(output, label)

print("网络输出为3个5类:")
print(output)
print("要计算loss的类别:")
print(label)
print("计算loss的结果:")
print(loss)

first = [0]*n
for i in range(n):
    first[i] = -output[i][label[i]]
second = [0]*n
for i in range(n):
    for j in range(5):
        second[i] += math.exp(output[i][j])
res = 0
for i in range(n):
    res += (first[i] + math.log(second[i]))
print("自己的计算结果：")
print(res /n)