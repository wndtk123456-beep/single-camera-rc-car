import torch
from torch import nn
import json
import numpy as np


rawdata = json.load(open('seq_ir_data.json'))

# N  T  C
# N, 6, 5

counter = [0, 0, 0]
dataset = []
for x, y in rawdata:
    x = [float(letter) for letter in x]
    x = torch.tensor(x) # shape(5, )

    if len(x) < 6 * 5:
        dst= torch.zeros(6*5)
        dst[-len(x):] = x
        x = dst
    x = x.reshape(6, 5) # shape(6, 5)

    l, r = y.split(',')
    if l == "0" and r != "0":
        label = 0
    elif l != "0" and r == "0":
        label = 1
    else:
        label = 2
    dataset.append((x, label))
    counter[label] += 1

print(dataset)
print(counter)



loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=50, drop_last=True)
for x, y in loader:
    break
print(x, y)
print(x.dtype, y.dtype)



import seq_ir_model
model = seq_ir_model.SeqIrModel()

opt = torch.optim.Adam(model.parameters())
weights = torch.tensor([617/60, 617/99, 1.0])
loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

for epoch in range(300):
    for x, y in loader:
        logit = model(x)
        loss = loss_fn(logit, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        print('\r', (logit.argmax(axis=1) == y).float().mean(), end ='')
    print("    ", loss)

print(loss)
torch.save(model.state_dict(), 'seq_ir_model.pt')






