import torch
from torch import nn

model = nn.Sequential(
    nn.Linear(5, 32),
    nn.ReLU(),
    nn.Linear(32, 3)
)

model.load_state_dict(torch.load('ir_model.pt'))

model.eval()

def predict(sensor):
    input = torch.tensor([float(letter) for letter in sensor]).float().reshape(-1, 5)
    with torch.no_grad():
        logit = model(input)
        prob = torch.softmax(logit, dim=1)
    a, d, w = prob.numpy().flatten()
    return a, d, w

if __name__ == '__main__': # 직접f5를 눌렀을 때 실행되는 것
    print(predict("01111"))
    print(predict("11011"))
    print(predict("11110"))
    
    
