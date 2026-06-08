import torch
from torch import nn

class SeqIrModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.rnn = nn.LSTM(input_size=5, hidden_size=32, batch_first=True, num_layers=2)
        self.lin = nn.Linear(32, 3)

    def forward(self, input):
        #input shape (N, 6, 5)
        y, _ = self.rnn(input) #y shape (N, 6, 32)       
        last = y[:, -1, :] #last shape (N, 32)
        z = self.lin(last)
        return z
    
model = None

def load():
    global model
    model = SeqIrModel()
    model.load_state_dict(torch.load('seq_ir_model.pt'))
    model.eval()

def predict(sensor):
    x = [float(letter) for letter in sensor]
    
    # 30글자 안 되면 앞을 1로 채우기
    if len(x) < 30:
        x = [1.0] * (30 - len(x)) + x
    
    input = torch.tensor(x).float().reshape(1, 6, 5)
    with torch.no_grad():
        logit = model(input)
        prob = torch.softmax(logit, dim=1)
    a, d, w = prob.numpy().flatten()
    return a, d, w

if __name__ == '__main__':
    model = SeqIrModel()          # ← 추가
    load()                         # ← 추가
    
    print("왼쪽:", predict("01111" * 6))
    print("오른쪽:", predict("11110" * 6))
    print("직진:", predict("11011" * 6))
