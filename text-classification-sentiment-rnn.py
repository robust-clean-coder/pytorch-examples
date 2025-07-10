import torch
import torch.nn as nn

# Dummy input: batch of 2 sentences, each with 5 tokens (embedded size 10)
input_data = torch.randn(2, 5, 10)

# Define simple RNN-based model
class SentimentRNN(nn.Module):
    def __init__(self):
        super(SentimentRNN, self).__init__()
        self.rnn = nn.RNN(input_size=10, hidden_size=20, batch_first=True)
        self.fc = nn.Linear(20, 2)  # 2 classes (positive/negative)

    def forward(self, x):
        out, _ = self.rnn(x)
        last_out = out[:, -1, :]
        return self.fc(last_out)

model = SentimentRNN()
outputs = model(input_data)
print("Logits:", outputs)
