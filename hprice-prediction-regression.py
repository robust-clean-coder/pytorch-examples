import torch
import torch.nn as nn
import torch.optim as optim

# Dummy data: 100 samples, 3 features
X = torch.randn(100, 3)
y = X @ torch.tensor([[2.0], [-1.0], [0.5]]) + 0.3  # Linear relation + bias

# Define model
model = nn.Linear(3, 1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(50):
    pred = model(X)
    loss = criterion(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Learned weights:", model.weight.data)
print("Learned bias:", model.bias.data)
