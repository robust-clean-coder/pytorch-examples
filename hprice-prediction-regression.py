# Import PyTorch library
import torch  # PyTorch core module

# Import neural network module from PyTorch
import torch.nn as nn  # For building neural networks

# Import optimization module from PyTorch
import torch.optim as optim  # For optimization algorithms

# Generate dummy input data: 100 samples, each with 3 features
X = torch.randn(100, 3)  # Input tensor of shape (100, 3)

# Generate target values using a known linear relationship
y = X @ torch.tensor([[2.0], [-1.0], [0.5]]) + 0.3  # True weights + bias

# Define a simple linear regression model with 3 input features and 1 output
model = nn.Linear(3, 1)  # Linear layer

# Mean Squared Error loss function for regression
criterion = nn.MSELoss()  # Loss function

# Stochastic Gradient Descent optimizer with learning rate 0.01
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Optimizer

# Training loop for 50 epochs
for epoch in range(50):  # Iterate over epochs
    # Forward pass: compute model predictions
    pred = model(X)  # Model output

    # Compute loss between predicted and actual values
    loss = criterion(pred, y)  # MSE loss

    # Zero out previous gradients before backpropagation
    optimizer.zero_grad()  # Reset gradients

    # Backward pass: compute gradients
    loss.backward()  # Backpropagation

    # Update model parameters using computed gradients
    optimizer.step()  # Optimizer step

# Print learned model weights
print("Learned weights:", model.weight.data)  # Final weights

# Print learned model bias
print("Learned bias:", model.bias.data)  # Final bias
