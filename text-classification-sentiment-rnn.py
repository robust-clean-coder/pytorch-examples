import torch  # Import PyTorch for tensor operations and model building
import torch.nn as nn  # Import the neural network module

# Dummy input: batch of 2 sentences, each with 5 tokens, each token represented by a 10-dimensional embedding
input_data = torch.randn(2, 5, 10)  # Shape: (batch_size=2, sequence_length=5, embedding_size=10)

# Define a simple RNN-based model for sentiment classification
class SentimentRNN(nn.Module):
    def __init__(self):
        super(SentimentRNN, self).__init__()  # Call the parent constructor
        self.rnn = nn.RNN(input_size=10, hidden_size=20, batch_first=True)  # RNN layer with 10 input features and 20 hidden units
        self.fc = nn.Linear(20, 2)  # Fully connected layer to output 2 classes (e.g., positive or negative)

    def forward(self, x):
        out, _ = self.rnn(x)  # Pass input through RNN, ignore hidden state
        last_out = out[:, -1, :]  # Get the output from the last time step for each sequence
        return self.fc(last_out)  # Pass the last output through the fully connected layer

model = SentimentRNN()  # Create an instance of the model
outputs = model(input_data)  # Forward pass with dummy input
print("Logits:", outputs)  # Print raw output scores (logits) for each class

