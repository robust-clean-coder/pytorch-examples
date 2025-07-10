import torch

# Create tensors
x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
y = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)

# Basic operations
add = x + y
mul = x * y
matmul = torch.matmul(x, y)

print("Addition:\n", add)
print("Element-wise Multiplication:\n", mul)
print("Matrix Multiplication:\n", matmul)
