# Import the PyTorch library
import torch  # PyTorch is used for tensor computations and deep learning

# Create tensors x and y with shape (2, 2) and float32 data type
x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)  # Tensor x: [[1, 2], [3, 4]]
y = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)  # Tensor y: [[5, 6], [7, 8]]

# Perform element-wise addition of x and y
add = x + y  # [[6, 8], [10, 12]]

# Perform element-wise multiplication of x and y
mul = x * y  # [[5, 12], [21, 32]]

# Perform matrix multiplication (dot product) of x and y
matmul = torch.matmul(x, y)  # [[19, 22], [43, 50]]

# Print the result of addition
print("Addition:\n", add)  # Display element-wise sum of tensors

# Print the result of element-wise multiplication
print("Element-wise Multiplication:\n", mul)  # Display element-wise product

# Print the result of matrix multiplication
print("Matrix Multiplication:\n", matmul)  # Display matrix multiplication result
