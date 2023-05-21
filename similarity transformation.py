import numpy as np
import matplotlib.pyplot as plt

# Original shape
square = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1], [1, 1]]).T

# Similarity transformation (rotation + scaling)
theta = np.pi / 4  # 45 degree rotation
scale = 1.5  # scale up by 1.5
transform = scale * np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

# Apply transformation
transformed_square = np.dot(transform, square)

# Plot
fig,ax = plt.subplots(1,1,dpi=200)
ax.plot(square[0, :], square[1, :], label='Original shape')
ax.plot(transformed_square[0, :], transformed_square[1, :], label='Transformed shape')
ax.legend(loc='upper left')
ax.grid()

plt.show()


import numpy as np

# Define the system in canonical form
A = np.array([[0, 1], [-3, -2]])
B = np.array([[0], [1]])
C = np.array([[2, 1]])
D = np.array([[0]])

# Define the transformation matrix
T = np.array([[1, 1], [0, 1]])

# Calculate the inverse of T
T_inv = np.linalg.inv(T)

# Perform the similarity transformation
A_prime = np.dot(T_inv, np.dot(A, T))
B_prime = np.dot(T_inv, B)
C_prime = np.dot(C, T)
# D_prime is the same as D for this transformation
D_prime = D

print("A' = \n", A_prime)
print("B' = \n", B_prime)
print("C' = \n", C_prime)
print("D' = \n", D_prime)

