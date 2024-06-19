# Import necessary libraries and modules
import matplotlib.pyplot as plt 
import numpy as np
from utils import get_random_regression_task

# Set dimensions for input, hidden, and output layers
in_dim = 2
hidden_dim = 5
out_dim = 7

# Set batch size
batch_size = 10

X, Y = get_random_regression_task(batch_size, in_dim, out_dim)

# Calculate input output correlation
sigma_yx = 1 / batch_size * Y @ X.T 

# Set lambda value
lmda = 10

# Function to construct matrix F based on lambda
def get_F(lmda):
    F = np.vstack([
        np.hstack([-lmda / 2 * np.eye(sigma_yx.shape[1]), sigma_yx.T]),
        np.hstack([sigma_yx, lmda / 2 * np.eye(sigma_yx.shape[0])])
    ]) 
    return F 

F = get_F(lmda)

U, S, Vt = np.linalg.svd(sigma_yx)
V = Vt.T

# Calculate intermediate matrices X and A (see thesis report for more details)
X = (np.sqrt(lmda**2 + 4 * S**2) - 2 * S) / lmda
A = np.diag(1 / (np.sqrt(1 + X**2)))

X = np.diag(X)

# Calculate the dimension difference between input and output dimensions
dim_diff = np.abs(in_dim - out_dim)

# If input and output dimensions are equal, construct O_final matrix
if dim_diff == 0:
    O_final = 1 / np.sqrt(2) * np.vstack([
        np.hstack([Vt.T @ (A - X @ A), Vt.T @ (A + X @ A)]),
        np.hstack([U @ (A + X @ A), -U @ (A - X @ A)])
    ])

# Handle cases where input and output dimensions are different
if in_dim < out_dim:
    U_hat = U[:, in_dim:]
    V_hat = np.zeros((in_dim, dim_diff))
    U = U[:, :in_dim]

elif in_dim > out_dim:
    U_hat = np.zeros((out_dim, dim_diff))
    V_hat = V[:, out_dim:]
    V = V[:, :out_dim]

else:
    U_hat = None 
    V_hat = None

Vt = V.T

# Construct O_extradim matrix for cases with different input and output dimensions
O_extradim = 1 / np.sqrt(2) * np.vstack([
    np.hstack([Vt.T @ (A - X @ A), Vt.T @ (A + X @ A), np.sqrt(2) * V_hat]),
    np.hstack([U @ (A + X @ A), -U @ (A - X @ A), np.sqrt(2) * U_hat])
])

# Calculate eigenvalues based on lambda and singular values
evals = np.sqrt(lmda**2 / 4 + S**2)

# Construct diagonal matrix D with eigenvalues
D = np.diag(np.concatenate((evals, -evals, np.sign(out_dim - in_dim) * lmda / 2 * np.ones(dim_diff))))

# Print statement for debugging
print('Print Statement For Debugging')

# Optional commented-out code for additional calculations and verification
# O1 = 1/np.sqrt(2) * np.vstack([
#     np.hstack([Vt.T, Vt.T]),
#     np.hstack([U, -U])
# ])

# theoretical_diag = np.vstack([
#     np.hstack([np.diag(S), -lmda/2 * np.eye(S.shape[0])]),
#     np.hstack([-lmda/2 * np.eye(S.shape[0]), -np.diag(S)])
# ])

# X = (np.sqrt(lmda**2 + 4*S**2) - 2* S)/lmda
# A = np.diag(1 / (np.sqrt(1 + X**2)))

# X = np.diag(X)

# P = np.vstack([
#     np.hstack([A, X @ A]),
#     np.hstack([-X @ A, A])
# ])

# O_final = 1/np.sqrt(2) * np.vstack([
#     np.hstack([Vt.T @ (A - X@ A), Vt.T @ (A + X@A)]),
#     np.hstack([U @ (A + X @ A), - U @ (A - X @ A)])
# ])

# evals = np.sqrt(lmda**2/4 + S**2)

# D = np.diag(np.concatenate((evals, -evals)))

# print(P.T @ O1.T @ F @ O1 @ P)
