from utils import get_random_regression_task, get_lambda_balanced
from linear_network import LinearNetwork
import numpy as np 
import matplotlib.pyplot as plt

# Set dimensions for input, hidden, and output layers (square network)
in_dim = 5
hidden_dim = 5
out_dim = 5

batch_size = 10

# Set lambda value
lmda = 1

# Set learning rate and number of epochs
learning_rate = 0.01
epochs = 1000

init_w1, init_w2 = get_lambda_balanced(lmda, in_dim, hidden_dim, out_dim)

X, Y = get_random_regression_task(batch_size, in_dim, out_dim)

# Calculate input output correlation
sigma_yx = 1/batch_size * Y @ X.T

model = LinearNetwork(in_dim, hidden_dim, out_dim, init_w1.copy(), init_w2.copy())

w1s, w2s, _ = model.train(X, Y, epochs, learning_rate)

losses = [1/(2 * batch_size) * np.linalg.norm(w2 @ w1 @ X - Y)**2 for (w1, w2) in zip(w1s, w2s)]

w2w1s = [w2 @ w1 for (w1, w2) in zip(w1s, w2s)]

# Calculate the derivative of  empirical loss 
ddt_losses = np.diff(losses)

# Calculate the derivative of the output w2w1
ddt_w2w1s = [(a - b) for (a, b) in zip(w2w1s, w2w1s[1:])]

# Calculate analytical rate of loss (formula from thesis report)
analytical = [-np.trace(ddt_w2w1 @ (w2w1.T - sigma_yx.T)) for (w2w1, ddt_w2w1) in zip(w2w1s, ddt_w2w1s)]

# Print statement for debugging
print('print statement for debugging')

# Plot the empirical and analytical rate of loss
plt.figure(figsize=(10, 6))
plt.plot(ddt_losses, 'b-', label='Empirical Rate of Loss')  # Blue solid line
plt.plot(analytical, 'gray', linestyle='--', label='Analytical Rate of Loss')  # Grey dashed line
plt.xlabel('Epochs')
plt.ylabel('Rate of Loss')
plt.title('Empirical and Analytical Rate of Loss')
plt.legend()
plt.grid(True)
plt.show()
