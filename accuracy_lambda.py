from utils import get_lambda_balanced, get_random_regression_task
from linear_network import LinearNetwork
import numpy as np
from qqt_lambda_balanced import QQT_lambda_balanced
import matplotlib.pyplot as plt

# Set training parameters
training_steps = 200
learning_rate = 0.0001

# Define a range of lambda values and network sizes
lmdas = np.logspace(np.log10(0.01), np.log10(100), 20)
network_sizes = [3, 5, 10, 20]

losses_analytical = dict()
losses_empirical = dict()
loss_deviations = dict()

# Set the number of trials for each experiment
num_tries = 20

# Function to calculate the difference between analytical and empirical losses
def loss_difference(loss_anal, loss_emp):
    return np.mean((loss_anal - loss_emp) / loss_emp)
    # return np.mean((loss_anal - loss_emp))

for network_size in network_sizes:

    # Set dimensions based on network size (equal input output hidden dimensions)
    in_dim = network_size 
    hidden_dim = network_size
    out_dim = network_size 

    batch_size = 2 * network_size 

    losses_empirical[network_size] = {}
    losses_analytical[network_size] = {}
    loss_deviations[network_size] = {}

    for lmda in lmdas:

        losses_empirical[network_size][lmda] = {}
        losses_analytical[network_size][lmda] = {}
        loss_deviations[network_size][lmda] = {}

        for n in range(num_tries):
            X, Y = get_random_regression_task(batch_size, in_dim, out_dim)
            init_w1, init_w2 = get_lambda_balanced(lmda, in_dim, hidden_dim, out_dim)

            model = LinearNetwork(in_dim, hidden_dim, out_dim, init_w1, init_w2)
            w1s, w2s, _ = model.train(X, Y, training_steps, learning_rate)

            # Calculate empirical loss
            loss = [1/(2 * batch_size) * np.linalg.norm(w2 @ w1 @ X - Y)**2 for (w1, w2) in zip(w1s, w2s)]
            losses_empirical[network_size][lmda][n] = loss

            # Calculate analytical loss using QQT_lambda_balanced
            analytical = QQT_lambda_balanced(init_w1.copy(), init_w2.copy(), X.T, Y.T, True)
            analytical = [analytical.forward(learning_rate) for _ in range(training_steps)]
            
            loss_analytical = np.array([1/(2*batch_size) * np.linalg.norm(w @ X - Y)**2 for w in analytical])
            losses_analytical[network_size][lmda][n] = loss_analytical

            # Calculate loss deviation
            loss_deviations[network_size][lmda][n] = loss_difference(loss_analytical, loss)

plt.figure(figsize=(10, 6))

for network_size in network_sizes:
    avg_deviations = []
    std_deviations = []
    for lmda in lmdas:
        trials = loss_deviations[network_size][lmda]
        avg_loss = np.mean(list(trials.values()))
        std_loss = np.std(list(trials.values()))
        avg_deviations.append(avg_loss)
        std_deviations.append(std_loss)
    
    avg_deviations = np.array(avg_deviations)
    std_deviations = np.array(std_deviations)
    plt.plot(lmdas, avg_deviations, linestyle='-', label=f'Network Size {network_size}')
    plt.fill_between(lmdas, avg_deviations - std_deviations, avg_deviations + std_deviations, alpha=0.2)

plt.xscale('log')
plt.xlabel('Lambda', fontsize=12)
plt.ylabel('Average Deviation', fontsize=12)
plt.title(f'Average Deviation vs Lambda for Different Network Sizes (learning rate = {learning_rate})', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()
