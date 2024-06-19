import numpy as np
import matplotlib.pyplot as plt
from utils import get_random_regression_task, get_lambda_balanced
from qqt_lambda_balanced import QQT_lambda_balanced
from linear_network import LinearNetwork

training_steps = 200
fixed_k = 0.01  # Set a fixed k value
lambda_values = np.logspace(np.log10(50), np.log10(200),20)
# network_sizes = [3, 5, 10, 20]

network_sizes  = [5, 10, 15, 20]

losses_analytical = dict()
losses_empirical = dict()
loss_deviations = dict()

num_tries =20

def loss_difference(loss_anal, loss_emp):
    return np.mean((loss_anal - loss_emp) / loss_emp)
    # return np.mean((loss_anal - loss_emp))

for network_size in network_sizes:

    in_dim = network_size 
    hidden_dim = network_size
    out_dim = network_size 

    batch_size = 2 * network_size 

    losses_empirical[network_size] = {}
    losses_analytical[network_size] = {}
    loss_deviations[network_size] = {}

    for lmda in lambda_values:
        learning_rate = fixed_k / lmda

        losses_empirical[network_size][lmda] = {}
        losses_analytical[network_size][lmda] = {}
        loss_deviations[network_size][lmda] = {}

        for n in range(num_tries):
            X, Y = get_random_regression_task(batch_size, in_dim, out_dim)
            init_w1, init_w2 = get_lambda_balanced(lmda, in_dim, hidden_dim, out_dim)

            model = LinearNetwork(in_dim, hidden_dim, out_dim, init_w1, init_w2)
            w1s, w2s, _ = model.train(X, Y, training_steps, learning_rate)

            loss = [1/(2 * batch_size) * np.linalg.norm(w2 @ w1 @ X - Y)**2 for (w1, w2) in zip(w1s, w2s)]
            losses_empirical[network_size][lmda][n] = loss

            analytical = QQT_lambda_balanced(init_w1.copy(), init_w2.copy(), X.T, Y.T, True)
            analytical = [analytical.forward(learning_rate) for _ in range(training_steps)]
            
            loss_analytical = np.array([1/(2*batch_size) * np.linalg.norm(w @ X - Y)**2 for w in analytical])
            losses_analytical[network_size][lmda][n] = loss_analytical

            loss_deviations[network_size][lmda][n] = loss_difference(loss_analytical, loss)

plt.figure(figsize=(10, 6))

for network_size in network_sizes:
    avg_deviations = []
    std_deviations = []
    for lmda in lambda_values:
        trials = loss_deviations[network_size][lmda]
        avg_loss = np.mean(list(trials.values()))
        std_loss = np.std(list(trials.values()))
        avg_deviations.append(avg_loss)
        std_deviations.append(std_loss)
    
    avg_deviations = np.array(avg_deviations)
    std_deviations = np.array(std_deviations)
    plt.plot(lambda_values, avg_deviations, linestyle='-', label=f'Network Size {network_size}')
    plt.fill_between(lambda_values, avg_deviations - std_deviations, avg_deviations + std_deviations, alpha=0.2)

plt.xscale('log')
plt.xlabel('Lambda', fontsize=12)
plt.ylabel('Average Deviation', fontsize=12)
plt.title(f'Average Deviation vs Lambda for Different Network Sizes (k = {fixed_k})', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()