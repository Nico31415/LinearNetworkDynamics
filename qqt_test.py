import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns
from tools import BlindColours
from utils import get_lambda_balanced, get_random_regression_task
from linear_network import LinearNetwork
from qqt_lambda_balanced import QQT_lambda_balanced


in_dim = 5
hidden_dim = 5
out_dim = 5

lmda = -50

batch_size = 10

learning_rate = 0.001
training_steps = 200
# learning_rate = 0.001 / lmda
# training_steps = int(200 * np.sqrt(lmda))

# init_w1, init_w2 = zero_balanced_weights(in_dim, hidden_dim, out_dim, 0.35)

init_w1, init_w2  = get_lambda_balanced(lmda, in_dim, hidden_dim, out_dim)


X, Y = get_random_regression_task(batch_size, in_dim, out_dim)

U_, S_, Vt_ = np.linalg.svd(Y @ X.T / batch_size)

model = LinearNetwork(in_dim, hidden_dim, out_dim, init_w1.copy(), init_w2.copy())
w1s, w2s, losses = model.train(X, Y, training_steps, learning_rate)
ws = np.array([w2 @ w1 for (w2, w1) in zip(w2s, w1s)])
ws = np.expand_dims(ws, axis=1)

analytical2 = QQT_lambda_balanced(init_w1.copy(), init_w2.copy(), X.T, Y.T, False)
analytical2 = np.asarray([analytical2.forward(learning_rate) for _ in range(training_steps)])

rep1 = [[w1.T @ w1] for w1 in w1s]
rep1_analytical = np.array([a[:in_dim, :in_dim] for a in analytical2])

rep2= [[w2 @ w2.T] for w2 in w2s]
rep2_analytical = np.array([a[in_dim:, in_dim:] for a in analytical2])

reps = (np.asarray(rep2)[:, 0, :, :])

plot_items_n = 4
blind_colours = BlindColours().get_colours()


outputs = (np.asarray(ws)[:, 0, :, :] @ X[:,:plot_items_n])



plt.figure()
reps = (np.asarray(rep2)[:, 0, :, :])
for color, output in zip(blind_colours, reps.T):
    for val in output:
        plt.plot(val, c=color, lw=2.5, label='Representation')
    plt.plot((rep2_analytical).reshape(training_steps, -1), lw=3., c="k", alpha=0.7, linestyle=(0, (1, 2)), label='analytical') # (0, (3, 4, 3, 1))
    
for color, target in zip(blind_colours, Y[:plot_items_n]):
    for value in target:
        plt.scatter(training_steps * 1.6, value, marker="_", color=color, lw=2.5)

plt.title(f'Representation Dynamics Lambda Balanced, Lambda: {lmda}')
plt.xlabel('Training Steps')
plt.ylabel('Network Representation (W2)')
plt.legend(['output', 'analytical'])


plt.figure()
reps = (np.asarray(rep1)[:, 0, :, :])
for color, output in zip(blind_colours, reps.T):
    for val in output:
        plt.plot(val, c=color, lw=2.5, label='Representation')
    plt.plot((rep1_analytical).reshape(training_steps, -1), lw=3., c="k", alpha=0.7, linestyle=(0, (1, 2)), label='analytical') # (0, (3, 4, 3, 1))
    
for color, target in zip(blind_colours, Y[:plot_items_n]):
    for value in target:
        plt.scatter(training_steps * 1.6, value, marker="_", color=color, lw=2.5)

plt.title(f'Representation Dynamics Lambda Balanced, Lambda: {lmda}')
plt.xlabel('Training Steps')
plt.ylabel('Network Representation (W1)')
plt.legend(['output', 'analytical'])




plt.figure()

analytical2 = [a[in_dim:, :in_dim] for a in analytical2]
for color, output in zip(blind_colours, outputs.T):
    for val in output:
        plt.plot(val, c=color, lw=2.5, label='output')
    plt.plot((analytical2 @ X[:,:plot_items_n]).reshape(training_steps, -1), lw=3., c="k", alpha=0.7, linestyle=(0, (1, 2)), label='analytical') # (0, (3, 4, 3, 1))
    
for color, target in zip(blind_colours, Y[:plot_items_n]):
    for value in target:
        plt.scatter(training_steps * 1.6, value, marker="_", color=color, lw=2.5)

plt.title(f'Learning Dynamics Lambda Balanced, Lambda: {lmda}')
plt.xlabel('Training Steps')
plt.ylabel('Network Output')
plt.legend(['output', 'analytical'])

plt.show()