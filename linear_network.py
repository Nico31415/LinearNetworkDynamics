import numpy as np
from utils import get_lambda_balanced

class LinearNetwork:
    def __init__(self, in_dim, hidden_dim, out_dim, init_w1 = None, init_w2 = None):

        if init_w1 is not None and init_w2 is not None:
            self.W1 = init_w1.copy()
            self.W2 = init_w2.copy()
        else:
            self.W1 = np.random.randn(hidden_dim, in_dim)
            self.W2 = np.random.randn(out_dim, hidden_dim)

    def forward(self, x): 
        self.z = self.W2 @ self.W1 @ x
        return self.z

    def backward(self, x, y, learning_rate):

        forward = self.W2 @ self.W1 @ x
        dW1 = 1/x.shape[1] * self.W2.T @ (forward-y) @ x.T 
        dW2 = 1/x.shape[1] * (forward - y) @ x.T @ self.W1.T 

        self.W2 -= learning_rate * dW2
        self.W1 -= learning_rate * dW1


    def train(self, X_train, Y_train, epochs, learning_rate):
        w1s = []
        w2s = []
        losses = []
        for _ in range(epochs):
            loss = np.mean((self.forward(X_train) - Y_train) ** 2)
            losses.append(loss)
            w1s.append(self.W1.copy())
            w2s.append(self.W2.copy())
            self.backward(X_train, Y_train, learning_rate)

        return w1s, w2s, losses
    


in_dim = 8
hidden_dim = 8 
out_dim = 8

learning_rate = 0.01
training_steps = 2000

X = np.eye(8)
Y = np.asarray([
            [1.,  1.,  1., -0.,  1., -0., -0., -0.],
            [1.,  1.,  1., -0., -1., -0., -0., -0.],
            [1.,  1., -1., -0., -0.,  1., -0., -0.],
            [1.,  1., -1., -0., -0., -1., -0., -0.],
            [1., -1., -0.,  1., -0., -0.,  1., -0.],
            [1., -1., -0.,  1., -0., -0., -1., -0.],
            [1., -1., -0., -1., -0., -0., -0.,  1.],
            [1., -1., -0., -1., -0., -0., -0., -1.]
        ])


init_w1, init_w2 = get_lambda_balanced(-100, in_dim, hidden_dim, out_dim)

model = LinearNetwork(in_dim, hidden_dim, out_dim, init_w1.copy(), init_w2.copy())

w1s, w2s, _ = model.train(X, Y, training_steps, learning_rate)

w1w1s = [w1.T @ w1 for w1 in w1s]
w2w2s = [w2 @ w2.T for w2 in w2s]
print('print statement for debugging')