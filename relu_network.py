import numpy as np 

# This class defines a simple ReLU-based neural network with one hidden layer.
# It includes methods for forward propagation, backward propagation (gradient descent),
# and training the network. The class can be initialized with custom weights or with random 
# weights if none are provided. It uses the ReLU activation function for the hidden layer.

class ReLUNetwork:
    def __init__(self, in_dim, hidden_dim, out_dim, init_w1=None, init_w2=None):
        if init_w1 is not None and init_w2 is not None:
            self.W1 = init_w1.copy()
            self.W2 = init_w2.copy()
        else:
            self.W1 = np.random.randn(hidden_dim, in_dim)
            self.W2 = np.random.randn(out_dim, hidden_dim)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)
    
    def forward(self, x):
        self.a1 = self.W1 @ x
        self.z1 = self.relu(self.a1)
        self.z2 = self.W2 @ self.z1
        return self.z2
    
    def backward(self, x, y, learning_rate):
        m = x.shape[1]
        
        # Forward pass
        self.forward(x)
        
        # Backward pass
        dz2 = (self.z2 - y) / m
        dW2 = dz2 @ self.z1.T
        
        dz1 = (self.W2.T @ dz2) * self.relu_derivative(self.a1)
        dW1 = dz1 @ x.T
        
        # Update weights
        self.W2 -= learning_rate * dW2
        self.W1 -= learning_rate * dW1
    
    def train(self, X_train, Y_train, epochs, learning_rate):
        w1s = []
        w2s = []
        losses = []
        
        for epoch in range(epochs):
            loss = np.mean((self.forward(X_train) - Y_train) ** 2)
            losses.append(loss)
            w1s.append(self.W1.copy())
            w2s.append(self.W2.copy())
            self.backward(X_train, Y_train, learning_rate)
            # Uncomment to print loss every epoch
            # print(f"Epoch {epoch+1}/{epochs}, Loss: {loss}")
        
        return w1s, w2s, losses
