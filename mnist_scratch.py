import numpy as np
from mnist import get_mnist  # Make sure to 'pip install get-mnist'

# 1. Load and Preprocess Data
x_train, y_train, x_test, y_test = get_mnist('MNIST')
x_train = x_train.T / 255.0  # Normalize to [0, 1]
y_train = y_train.flatten()

# 2. Parameters Initialization
def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

# 3. Activation Functions
def ReLU(Z): return np.maximum(Z, 0)
def softmax(Z): return np.exp(Z) / np.sum(np.exp(Z), axis=0)

# 4. Forward Propagation
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

# 5. Backpropagation
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y.T

def backward_prop(Z1, A1, Z2, A2, W2, X, Y):
    m = Y.size
    dZ2 = A2 - one_hot(Y)
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * (Z1 > 0)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

# 6. Training Loop
def train(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W2, X, Y)
        W1 -= alpha * dW1; b1 -= alpha * db1
        W2 -= alpha * dW2; b2 -= alpha * db2
    np.save('weights_W1.npy', W1); np.save('weights_b1.npy', b1)
    print("Training Complete. Weights Saved.")