import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def initialize_parameters(n_features, n_classes):
    W = np.random.randn(n_features, n_classes) * 0.01
    b = np.zeros((n_classes, 1))
    return W, b

def forward_propagation(X, W, b):
    Z = np.dot(W.T, X) + b
    A = sigmoid(Z)
    return A

def compute_cost(A, Y):
    m = Y.shape[1]
    logloss = np.dot(Y, np.log(A).T) + np.dot(1 - Y, np.log(1 - A).T)
    cost = -1/m * np.sum(logloss)
    return np.squeeze(cost)

def backward_propagation(X, A, Y):
    m = X.shape[1]
    dW = 1/m * np.dot(X, (A - Y).T)
    db = 1/m * np.sum(A - Y, axis=1, keepdims=True)
    return dW, db

def update_parameters(W, b, dW, db, learning_rate):
    W = W - learning_rate * dW
    b = b - learning_rate * db
    return W, b

def logistic_regression(X, Y, num_iterations=3000, learning_rate=0.1, print_cost=False):
    n_features, m = X.shape
    n_classes = Y.shape[0]
    
    W, b = initialize_parameters(n_features, n_classes)
    
    for i in range(num_iterations):
        A = forward_propagation(X, W, b)
        cost = compute_cost(A, Y)
        dW, db = backward_propagation(X, A, Y)
        W, b = update_parameters(W, b, dW, db, learning_rate)
        
        if print_cost and i % 100 == 0:
            print(f"Cost after iteration {i}: {cost}")
    
    params = {"W": W, "b": b}
    return params

def predict(X, params):
    W = params["W"]
    b = params["b"]
    A = forward_propagation(X, W, b)
    return np.argmax(A, axis=0)