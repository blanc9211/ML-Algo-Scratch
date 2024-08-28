import numpy as np
from loss_function import mean_squared_error, binary_cross_entropy, categorical_cross_entropy
from optimizers import  Adam
from activation_functions import relu, relu_derivative, sigmoid, sigmoid_derivative, tanh, tanh_derivative, softmax

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.001):
        self.hidden_sizes = hidden_sizes  # List of hidden layer neurons
        self.num_layers = len(hidden_sizes) + 1
        
        self.weights = []
        self.biases = []
        
        # Input layer to first hidden layer
        self.weights.append(np.random.randn(input_size, hidden_sizes[0]))
        self.biases.append(np.zeros((1, hidden_sizes[0])))
        
        # Hidden layers
        for i in range(1, len(hidden_sizes)):
            self.weights.append(np.random.randn(hidden_sizes[i-1], hidden_sizes[i]))
            self.biases.append(np.zeros((1, hidden_sizes[i])))
        
        # Last hidden layer to output layer
        self.weights.append(np.random.randn(hidden_sizes[-1], output_size))
        self.biases.append(np.zeros((1, output_size)))
        
        # Optimizer
        self.optimizer = Adam(learning_rate=learning_rate)
    
    def forward(self, X):
        self.a = [X]
        self.z = []
        
        # Forward pass through each layer
        for i in range(self.num_layers):
            z = np.dot(self.a[-1], self.weights[i]) + self.biases[i]
            self.z.append(z)
            a = relu(z) if i < self.num_layers - 1 else z
            self.a.append(a)
        
        return self.a[-1]
    
    def backward(self, X, y, y_pred):
        loss_grad = 2 * (y_pred - y) / y.shape[0]  # Gradient of MSE
        d_weights = []
        d_biases = []
        
        # Gradients for the output layer
        d_weights.append(np.dot(self.a[-2].T, loss_grad))
        d_biases.append(np.sum(loss_grad, axis=0, keepdims=True))
        
        # Backpropagation through hidden layers
        d_a = np.dot(loss_grad, self.weights[-1].T)
        for i in range(self.num_layers - 2, -1, -1):
            d_z = d_a * relu_derivative(self.z[i])
            d_weights.insert(0, np.dot(self.a[i].T, d_z))
            d_biases.insert(0, np.sum(d_z, axis=0, keepdims=True))
            d_a = np.dot(d_z, self.weights[i].T)
        
        # Update weights and biases
        self.optimizer.update(self.weights + self.biases, d_weights + d_biases)
    
    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward(X)
            # Compute loss
            loss = mean_squared_error(y, y_pred)
            # Backward pass and update
            self.backward(X, y, y_pred)
            
            if (epoch + 1) % 1000 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")
    
    def predict(self, X):
        return self.forward(X)
    
    def evaluate(self, X_test, y_test):
        # Predict on the test set
        y_pred = self.predict(X_test)
        # Compute loss
        loss = mean_squared_error(y_test, y_pred)
        print(f"Test Loss: {loss}")



## Train

np.random.seed(42)  # For reproducibility

X = np.random.randn(1000, 2) 
# Calculate z = 2x + y + 1
y = np.array([[2 * x[0] + x[1] + 1] for x in X]) 

nn = SimpleNeuralNetwork(input_size=X.shape[1], hidden_sizes=[8, 4] , output_size=1, learning_rate=0.001)
nn.train(X, y, epochs=10000)


## Evaluate Accuract

X_test = np.random.randn(1000, 2) 
y_test = np.array([[2 * x[0] + x[1] + 1] for x in X_test]) 
nn.evaluate(X_test, y_test)



## Simple test
new_X = np.array([[1.0, 2.0], [3.0, 4.0]]) 
predictions = nn.predict(new_X)
new_y = np.array([[2 * x[0] + x[1] + 1] for x in new_X])
print("Prediction, expected_output")
for x, y in zip(predictions, new_y):
    print(x, y)
