import numpy as np

class MLP:
    """
    A simple Multi-Layer Perceptron (MLP) for binary classification.

    Parameters
    ------------
    input_size : int
        Number of input features.
    hidden_size : int
        Number of neurons in the hidden layer.
    output_size : int
        Number of output neurons (1 for binary classification).
    learning_rate : float
        The learning rate for weight updates.
    n_iterations : int
        The number of training epochs.

    Attributes
    ------------
    weights_input_hidden : array
        Weights connecting input to hidden layer.
    bias_hidden : array
        Bias for the hidden layer.
    weights_hidden_output : array
        Weights connecting hidden to output layer.
    bias_output : array
        Bias for the output layer.
    """
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, n_iterations=1000):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.01
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.01
        self.bias_output = np.zeros((1, output_size))

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        # Input to hidden layer
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self._sigmoid(self.hidden_input)

        # Hidden to output layer
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.output = self._sigmoid(self.output_input)
        return self.output

    def backward(self, X, y, output):
        # Calculate output layer error
        output_error = y - output
        output_delta = output_error * self._sigmoid_derivative(output)

        # Calculate hidden layer error
        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * self._sigmoid_derivative(self.hidden_output)

        # Update weights and biases
        self.weights_hidden_output += np.dot(self.hidden_output.T, output_delta) * self.learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * self.learning_rate
        self.weights_input_hidden += np.dot(X.T, hidden_delta) * self.learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * self.learning_rate

    def fit(self, X, y):
        y = y.reshape(-1, 1) # Ensure y is a column vector
        for i in range(self.n_iterations):
            output = self.forward(X)
            self.backward(X, y, output)
            if (i % 100 == 0):
                loss = np.mean(np.square(y - output))
                # print(f"Epoch {i}, Loss: {loss:.4f}")

    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int)

if __name__ == "__main__":
    # Example usage: XOR problem
    X_xor = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y_xor = np.array([0, 1, 1, 0])

    mlp = MLP(input_size=2, hidden_size=4, output_size=1, learning_rate=0.1, n_iterations=10000)
    mlp.fit(X_xor, y_xor)

    predictions = mlp.predict(X_xor)
    print("\nXOR Problem Predictions:")
    print(predictions.flatten())
    print("True labels:", y_xor)

    # Another example: Simple AND gate
    X_and = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y_and = np.array([0, 0, 0, 1])

    mlp_and = MLP(input_size=2, hidden_size=2, output_size=1, learning_rate=0.05, n_iterations=5000)
    mlp_and.fit(X_and, y_and)

    predictions_and = mlp_and.predict(X_and)
    print("\nAND Gate Problem Predictions:")
    print(predictions_and.flatten())
    print("True labels:", y_and)
