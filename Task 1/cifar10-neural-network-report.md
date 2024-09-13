# CIFAR-10 Neural Network Implementation Report

## 1. Imports and Data Loading

```python
import numpy as np
import pickle
import os
from sklearn.metrics import precision_recall_fscore_support
```

These lines import necessary libraries:
- `numpy` for efficient numerical computations
- `pickle` for deserializing the CIFAR-10 data
- `os` for file path operations
- `sklearn.metrics` for calculating precision and recall

```python
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
```

This function reads a pickled file and returns its contents. It's used to load the CIFAR-10 dataset, which is stored in a pickled format.

```python
def load_cifar10_batch(filename):
    batch = unpickle(filename)
    data = batch[b'data'].reshape((10000, 3, 32, 32)).transpose(0, 2, 3, 1).astype(np.float32) / 255
    labels = np.array(batch[b'labels'])
    return data, labels
```

This function loads a single batch of the CIFAR-10 dataset:
- It unpickles the file
- Reshapes the data from a flat array to a 4D array (10000 images, 32x32 pixels, 3 color channels)
- Normalizes the pixel values to the range [0, 1]
- Returns the data and labels

```python
def load_cifar10(data_dir):
    x_train = []
    y_train = []
    
    for i in range(1, 6):
        fname = os.path.join(data_dir, f'data_batch_{i}')
        data, labels = load_cifar10_batch(fname)
        x_train.append(data)
        y_train.append(labels)
    
    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)
    
    test_fname = os.path.join(data_dir, 'test_batch')
    x_test, y_test = load_cifar10_batch(test_fname)
    
    return x_train, y_train, x_test, y_test
```

This function loads the entire CIFAR-10 dataset:
- It loads 5 training batches and concatenates them
- Loads the test batch separately
- Returns training and test data and labels

## 2. Neural Network Layers

### Dense Layer

```python
class Dense:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        self.bias = np.zeros((1, output_size))
        # Adam optimizer parameters
        self.m_weights = np.zeros_like(self.weights)
        self.v_weights = np.zeros_like(self.weights)
        self.m_bias = np.zeros_like(self.bias)
        self.v_bias = np.zeros_like(self.bias)
        self.t = 0  
```

This initializes a fully connected (dense) layer:
- Weights are initialized using He initialization for better gradient flow
- Biases are initialized to zero
- Parameters for the Adam optimizer are initialized

```python
def forward(self, inputs, training=True):
    self.inputs = inputs
    return np.dot(inputs, self.weights) + self.bias
```

The forward pass computes the dot product of inputs and weights, then adds the bias.

```python
def backward(self, grad_output, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
    grad_input = np.dot(grad_output, self.weights.T)
    grad_weights = np.dot(self.inputs.T, grad_output)
    grad_bias = np.sum(grad_output, axis=0, keepdims=True)

    # Adam weight update
    self.t += 1
    self.m_weights = beta1 * self.m_weights + (1 - beta1) * grad_weights
    self.v_weights = beta2 * self.v_weights + (1 - beta2) * (grad_weights ** 2)

    m_hat_weights = self.m_weights / (1 - beta1 ** self.t)
    v_hat_weights = self.v_weights / (1 - beta2 ** self.t)

    self.weights -= learning_rate * m_hat_weights / (np.sqrt(v_hat_weights) + epsilon)

    # Adam bias update
    self.m_bias = beta1 * self.m_bias + (1 - beta1) * grad_bias
    self.v_bias = beta2 * self.v_bias + (1 - beta2) * (grad_bias ** 2)

    m_hat_bias = self.m_bias / (1 - beta1 ** self.t)
    v_hat_bias = self.v_bias / (1 - beta2 ** self.t)

    self.bias -= learning_rate * m_hat_bias / (np.sqrt(v_hat_bias) + epsilon)
    
    return grad_input
```

The backward pass:
- Computes gradients with respect to inputs, weights, and biases
- Implements the Adam optimization algorithm for updating weights and biases
- Returns the gradient with respect to the input for backpropagation

### Batch Normalization Layer

```python
class BatchNormalization:
    def __init__(self, input_size, epsilon=1e-5, momentum=0.9):
        self.gamma = np.ones((1, input_size))
        self.beta = np.zeros((1, input_size))
        self.epsilon = epsilon
        self.momentum = momentum
        self.running_mean = np.zeros((1, input_size))
        self.running_var = np.ones((1, input_size))
        # Adam parameters
        self.m_gamma = np.zeros_like(self.gamma)
        self.v_gamma = np.zeros_like(self.gamma)
        self.m_beta = np.zeros_like(self.beta)
        self.v_beta = np.zeros_like(self.beta)
        self.t = 0
```

This initializes a batch normalization layer:
- `gamma` and `beta` are learnable parameters
- Running mean and variance are maintained for inference
- Adam optimizer parameters are initialized

```python
def forward(self, inputs, training=True):
    if training:
        mean = np.mean(inputs, axis=0, keepdims=True)
        var = np.var(inputs, axis=0, keepdims=True)
        self.inputs = inputs
        self.mean = mean
        self.var = var
        
        self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
        self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        
        normalized = (inputs - mean) / np.sqrt(var + self.epsilon)
    else:
        normalized = (inputs - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
    
    return self.gamma * normalized + self.beta
```

The forward pass:
- During training, computes mean and variance of the current batch
- Updates running mean and variance for use during inference
- Normalizes the inputs
- Scales and shifts the normalized values using `gamma` and `beta`

```python
def backward(self, grad_output, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
    # ... (Gradient computation for batch normalization)
    
    # Adam gamma and beta update
    self.t += 1
    self.m_gamma = beta1 * self.m_gamma + (1 - beta1) * grad_gamma
    self.v_gamma = beta2 * self.v_gamma + (1 - beta2) * (grad_gamma ** 2)
    
    m_hat_gamma = self.m_gamma / (1 - beta1 ** self.t)
    v_hat_gamma = self.v_gamma / (1 - beta2 ** self.t)
    self.gamma -= learning_rate * m_hat_gamma / (np.sqrt(v_hat_gamma) + epsilon)
    
    self.m_beta = beta1 * self.m_beta + (1 - beta1) * grad_beta
    self.v_beta = beta2 * self.v_beta + (1 - beta2) * (grad_beta ** 2)
    
    m_hat_beta = self.m_beta / (1 - beta1 ** self.t)
    v_hat_beta = self.v_beta / (1 - beta2 ** self.t)
    self.beta -= learning_rate * m_hat_beta / (np.sqrt(v_hat_beta) + epsilon)
    
    return grad_input
```

The backward pass:
- Computes gradients with respect to inputs, gamma, and beta
- Updates gamma and beta using the Adam optimization algorithm

### Dropout Layer

```python
class Dropout:
    def __init__(self, drop_rate=0.5):
        self.drop_rate = drop_rate
    
    def forward(self, inputs, training=True):
        if training:
            self.mask = np.random.binomial(1, 1 - self.drop_rate, size=inputs.shape) / (1 - self.drop_rate)
            return inputs * self.mask
        else:
            return inputs
    
    def backward(self, grad_output, learning_rate):
        return grad_output * self.mask
```

This implements dropout regularization:
- During training, randomly sets some activations to zero
- During inference, all neurons are active
- The backward pass applies the same mask to the gradient

### Activation Functions

```python
class ReLU:
    def forward(self, inputs, training=True):
        self.inputs = inputs
        return np.maximum(0, inputs)
    
    def backward(self, grad_output, learning_rate):
        return grad_output * (self.inputs > 0)

class Softmax:
    def forward(self, inputs, training=True):
        exp = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)
    
    def backward(self, grad_output, learning_rate):
        return grad_output
```

These implement the ReLU and Softmax activation functions:
- ReLU: max(0, x)
- Softmax: used for multi-class classification output

## 3. Loss Function

```python
def cross_entropy_loss(y_pred, y_true):
    m = y_true.shape[0]
    log_likelihood = -np.log(y_pred[range(m), y_true])
    loss = np.sum(log_likelihood) / m
    return loss
```

This implements the cross-entropy loss function, commonly used for classification tasks.

## 4. Model Class

```python
class Model:
    def __init__(self):
        self.layers = [
            Dense(3072, 256),
            BatchNormalization(256),
            ReLU(),
            Dropout(0.3),
            Dense(256, 128),
            BatchNormalization(128),
            ReLU(),
            Dropout(0.3),
            Dense(128, 10),
            Softmax()
        ]
```

This defines the neural network architecture:
- Input layer: 3072 neurons (32x32x3 flattened image)
- Two hidden layers with 256 and 128 neurons
- Output layer: 10 neurons (one for each CIFAR-10 class)
- Uses batch normalization and dropout for regularization

```python
def forward(self, x, training=True):
    for layer in self.layers:
        x = layer.forward(x, training)
    return x

def backward(self, grad, learning_rate):
    for layer in reversed(self.layers):
        grad = layer.backward(grad, learning_rate)
```

These methods implement the forward and backward passes through the network.

```python
def train(self, x_train, y_train, x_val, y_val, epochs, batch_size, learning_rate):
    for epoch in range(epochs):
        # Shuffle training data
        permutation = np.random.permutation(len(x_train))
        x_train = x_train[permutation]
        y_train = y_train[permutation]
        
        for i in range(0, len(x_train), batch_size):
            batch_x = x_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            y_pred = self.forward(batch_x, training=True)
            
            # Compute gradient
            grad = y_pred.copy()
            grad[range(batch_y.shape[0]), batch_y] -= 1
            grad /= batch_y.shape[0]
            
            # Backpropagation
            self.backward(grad, learning_rate)

        # Validation 
        val_pred = self.forward(x_val, training=False)
        val_loss = cross_entropy_loss(val_pred, y_val)
        val_acc = np.mean(np.argmax(val_pred, axis=1) == y_val)
        
        print(f"Epoch {epoch+1}/{epochs}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
```

This method trains the model:
- Implements mini-batch gradient descent
- Shuffles the training data at each epoch
- Performs forward and backward passes
- Computes validation loss and accuracy after each epoch

```python
def evaluate(self, x, y_true):
    y_pred = self.forward(x)
    loss = cross_entropy_loss(y_pred, y_true)
    y_pred_class = np.argmax(y_pred, axis=1)
    accuracy = np.mean(y_pred_class == y_true)
    precision, recall, _, _ = precision_recall_fscore_support(y_true, y_pred_class, average=None)
    
    return loss, accuracy, precision, recall
```

This method evaluates the model's performance:
- Computes loss, accuracy, precision, and recall

## 5. Main Execution

```python
if __name__ == "__main__":
    data_dir = r'C:\Users\Administrator\Downloads\cifar-10-python\cifar-10-batches-py'
    x_train, y_train, x_test, y_test = load_cifar10(data_dir)
    
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    
    model = Model()
    model.train(x_train, y_train, x_test, y_test, epochs=20, batch_size=64, learning_rate=0.001)
    
    test_loss, test_acc, test_precision, test_recall = model.evaluate(x_test, y_test)
    test_pred = model.forward(x_test, training=False)
    test_loss = cross_entropy_loss(test_pred, y_test)
    test_acc = np.mean(np.argmax(test_pred, axis=1) == y_test)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    print