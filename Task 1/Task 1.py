import numpy as np
import pickle
import os
from sklearn.metrics import precision_recall_fscore_support

# Helper functions for data loading and preprocessing
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_cifar10_batch(filename):
    batch = unpickle(filename)
    data = batch[b'data'].reshape((10000, 3, 32, 32)).transpose(0, 2, 3, 1).astype(np.float32) / 255
    labels = np.array(batch[b'labels'])
    return data, labels

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

class Dense:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        self.bias = np.zeros((1, output_size))
        # Adam optimizer parameters
        self.m_weights = np.zeros_like(self.weights)
        self.v_weights = np.zeros_like(self.weights)
        self.m_bias = np.zeros_like(self.bias)
        self.v_bias = np.zeros_like(self.bias)
        self.t = 0  # timestep for Adam optimizer
    
    def forward(self, inputs, training=True):
        self.inputs = inputs
        return np.dot(inputs, self.weights) + self.bias
    
    def backward(self, grad_output, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        grad_input = np.dot(grad_output, self.weights.T)
        grad_weights = np.dot(self.inputs.T, grad_output)
        grad_bias = np.sum(grad_output, axis=0, keepdims=True)

        # Adam update for weights
        self.t += 1
        self.m_weights = beta1 * self.m_weights + (1 - beta1) * grad_weights
        self.v_weights = beta2 * self.v_weights + (1 - beta2) * (grad_weights ** 2)

        m_hat_weights = self.m_weights / (1 - beta1 ** self.t)
        v_hat_weights = self.v_weights / (1 - beta2 ** self.t)

        self.weights -= learning_rate * m_hat_weights / (np.sqrt(v_hat_weights) + epsilon)

        # Adam update for bias
        self.m_bias = beta1 * self.m_bias + (1 - beta1) * grad_bias
        self.v_bias = beta2 * self.v_bias + (1 - beta2) * (grad_bias ** 2)

        m_hat_bias = self.m_bias / (1 - beta1 ** self.t)
        v_hat_bias = self.v_bias / (1 - beta2 ** self.t)

        self.bias -= learning_rate * m_hat_bias / (np.sqrt(v_hat_bias) + epsilon)
        
        return grad_input

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
    
    def backward(self, grad_output, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        m = self.inputs.shape[0]
        normalized = (self.inputs - self.mean) / np.sqrt(self.var + self.epsilon)
        
        grad_gamma = np.sum(grad_output * normalized, axis=0, keepdims=True)
        grad_beta = np.sum(grad_output, axis=0, keepdims=True)
        
        grad_normalized = grad_output * self.gamma
        grad_var = np.sum(grad_normalized * (self.inputs - self.mean) * -0.5 * (self.var + self.epsilon)**(-1.5), axis=0, keepdims=True)
        grad_mean = np.sum(grad_normalized * -1 / np.sqrt(self.var + self.epsilon), axis=0, keepdims=True) + grad_var * np.mean(-2 * (self.inputs - self.mean), axis=0, keepdims=True)
        
        grad_input = grad_normalized / np.sqrt(self.var + self.epsilon) + grad_var * 2 * (self.inputs - self.mean) / m + grad_mean / m

        # Adam update for gamma and beta
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

# Loss function
def cross_entropy_loss(y_pred, y_true):
    m = y_true.shape[0]
    log_likelihood = -np.log(y_pred[range(m), y_true])
    loss = np.sum(log_likelihood) / m
    return loss

# Model class
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
    
    def forward(self, x, training=True):
        for layer in self.layers:
            x = layer.forward(x, training)
        return x
    
    def backward(self, grad, learning_rate):
        for layer in reversed(self.layers):
            grad = layer.backward(grad, learning_rate)
    
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
            
            # Evaluate on validation set
            val_pred = self.forward(x_val, training=False)
            val_loss = cross_entropy_loss(val_pred, y_val)
            val_acc = np.mean(np.argmax(val_pred, axis=1) == y_val)
            
            print(f"Epoch {epoch+1}/{epochs}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    def evaluate(self, x, y_true):
        y_pred = self.forward(x)
        loss = cross_entropy_loss(y_pred, y_true)
        y_pred_class = np.argmax(y_pred, axis=1)
        accuracy = np.mean(y_pred_class == y_true)
        
        # Calculate precision and recall for each class
        precision, recall, _, _ = precision_recall_fscore_support(y_true, y_pred_class, average=None)
        
        return loss, accuracy, precision, recall

# Main execution
if __name__ == "__main__":

    # Load and preprocess data
    data_dir = r'C:\Users\Administrator\Downloads\cifar-10-python\cifar-10-batches-py'  # Update this path
    x_train, y_train, x_test, y_test = load_cifar10(data_dir)
    
    # Flatten the images
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    
    # Create and train the model
    model = Model()
    model.train(x_train, y_train, x_test, y_test, epochs=20, batch_size=64, learning_rate=0.001)
    
    # Evaluate on test set
    test_loss, test_acc, test_precision, test_recall = model.evaluate(x_test, y_test)
    test_pred = model.forward(x_test, training=False)
    test_loss = cross_entropy_loss(test_pred, y_test)
    test_acc = np.mean(np.argmax(test_pred, axis=1) == y_test)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    print("\nPrecision and Recall for each class:")
    for i, (p, r) in enumerate(zip(test_precision, test_recall)):
        print(f"Class {i}: Precision = {p:.4f}, Recall = {r:.4f}")
    
    # Calculate and print average precision and recall
    avg_precision = np.mean(test_precision)
    avg_recall = np.mean(test_recall)
    print(f"\nAverage Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
