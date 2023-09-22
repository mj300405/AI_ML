import numpy as np
from sklearn.utils import shuffle


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, model_file=None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        if model_file is not None:
            self.load_model(model_file)
        else:
            self.w1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
            self.b1 = np.zeros((1, hidden_size))
            self.w2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
            self.b2 = np.zeros((1, output_size))
    
    def forward(self, X):
        self.z1 = np.dot(X, self.w1) + self.b1
        self.a1 = np.tanh(self.z1)
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        exp_scores = np.exp(self.z2)
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return self.probs
    
    def backward(self, X, y, learning_rate):
        delta3 = self.probs
        delta3[range(len(X)), y] -= 1
        d_w2 = np.dot(self.a1.T, delta3)
        d_b2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = np.dot(delta3, self.w2.T) * (1 - np.power(self.a1, 2))
        d_w1 = np.dot(X.T, delta2)
        d_b1 = np.sum(delta2, axis=0)
        self.w1 -= learning_rate * d_w1
        self.b1 -= learning_rate * d_b1
        self.w2 -= learning_rate * d_w2
        self.b2 -= learning_rate * d_b2
    
    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)
    
    def save_model(self, model_file):
        np.savez(model_file, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2)
        print("Model saved to file:", model_file)
    
    def load_model(self, model_file):
        with np.load(model_file) as data:
            self.w1 = data['w1']
            self.b1 = data['b1']
            self.w2 = data['w2']
            self.b2 = data['b2']
        print("Model loaded from file:", model_file)
    
    def fit(self, X, y, learning_rate=0.01, epochs=10, batch_size=32):
        num_batches = len(X) // batch_size
        for epoch in range(epochs):
            X, y = shuffle(X, y)
            for i in range(num_batches):
                start = i * batch_size
                end = start + batch_size
                X_batch = X[start:end]
                y_batch = y[start:end]
                self.forward(X_batch)
                self.backward(X_batch, y_batch, learning_rate)
            train_acc = np.mean(self.predict(X) == y)
            print("Epoch:", epoch+1, "Train accuracy:", train_acc)
