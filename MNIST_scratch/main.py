import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Load MNIST data
train_df = pd.read_csv('mnist_train.csv')
test_df = pd.read_csv('mnist_test.csv')

# Split into features and labels
x_train = train_df.iloc[:, 1:].values.astype('float32')
y_train = train_df.iloc[:, 0].values.astype('int32')
x_test = test_df.iloc[:, 1:].values.astype('float32')
y_test = test_df.iloc[:, 0].values.astype('int32')

# Normalize pixel values
x_train /= 255
x_test /= 255

# Convert labels to one-hot encoding
num_classes = 10
y_train = np.eye(num_classes)[y_train]
y_test = np.eye(num_classes)[y_test]

# Define neural network architecture
input_size = x_train.shape[1]
hidden_size = 256
output_size = num_classes

# Initialize parameters
if os.path.isfile('weights.npz'):
    weights = np.load('weights.npz')
    w1 = weights['w1']
    b1 = weights['b1']
    w2 = weights['w2']
    b2 = weights['b2']
else:
    w1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
    b1 = np.zeros((1, hidden_size))
    w2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
    b2 = np.zeros((1, output_size))

# Define activation functions
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Define forward pass function
def forward(x, w1, b1, w2, b2):
    h1 = np.dot(x, w1) + b1
    a1 = relu(h1)
    h2 = np.dot(a1, w2) + b2
    a2 = softmax(h2)
    return h1, a1, h2, a2

# Define backward pass function
def backward(x, y, h1, a1, h2, a2, w1, b1, w2, b2, learning_rate):
    m = y.shape[0]
    delta3 = a2 - y
    dw2 = np.dot(a1.T, delta3)
    db2 = np.sum(delta3, axis=0, keepdims=True)
    delta2 = np.dot(delta3, w2.T) * (h1 > 0)
    dw1 = np.dot(x.T, delta2)
    db1 = np.sum(delta2, axis=0)
    w2 -= learning_rate * dw2 / m
    b2 -= learning_rate * db2 / m
    w1 -= learning_rate * dw1 / m
    b1 -= learning_rate * db1 / m
    return w1, b1, w2, b2

# Define function to evaluate accuracy
def evaluate_accuracy(x, y, w1, b1, w2, b2):
    _, _, _, a2 = forward(x, w1, b1, w2, b2)
    predictions = np.argmax(a2, axis=1)
    true_labels = np.argmax(y, axis=1)
    accuracy = np.sum(predictions == true_labels) / y.shape[0]
    return accuracy


if __name__=='__main__':
    # Train neural network
    num_iterations = 1000
    learning_rate = 0.01
    batch_size = 128
    num_batches = x_train.shape[0] // batch_size

    for i in range(num_iterations):
        # Shuffle training data
        permutation = np.random.permutation(x_train.shape[0])
        x_train_shuffled = x_train[permutation]
        y_train_shuffled = y_train[permutation]
        
        # Train on batches
        for j in range(num_batches):
            start = j * batch_size
            end = (j + 1) * batch_size
            x_batch = x_train_shuffled[start:end]
            y_batch = y_train_shuffled[start:end]
            
            # Forward pass
            h1, a1, h2, a2 = forward(x_batch, w1, b1, w2, b2)
            
            # Backward pass
            w1, b1, w2, b2 = backward(x_batch, y_batch, h1, a1, h2, a2, w1, b1, w2, b2, learning_rate)
        
        # Evaluate accuracy on training and test data
        train_accuracy = evaluate_accuracy(x_train, y_train, w1, b1, w2, b2)
        test_accuracy = evaluate_accuracy(x_test, y_test, w1, b1, w2, b2)
        
        # Print progress
        if (i+1) % 10 == 0:
            print(f"Iteration {i+1}/{num_iterations}: train_acc = {train_accuracy:.4f}, test_acc = {test_accuracy:.4f}")

    np.savez('weights.npz', w1=w1, b1=b1, w2=w2, b2=b2)

 


    # making a plot
    fig, axs = plt.subplots(nrows=10, ncols=10, figsize=(10, 6))

    # randomly select six images
    indices = np.random.choice(x_test.shape[0], size=100, replace=False)

    # loop over the subplots and plot each image with its prediction
    for i, ax in enumerate(axs.flat):
        idx = indices[i]
        x = x_test[idx]
        y = y_test[idx]

        # Reshape the image and make a prediction
        x = x.reshape(1, -1)
        _, _, _, a2 = forward(x, w1, b1, w2, b2)
        pred = np.argmax(a2)

        # Display the image and the prediction
        ax.imshow(x.reshape(28, 28), cmap='gray')
        ax.axis('off')
        ax.set_title(f"True: {np.argmax(y)}\nPred: {pred}")

    # adjust spacing between subplots and display the plot
    fig.subplots_adjust(hspace=1.0, wspace=1.0)
    plt.show()