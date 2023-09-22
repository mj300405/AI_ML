import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from neural_network import NeuralNetwork

# Load the MNIST dataset
digits = load_digits()
X, y = digits.data, digits.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the neural network
nn = NeuralNetwork(input_size=X.shape[1], hidden_size=128, output_size=len(np.unique(y)))

# Train the network on the training set
nn.fit(X_train, y_train, learning_rate=0.001, epochs=500, batch_size=32)

# Evaluate the network on the testing set
test_acc = np.mean(nn.predict(X_test) == y_test)
print("Test accuracy:", test_acc)

# Save the trained model to a file
nn.save_model("mnist_model.npz")

# Select some random images from the testing set
num_images = 10
indices = np.random.choice(X_test.shape[0], num_images)

# Make predictions for the selected images using the trained neural network
predictions = nn.predict(X_test[indices])

# Plot the selected images and their corresponding predictions
fig, axs = plt.subplots(1, num_images, figsize=(15, 5))
for i, idx in enumerate(indices):
    axs[i].imshow(X_test[idx].reshape(8, 8), cmap='gray')
    axs[i].set_title(f"Prediction: {predictions[i]}, Label: {y_test[idx]}")
plt.show()