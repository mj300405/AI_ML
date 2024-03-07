import os
import sys
import random
import pygame
import numpy as np
import torch
from QNet_model import QNet, QTrainer  # Import the QNet model and QTrainer
from game import FlappyBirdGame, FPS

# Set device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the game environment
game_env = FlappyBirdGame()
clock = pygame.time.Clock()

# Define the network architecture
input_size = 6
hidden_sizes = [128, 64, 32, 16]  # Adjust the size as needed
output_size = 2
learning_rate = 0.001
gamma = 0.9

# Initialize the QNet model and QTrainer
model = QNet(input_size, hidden_sizes, output_size).to(device)
trainer = QTrainer(model, lr=learning_rate, gamma=gamma)

# Training hyperparameters
num_episodes = 50000
epsilon_start = 0.9
epsilon_end = 0.01

def decay_epsilon(episode, num_episodes=num_episodes, epsilon_start=epsilon_start, epsilon_end=epsilon_end, decay_factor=2.0):
    decay_rate = (epsilon_start - epsilon_end) * decay_factor / num_episodes
    return max(epsilon_end, epsilon_start - decay_rate * episode)


# Load model if it exists
model_path = 'flappy_bird_qnet.pth'
if os.path.isfile(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded from {model_path}")

# Initialize epsilon
epsilon = epsilon_start

# Initialize best score
best_score = -float('inf')

render_enabled = False
print_every = 100
save_interval = 100

# Training loop
for episode in range(num_episodes):
    state = game_env.reset()
    total_reward = 0
    done = False

    while not done:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    render_enabled = not render_enabled

        # Epsilon-greedy action selection
        if random.random() > epsilon:
            state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(device)
            action = model(state_tensor).argmax().item()
        else:
            action = random.randrange(output_size)

        # Take action and observe new state and reward
        next_state, reward, done = game_env.step(action)
        total_reward += reward

        # Store transition and train
        trainer.train_step(state, action, reward, next_state, done)

        state = next_state

        # Render game if enabled
        if render_enabled:
            game_env.render()

        # Update epsilon
        epsilon = decay_epsilon(episode)

    # Saving model and printing episode stats
    if total_reward > best_score:
        best_score = total_reward
        model.save(file_name='flappy_bird_qnet_best.pth')
    if episode % save_interval == 0:
        model.save(file_name='flappy_bird_qnet.pth')
    if episode % print_every == 0:
        print(f"Episode {episode} complete with total reward: {total_reward}, Epsilon: {epsilon}")

print(f"Training completed over {num_episodes} episodes")
print(f"Final Epsilon value: {epsilon}")

# Close the game when training is complete
pygame.quit()
sys.exit()
