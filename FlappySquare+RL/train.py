import os
import sys
import random
import pygame
import numpy as np
import torch
import torch.optim as optim
from game import FPS
from collections import deque
from itertools import count
from model import LSTMNet  # Ensure your LSTMNet is defined in model.py
from game import FlappyBirdGame


# Set device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the game environment
game_env = FlappyBirdGame()
clock = pygame.time.Clock()

# Define the LSTM network architecture
input_size = 6  # Number of features in your state vector
hidden_size = 128  # Number of features in LSTM hidden state
output_size = 2  # Number of actions the agent can take
num_layers = 2  # Number of LSTM layers
model = LSTMNet(input_size=input_size,hidden_size=hidden_size,output_size=output_size,num_layers=num_layers).to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01)
loss_fn = torch.nn.MSELoss()

# Experience replay memory
memory = deque(maxlen=10000)

# Training hyperparameters
num_episodes = 1000
batch_size = 1
gamma = 0.99  # discount factor
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.995

# Load model if it exists
model_path = 'flappy_bird_lstm.pth'
if os.path.isfile(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded from {model_path}")

# Function to decrease epsilon
def decay_epsilon(episode):
    return epsilon_end + (epsilon_start - epsilon_end) * np.exp(-1. * episode / epsilon_decay)

# Initialize epsilon
epsilon = epsilon_start

# Training loop
for episode in range(num_episodes):
    # Instantiate a new game object to reset the game
    game_env = FlappyBirdGame()
    state = game_env.get_state()
    hidden = model.init_hidden(batch_size, device)  # Initialize hidden state for the LSTM layers
    total_reward = 0

    for t in count():
        # Handle events (e.g., check if window close button clicked)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Select and perform an action
        state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(device)
        q_values, hidden = model(state_tensor, hidden)
        hidden = ((hidden[0][0].detach(), hidden[0][1].detach()), 
                  (hidden[1][0].detach(), hidden[1][1].detach()))
        
        if random.random() > epsilon:
            action = q_values.max(1)[1].item()
        else:
            action = random.randrange(output_size)

        # Take action and observe new state and reward
        next_state, reward, done = game_env.step(action)
        total_reward += reward

        # Store the transition in replay memory
        memory.append((state, action, reward, next_state, done))

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        if len(memory) > batch_size:
            transitions = random.sample(memory, batch_size)
            batch = zip(*transitions)
            states, actions, rewards, next_states, dones = map(lambda x: torch.tensor(np.array(x), device=device, dtype=torch.float), batch)

            # Flatten the batch of states and next states for LSTM input
            states = states.view(batch_size, 1, -1)
            next_states = next_states.view(batch_size, 1, -1)

            # Forward pass through the network
            current_q_values, _ = model(states, hidden)
            next_q_values, _ = model(next_states, hidden)

            # Compute the expected Q values
            max_next_q_values = next_q_values.detach().max(1)[0]
            expected_q_values = rewards + (gamma * max_next_q_values * (1 - dones))

            # Compute loss
            loss = loss_fn(current_q_values.squeeze(), expected_q_values.unsqueeze(1))

            # Backpropagate the error
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

        # Update epsilon
        epsilon = decay_epsilon(episode)

        if done:
            break

    # Optionally render the game
    if episode % 10 == 0:
        game_env.render()

    # Save the model periodically
    if episode % 100 == 0:
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    print(f"Episode {episode} complete with total reward: {total_reward}")

# Close the game when training is complete
pygame.quit()
sys.exit()