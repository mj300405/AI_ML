import os
import sys
import random
import pygame
import numpy as np
import torch
import torch.optim as optim
from collections import deque
from itertools import count
from model import LSTMNet  # Ensure your LSTMNet is correctly defined in model.py
from game import FlappyBirdGame, FPS

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
model = LSTMNet(input_size=input_size, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
loss_fn = torch.nn.MSELoss()

# Experience replay memory
memory = deque(maxlen=10000)

# Training hyperparameters
num_episodes = 20000
batch_size = 1
gamma = 0.99  # discount factor
epsilon_start = 0.7
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
    return max(epsilon_end, epsilon_start - (episode / num_episodes) * (epsilon_start - epsilon_end))

# Initialize epsilon
epsilon = epsilon_start

# Initialize best score
best_score = -float('inf')

# Training loop
for episode in range(num_episodes):
    # Instantiate a new game object to reset the game
    game_env.reset()
    state = game_env.get_state()
    hidden = model.init_hidden(batch_size, device)  # Initialize hidden state for the LSTM layers
    total_reward = 0

    for t in count():
        # Handle events (e.g., check if window close button clicked)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Convert the state into a tensor and add batch and sequence dimensions
        state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(device)

        # Get Q values and hidden state from the model
        q_values, hidden = model(state_tensor, hidden)

        # Detach hidden states from the graph
        hidden = (hidden[0][0].detach(), hidden[0][1].detach()), (hidden[1][0].detach(), hidden[1][1].detach())

        # Epsilon-greedy action selection
        if random.random() > epsilon:
            action = q_values.max(1)[1].view(1, 1).item()
        else:
            action = random.randrange(output_size)

        # Take the selected action and observe the new state and reward
        next_state, reward, done = game_env.step(action)
        total_reward += reward

        # Store the transition in replay memory
        memory.append((state, action, reward, next_state, done))

        # Move to the next state
        state = next_state

        # Learning step: check if we can sample from memory to learn
        if len(memory) > batch_size:
            transitions = random.sample(memory, batch_size)
            # Convert batch-array of transitions to transition of batch-arrays
            states, actions, rewards, next_states, dones = map(
                lambda x: torch.tensor(np.array(x), device=device, dtype=torch.float), 
                zip(*transitions)
            )
            
            # Ensure that actions are of the correct long type for use as indices
            actions = actions.long().view(-1, 1)
            
            # Reshape states for LSTM input
            states = states.view(batch_size, -1, input_size)
            next_states = next_states.view(batch_size, -1, input_size)
            
            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
            current_q_values = model(states, hidden)[0].gather(1, actions).squeeze(-1)
            
            # Compute V(s_{t+1}) for all next states.
            next_q_values = model(next_states, hidden)[0].max(1)[0]
            
            # Compute the expected Q values
            expected_q_values = rewards + (gamma * next_q_values * (1 - dones))
            
            # Compute loss using MSE
            loss = loss_fn(current_q_values, expected_q_values)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        # Update epsilon
        epsilon = decay_epsilon(episode)

        if done:
            break

        # Optionally render the game
        # Optionally render the game
    if episode % 100 == 0:  # Change this as needed to control how often you render
        for _ in range(5):  # Render the game for a few frames
            game_env.render()
            pygame.time.delay(200)  # Delay to slow down the rendering, in milliseconds
            clock.tick(30)  # Control the loop to run at 30 FPS or any other suitable FPS

    # Check if the current episode's score is better than the best score
    if total_reward > best_score:
        best_score = total_reward
        # Save the model with a new filename indicating a performance improvement
        best_model_path = f'flappy_bird_lstm_best.pth'
        torch.save(model.state_dict(), best_model_path)
        print(f"New best score! Model saved to {best_model_path}")

    # Save the model periodically
    if episode % 100 == 0:
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    print(f"Episode {episode} complete with total reward: {total_reward}")

# Close the game when training is complete
pygame.quit()
sys.exit()
