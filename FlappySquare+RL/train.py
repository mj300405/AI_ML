import os
import sys
import random
import pygame
import numpy as np
import torch
import torch.optim as optim
from collections import deque
from itertools import count
from LSTM_model import LSTMNet  # Ensure your LSTMNet is correctly defined in model.py
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
num_layers = 4  # Number of LSTM layers
model = LSTMNet(input_size=input_size, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
loss_fn = torch.nn.MSELoss()

# Experience replay memory
memory = deque(maxlen=100000)

# Training hyperparameters
num_episodes = 25000
batch_size = 1
gamma = 0.99  # discount factor
epsilon_start = 0.0
epsilon_end = 0.001
epsilon_decay = 0.01
decay_scale = 1

# Load model if it exists
model_path = 'flappy_bird_lstm.pth'
if os.path.isfile(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded from {model_path}")

# Function to decrease epsilon
def decay_epsilon(episode, num_episodes=num_episodes, epsilon_start=epsilon_start, epsilon_end=epsilon_end, decay_factor=2):
    decay_rate = (epsilon_start - epsilon_end) * decay_factor / num_episodes
    return max(epsilon_end, epsilon_start - decay_rate * episode)



# Initialize epsilon
epsilon = epsilon_start

# Initialize best score
best_score = -float('inf')

render_enabled = False
print_every = 10
save_interval = 100


# Training loop
for episode in range(num_episodes):
    # Instantiate a new game object to reset the game
    game_env.reset()
    state = game_env.get_state()
    hidden = model.init_hidden(batch_size, device)  # Initialize hidden state for each LSTM layer
    total_reward = 0

    for t in count():
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    render_enabled = not render_enabled

        # Prepare state tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(device)

        # Get Q values and hidden state from the model
        q_values, hidden = model(state_tensor, hidden)

        # Detach hidden states from the graph for each LSTM layer
        hidden = [(h.detach(), c.detach()) for h, c in hidden]

        # Epsilon-greedy action selection
        if random.random() > epsilon:
            action = q_values.max(1)[1].view(1, 1).item()
        else:
            action = random.randrange(output_size)

        # Take action and observe new state and reward
        next_state, reward, done = game_env.step(action)
        total_reward += reward

        # Store transition in replay memory
        memory.append((state, action, reward, next_state, done))

        # Move to the next state
        state = next_state

        # Learning step
        if len(memory) > batch_size:
            transitions = random.sample(memory, batch_size)
            batch = map(lambda x: torch.tensor(np.array(x), device=device, dtype=torch.float), zip(*transitions))
            states, actions, rewards, next_states, dones = batch

            # Reshape states for LSTM input and reinitialize hidden state for learning
            states = states.view(batch_size, -1, input_size)
            next_states = next_states.view(batch_size, -1, input_size)
            hidden_train = model.init_hidden(batch_size, device)

            # Compute Q values
            current_q_values = model(states, hidden_train)[0].gather(1, actions.long().view(-1, 1)).squeeze(-1)
            next_q_values = model(next_states, hidden_train)[0].max(1)[0]

            # Expected Q values
            expected_q_values = rewards + (gamma * next_q_values * (1 - dones))

            # Loss and backpropagation
            loss = loss_fn(current_q_values, expected_q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Update epsilon
        epsilon = decay_epsilon(episode)

        # Check for end of episode
        if done:
            break

        # Render game if enabled
        if render_enabled:
            game_env.render()

    # Saving model and printing episode stats
    if total_reward > best_score:
        best_score = total_reward
        torch.save(model.state_dict(), 'flappy_bird_lstm_best.pth')
    if episode % save_interval == 0:
        torch.save(model.state_dict(), 'flappy_bird_lstm.pth')
    if episode % print_every == 0:
        print(f"Episode {episode} complete with total reward: {total_reward}, Epsilon: {epsilon}")

print(f"Training completed over {num_episodes} episodes")
print(f"Final Epsilon value: {epsilon}")




# Close the game when training is complete
pygame.quit()
sys.exit()
