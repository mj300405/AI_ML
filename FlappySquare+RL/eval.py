from model import LSTMNet
import torch
from game import FlappyBirdGame
import os
import pygame

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(input_size, hidden_size, output_size, num_layers, model_path = 'flappy_bird_lstm.pth'):
    model = LSTMNet(input_size=input_size, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers).to(device)
    if os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"Model loaded from {model_path}")
    return model

# Evaluation loop
def evaluate_model(game_env, model, device):
    state = game_env.reset()
    hidden = model.init_hidden(1, device)  # Initialize hidden state for the LSTM layers
    
    while True:  # Run until 'Q' is pressed
        # Convert the state into a tensor and add batch and sequence dimensions
        state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(device)

        # Get Q values and hidden state from the model
        with torch.no_grad():  # No need to compute gradients when not training
            q_values, hidden = model(state_tensor, hidden)
            hidden = (hidden[0][0].detach(), hidden[0][1].detach()), (hidden[1][0].detach(), hidden[1][1].detach())
        # Choose the action with the highest Q-value
        action = q_values.max(1)[1].view(1, 1).item()

        # Take the selected action and observe the new state and reward
        next_state, _, done = game_env.step(action)

        # Move to the next state
        state = next_state

        # Render the game
        game_env.render()

        # Check for 'Q' to quit
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                return

        # Check if the game is over
        if done:
            state = game_env.reset()  # Reset the game if it's over


# If running this module directly, call play_with_model
if __name__ == "__main__":

    input_size = 6  # Number of features in your state vector
    hidden_size = 128  # Number of features in LSTM hidden state
    output_size = 2  # Number of actions the agent can take
    num_layers = 2  # Number of LSTM layers

    game = FlappyBirdGame()
    model = load_model(input_size, hidden_size, output_size, num_layers, 'flappy_bird_lstm.pth')
    model.eval()  # Set the model to evaluation mode
    model.to(device)

    evaluate_model(game, model, device)
