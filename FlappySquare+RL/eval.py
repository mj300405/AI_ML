from model import LSTMNet
import torch
from game import FlappyBirdGame
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(input_size, hidden_size, output_size, num_layers, model_path = 'flappy_bird_lstm.pth'):
    model = LSTMNet(input_size=input_size, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers).to(device)
    if os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"Model loaded from {model_path}")
    return model

def play_with_model(game, model, device='cpu'):
    state = game.reset()
    hidden = model.init_hidden(1, device)  # Batch size is 1 since we're evaluating one step at a time
    
    while game.running:
        # Preprocess state for LSTM input
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)  # Add batch dimension
        with torch.no_grad():  # No need to track gradients
            # Forward pass through the model
            action_probabilities, hidden = model(state_tensor, hidden)
        
        # Assuming the model outputs raw scores (logits)
        action = torch.argmax(action_probabilities, dim=1).item()

        # Perform the action in the environment
        next_state, reward, done = game.step(action)
        
        # Render the game to visualize the agent's performance
        game.render()
        
        if done:
            break  # Exit the loop if the game is over
        
        state = next_state  # Update the state

if __name__ == '__main__':

    input_size = 6  # Number of features in your state vector
    hidden_size = 128  # Number of features in LSTM hidden state
    output_size = 2  # Number of actions the agent can take
    num_layers = 2  # Number of LSTM layers

    game = FlappyBirdGame()
    model = load_model(input_size, hidden_size, output_size, num_layers)
    play_with_model(game, model, device)