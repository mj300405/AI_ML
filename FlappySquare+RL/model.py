import torch
import torch.nn as nn

class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # The first LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # The second LSTM layer, with the same hidden size
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        
        # Fully connected layer that outputs the final prediction
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        # Forward pass through the first LSTM layer
        x, hidden1 = self.lstm(x, hidden[0])

        # Forward pass through the second LSTM layer
        x, hidden2 = self.lstm2(x, hidden1)

        x = x[:, -1, :]  # Assuming we only want the last sequence output
        x = self.fc(x)

        return x, (hidden1, hidden2)

    def init_hidden(self, batch_size, device='cpu'):
        # Initialize hidden and cell states for both LSTM layers
        hidden_state = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        cell_state = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

        return (hidden_state, cell_state), (hidden_state.clone(), cell_state.clone())
