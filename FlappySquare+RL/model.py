import torch
import torch.nn as nn

class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # The first LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # The second LSTM layer, with half the hidden size of the first
        self.lstm2 = nn.LSTM(hidden_size, hidden_size // 2, num_layers, batch_first=True)
        
        # Fully connected layer that outputs the final prediction
        self.fc = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x, hidden):
        # Forward pass through the first LSTM layer
        x, hidden1 = self.lstm(x, hidden[0])

        # Adjust the hidden state size for the second LSTM layer if necessary
        # If the second LSTM layer expects half the size, you need to slice the hidden state
        hidden1_adjusted = (hidden1[0][:, :, :self.hidden_size // 2],
                            hidden1[1][:, :, :self.hidden_size // 2])

        # Forward pass through the second LSTM layer with the adjusted hidden state
        x, hidden2 = self.lstm2(x, hidden1_adjusted)

        x = x[:, -1, :]  # Assuming we only want the last sequence output
        x = self.fc(x)

        return x, (hidden1, hidden2)

    def init_hidden(self, batch_size, device):
        # Initialize hidden and cell states for the first LSTM layer
        hidden_state1 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        cell_state1 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

        # The second LSTM layer's hidden and cell states should be half the size of the first's
        hidden_state2 = torch.zeros(self.num_layers, batch_size, self.hidden_size // 2, device=device)
        cell_state2 = torch.zeros(self.num_layers, batch_size, self.hidden_size // 2, device=device)

        return ((hidden_state1, cell_state1), (hidden_state2, cell_state2))


