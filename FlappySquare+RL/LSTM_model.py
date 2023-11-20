# import torch
# import torch.nn as nn

# class LSTMNet(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, num_layers):
#         super(LSTMNet, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
        
#         # The first LSTM layer
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
#         # The second LSTM layer, with the same hidden size
#         self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        
#         # Fully connected layer that outputs the final prediction
#         self.fc = nn.Linear(hidden_size, output_size)

#     def forward(self, x, hidden):
#         # Forward pass through the first LSTM layer
#         x, hidden1 = self.lstm(x, hidden[0])

#         # Forward pass through the second LSTM layer
#         x, hidden2 = self.lstm2(x, hidden1)

#         x = x[:, -1, :]  # Assuming we only want the last sequence output
#         x = self.fc(x)

#         return x, (hidden1, hidden2)

#     def init_hidden(self, batch_size, device='cpu'):
#         # Initialize hidden and cell states for both LSTM layers
#         hidden_state = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
#         cell_state = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

#         return (hidden_state, cell_state), (hidden_state.clone(), cell_state.clone())

import torch
import torch.nn as nn

class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout=0.3):
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define the LSTM layers with dropout
        self.lstm_layers = nn.ModuleList()
        for i in range(num_layers):
            input_dim = input_size if i == 0 else hidden_size
            layer = nn.LSTM(input_dim, hidden_size, num_layers=1, batch_first=True, dropout=(dropout if i < num_layers - 1 else 0))
            self.lstm_layers.append(layer)

        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)

        # Activation function
        self.activation = nn.LeakyReLU()

    def forward(self, x, hidden):
        # Forward pass through each LSTM layer
        for i, lstm_layer in enumerate(self.lstm_layers):
            x, hidden[i] = lstm_layer(x, hidden[i])

        # Select the last sequence output
        x = x[:, -1, :]

        # Fully connected layer
        x = self.fc(x)
        x = self.activation(x)

        return x, hidden

    def init_hidden(self, batch_size, device='cpu'):
        # Initialize hidden and cell states for all LSTM layers
        hidden_states = [(torch.zeros(1, batch_size, self.hidden_size, device=device), 
                          torch.zeros(1, batch_size, self.hidden_size, device=device)) for _ in range(self.num_layers)]
        return hidden_states

