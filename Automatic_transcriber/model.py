import torch
import torch.nn as nn
import torch.nn.functional as F

class FeaturePreprocessor(nn.Module):
    def __init__(self, input_size, output_size=128):
        super(FeaturePreprocessor, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.bn = nn.BatchNorm1d(output_size)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the feature if it's not already
        x = F.relu(self.bn(self.linear(x)))
        return x

class PolyphonicPianoModel(nn.Module):
    def __init__(self, feature_dims):
        super(PolyphonicPianoModel, self).__init__()
        # Initialize preprocessors for each feature
        self.preprocessors = nn.ModuleDict({
            name: FeaturePreprocessor(input_size) for name, input_size in feature_dims.items()
        })
        
        # Calculate the total combined feature size after preprocessing
        total_feature_size = 128 * len(feature_dims)  # Each preprocessor outputs a size of 128
        
        # LSTM layer to process the combined features
        self.lstm = nn.LSTM(total_feature_size, 256, batch_first=True, num_layers=2, bidirectional=True)
        
        # Attention layer
        self.attention = nn.Linear(256 * 2, 1)  # For bidirectional LSTM
        
        # Final layer to predict the notes
        self.fc = nn.Linear(256 * 2, 88)  # Assuming 88 keys on a piano
        
    def forward(self, features):
        # Preprocess and combine features
        preprocessed_features = [self.preprocessors[name](feature) for name, feature in features.items()]
        combined_features = torch.cat(preprocessed_features, dim=-1)
        
        # Reshape for LSTM input
        combined_features = combined_features.unsqueeze(1)  # Add a sequence dimension
        
        # Sequence processing
        lstm_out, _ = self.lstm(combined_features)
        
        # Apply attention
        attn_weights = F.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(lstm_out * attn_weights, dim=1)
        
        # Final prediction
        output = torch.sigmoid(self.fc(context))
        return output
