import torch
import torch.nn as nn

class GaitNet(nn.Module):
    def __init__(self, input_channels: int):
        super(GaitNet, self).__init__()
        
        # 1. Feature Extraction Layers
        # We use a kernel_size of 5 to 11 to capture the 'shape' of a single step
        self.conv_block = nn.Sequential(
            # Layer 1: Detecting basic edges/spikes
            nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=9, stride=1, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), # Reduces 750 -> 375
            
            # Layer 2: Detecting patterns (rhythm)
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), # Reduces 375 -> 187
            
            # Layer 3: Higher-level biomechanical signatures
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1) # Squashes the time dimension to a single vector
        )
        
        # 2. Regression Head
        # Takes the 128 'features' learned by the CNN and predicts speed
        self.regressor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2), # Prevents the model from memorizing your specific run
            nn.Linear(64, 1) # Output: Single float for speed_mps
        )

    def forward(self, x):
        # x shape: (Batch, Channels, Seq_Length) -> (Batch, 4, 750)
        x = self.conv_block(x)
        x = x.view(x.size(0), -1) # Flatten
        x = self.regressor(x)
        return x
