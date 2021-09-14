import torch
import torch.nn as nn

class LSTMClog(nn.Module):
    def __init__(self, channels_num, seq_size, out_classes):
        super().__init__()
        
        hidden_size=32
        bidirectional=True
        
        self.lstm = nn.LSTM(
            input_size=channels_num,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True, # (batch, seq, feature)
            bidirectional=bidirectional
        )
        
        self.fc = nn.Linear(
            seq_size * (int(bidirectional)+1) * hidden_size,
            out_classes
        )
    
    def forward(self, x):
        y, _ = self.lstm(x)
        y = y.reshape(x.shape[0], -1)
        return self.fc(y)