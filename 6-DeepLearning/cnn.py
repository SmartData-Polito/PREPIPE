import torch
import torch.nn as nn

class CNNClog(nn.Module):
    def __init__(self, channel_num, seq_len, out_classes):
        super().__init__()
        
        self.conv1 = nn.Conv1d(channel_num, 10, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(10, 10, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(10, 10, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(10, 10, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(10, 1, kernel_size=3, padding=1)
    
        self.do = nn.Dropout()
        
        self.fc = nn.Linear(seq_len, out_classes)
        self.out_classes = out_classes
        
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = self.conv5(x)
        
        x = x.reshape((x.shape[0], -1))
        
        x = self.do(x)
        
        return self.fc(x)