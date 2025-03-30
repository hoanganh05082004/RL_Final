import torch.nn as nn
import torch
import numpy as np


class QNetwork(nn.Module):
    def __init__(self, observation_shape, action_shape):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(observation_shape[-1], observation_shape[-1], 3),
            nn.ReLU(),
            nn.Conv2d(observation_shape[-1], observation_shape[-1], 3),
            nn.ReLU(),
        )
        dummy_input = torch.randn(observation_shape).permute(2, 0, 1)
        dummy_output = self.cnn(dummy_input)
        flatten_dim = dummy_output.view(-1).shape[0]
        self.network = nn.Sequential(
            nn.Linear(flatten_dim, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, action_shape),
        )

    def forward(self, x):
        assert len(x.shape) >= 3, "only support magent input observation"
        x = self.cnn(x)
        if len(x.shape) == 3:
            batchsize = 1
        else:
            batchsize = x.shape[0]
        x = x.reshape(batchsize, -1)
        return self.network(x)


class QNetwork3(nn.Module):
    def __init__(self, observation_shape, action_shape):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(observation_shape[-1], 32, 3),  # Số kênh đầu vào = observation_shape[-1]
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),                     # Số kênh trung gian
            nn.ReLU(),
        )
        # Tính toán flatten_dim
        dummy_input = torch.randn(1, observation_shape[-1], observation_shape[0], observation_shape[1])
        dummy_output = self.cnn(dummy_input)
        flatten_dim = dummy_output.view(-1).shape[0]
        
        self.network = nn.Sequential(
            nn.Linear(flatten_dim, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, action_shape),
        )

    def forward(self, x):
        assert len(x.shape) >= 3, "Input observation must have at least 3 dimensions (HWC or CHW)."
        
        if len(x.shape) == 3:  # Nếu thiếu batch dimension
            x = x.unsqueeze(0)  # Thêm batch dimension
        
        batchsize = x.shape[0]
        
        # Chuyển từ HWC (Height, Width, Channels) sang CHW (Channels, Height, Width)
        x = torch.fliplr(x).permute(0, 3, 1, 2)
        
        # Dẫn truyền qua CNN
        x = self.cnn(x)
        
        # Flatten đầu ra từ CNN
        x = x.view(batchsize, -1)
        
        # Dẫn truyền qua Fully Connected Layers
        return self.network(x)
