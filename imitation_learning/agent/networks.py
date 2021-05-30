import torch.nn as nn
import torch
import torch.nn.functional as F
from torchsummary import summary

"""
Imitation learning network
"""

# class CNN(nn.Module):

#     def __init__(self, history_length=0, n_classes=3): 
#         super(CNN, self).__init__()
#         # TODO : define layers of a convolutional neural network

#     def forward(self, x):
#         # TODO: compute forward pass
#         return x

class CNN(nn.Module):
    def __init__(self, history_length=0, n_classes=4):
        super(CNN, self).__init__()
        self.cnn_block = nn.Sequential(
            nn.Conv2d(in_channels=history_length + 1,
                      out_channels=32,
                      kernel_size=3),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(negative_slope=0.01),  
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=64, out_channels=10, kernel_size=3),
            nn.BatchNorm2d(num_features=10),
            nn.LeakyReLU(negative_slope=0.01),
        )
        # --- use the line below and test bc_agent to find out the output of conv
        # summary(self.cnn_block, (history_length + 1, 96, 96))

        # --- these are the result of the above convolution ---
        w = 16
        h = 16
        channels = 10

        # self.fc_block = nn.Sequential(
        #     nn.Linear(in_features=w * h * channels, out_features=32),
        #     nn.ReLU(),
        #     nn.Linear(in_features=32, out_features=4),
        # )
        self.fc_block = nn.Sequential(
            nn.Linear(in_features=w * h * channels, out_features=256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256, out_features=128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=128, out_features=4),
        )


    def _conv_out(self, w, k=3, s=1, p=0):
        return ((2 * p + w - k) / s) + 1

    def forward(self, x):
        batch_size = x.size(0)
        x = self.cnn_block(x)
        x = x.view(batch_size, -1)
        x = self.fc_block(x)
        return x
