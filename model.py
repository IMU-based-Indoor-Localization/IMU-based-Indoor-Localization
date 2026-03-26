import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock1D, self).__init__()
        self.conv_path = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(out_channels)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        return F.relu(self.conv_path(x) + self.shortcut(x))
    
class IMU_ResNet_MTL(nn.Module):
    def __init__(self, in_channels=12, num_classes=6): 
        super(IMU_ResNet_MTL, self).__init__()
        
        self.initial = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )
        
        self.layer1 = ResidualBlock1D(64, 64, stride=1)
        self.layer2 = ResidualBlock1D(64, 128, stride=2)
        self.layer3 = ResidualBlock1D(128, 256, stride=2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        self.classifier = nn.Linear(256, num_classes)
        self.regressor = nn.Linear(256, 12) 

    def forward(self, x):
        feat = self.layer3(self.layer2(self.layer1(self.initial(x))))
        feat = self.gap(feat).view(feat.size(0), -1) 
        
        out_cls = self.classifier(feat)
        out_reg = self.regressor(feat)
        
        mu = out_reg[:, :6]
        log_var = out_reg[:, 6:]
        
        return out_cls, mu, log_var