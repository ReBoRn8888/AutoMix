import torch.nn as nn
import torch.nn.functional as F

class FeatureNet(nn.Module):
    def __init__(self, input_shape):
        super(FeatureNet, self).__init__()
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),)
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True))

        self.globalAvgPool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(in_features=128, out_features=1024)

    def forward(self, x):
        out = self.conv1_1(x)
        out = self.conv1_2(out)
        out = self.globalAvgPool(out)

        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out