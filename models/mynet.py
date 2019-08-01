import torch.nn as nn
import torch.nn.functional as F

# nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
# nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
# nn.AvgPool2d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True)
# nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# nn.ReLU(inplace=False)
# nn.Linear(in_features, out_features, bias=True)

class MyNet(nn.Module):
    def __init__(self, input_shape, num_classes=10):
        super(MyNet, self).__init__()
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),)
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True))

        self.conv2_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True))
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True))

        self.conv3_1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True))
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True))
        self.conv3_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True))
        self.conv3_4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True))

        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.averagePool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.globalAvgPool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(in_features=256, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=1024)
        self.fc3 = nn.Linear(in_features=1024, out_features=num_classes)

    def forward(self, x):
        out = self.conv1_1(x)
        out = self.conv1_2(out)
        out = self.maxPool(out)

        out = self.conv2_1(out)
        out = self.conv2_2(out)
        out = self.maxPool(out)

        out = self.conv3_1(out)
        out = self.conv3_2(out)
        out = self.conv3_3(out)
        out = self.conv3_4(out)
        out = self.globalAvgPool(out)

        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out