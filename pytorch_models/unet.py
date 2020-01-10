import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

CHANNEL_SIZE = 16

def mixup_process(out, target_reweighted, indices, lam):
    shuffled = out[indices]
    out = out*lam + shuffled*(1-lam)
    target_shuffled_onehot = target_reweighted[indices]
    target_reweighted = target_reweighted * lam + target_shuffled_onehot * (1 - lam)

    return out, target_reweighted

class UNet(nn.Module):
    def conv_block(self, in_channels, out_channels, kernel_size=3):
        block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
#             nn.ReLU(True),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
#             nn.ReLU(True)
            nn.LeakyReLU(0.2, True),
        )
        return block

    def deconv_block(self, in_channels, mid_channels, out_channels, output_padding=0, kernel_size=3):
        block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
#             nn.ReLU(True),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
#             nn.ReLU(True),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=2, stride=2, output_padding=output_padding),
            nn.BatchNorm2d(out_channels),
#             nn.ReLU(True),
            nn.LeakyReLU(0.2, True),
        )
        return block

    def final_block(self, in_channels, mid_channels, out_channels, kernel_size=3):
        block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
#             nn.ReLU(True),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
#             nn.ReLU(True),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
#             nn.ReLU(True),
            nn.LeakyReLU(0.2, True),
        )
        return block

    def crop_and_concat(self, upsampled, bypass, crop=False):
        if(crop):
            diffY = bypass.size()[2] - upsampled.size()[2]
            diffX = bypass.size()[3] - upsampled.size()[3]
            bypass = F.pad(bypass, (diffX // 2, diffX - diffX//2, diffY // 2, diffY - diffY//2))
            # c = (bypass.size()[2] - upsampled.size()[2]) // 2
            # print(c)
            # bypass = F.pad(bypass, (c, c, c, c))
        return torch.cat((bypass, upsampled), 1)

    def __init__(self, input_shape, output_shape, num_classes):
        super(UNet, self).__init__()
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.input_size = input_shape[0] * input_shape[1] * input_shape[2]
        self.output_size = output_shape[0] * output_shape[1] * output_shape[2]

        # self.labelDense = nn.Sequential(
        #                  nn.Linear(num_classes, self.input_size),
        #                  # nn.BatchNorm1d(self.input_size),
        #                  # nn.LeakyReLU(0.2, True),
        #              )
        
        # self.imageDense = nn.Sequential(
        #                  nn.Linear(self.input_size, self.input_size),
        #                  nn.BatchNorm1d(self.input_size),
        #                  nn.LeakyReLU(0.2, True),
        #              )
        # self.out = nn.Sequential(
        #                  nn.Linear(3072, self.output_size),
        #                  nn.BatchNorm1d(self.output_size),
        #                  nn.LeakyReLU(0.2, True),
        #              )
#         self.dense = nn.Sequential(
#                          nn.Linear(self.input_size * 2 + num_classes, 1024),
#                          nn.Linear(1024, self.input_size),
# #                          nn.LeakyReLU(0.2, True),
#                      )
        
        # self.conv = nn.Conv2d(self.input_shape[0]*2, input_shape[0], kernel_size=3, stride=1, padding=1)
        
        # self.conv1 = self.conv_block(input_shape[0], CHANNEL_SIZE)
        # self.conv2 = self.conv_block(CHANNEL_SIZE, CHANNEL_SIZE * 2)
        # self.deconv31, self.deconv32 = self.deconv_block(CHANNEL_SIZE * 2, CHANNEL_SIZE * 4, CHANNEL_SIZE * 2),    self.deconv_block(CHANNEL_SIZE * 8, CHANNEL_SIZE * 4, CHANNEL_SIZE * 2, 1)
        # self.deconv41, self.deconv42 = self.deconv_block(CHANNEL_SIZE * 4, CHANNEL_SIZE * 2, CHANNEL_SIZE),     self.deconv_block(CHANNEL_SIZE * 4, CHANNEL_SIZE * 2, CHANNEL_SIZE, 1)
        # self.finalLayer = self.final_block(CHANNEL_SIZE * 2, CHANNEL_SIZE, self.output_shape[0])

        self.conv1 = self.conv_block(input_shape[0], CHANNEL_SIZE)
        self.conv2 = self.conv_block(CHANNEL_SIZE, CHANNEL_SIZE * 2)
        self.conv3 = self.conv_block(CHANNEL_SIZE * 2, CHANNEL_SIZE * 4)
        self.conv4 = self.conv_block(CHANNEL_SIZE * 4, CHANNEL_SIZE * 8)

        self.deconv11, self.deconv12 = self.deconv_block(CHANNEL_SIZE * 8, CHANNEL_SIZE * 16, CHANNEL_SIZE * 8),   self.deconv_block(CHANNEL_SIZE * 8, CHANNEL_SIZE * 16, CHANNEL_SIZE * 8, 1)
        self.deconv21, self.deconv22 = self.deconv_block(CHANNEL_SIZE * 16, CHANNEL_SIZE * 8, CHANNEL_SIZE * 4),   self.deconv_block(CHANNEL_SIZE * 16, CHANNEL_SIZE * 8, CHANNEL_SIZE * 4, 1)
        self.deconv31, self.deconv32 = self.deconv_block(CHANNEL_SIZE * 8, CHANNEL_SIZE * 4, CHANNEL_SIZE * 2),    self.deconv_block(CHANNEL_SIZE * 8, CHANNEL_SIZE * 4, CHANNEL_SIZE * 2, 1)
        self.deconv41, self.deconv42 = self.deconv_block(CHANNEL_SIZE * 4, CHANNEL_SIZE * 2, CHANNEL_SIZE),     self.deconv_block(CHANNEL_SIZE * 4, CHANNEL_SIZE * 2, CHANNEL_SIZE, 1)

        self.finalLayer = self.final_block(CHANNEL_SIZE * 2, CHANNEL_SIZE, self.output_shape[0])

        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgPool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x, y, lam):
        # yDense = self.labelDense(y)
        # x1Dense = self.imageDense(x1.view(x1.shape[0], -1))
        # x2Dense = self.imageDense(x2.view(x2.shape[0], -1))
        # x = torch.cat([x1Dense, x2Dense, yDense], 1).view(x1.shape[0], self.output_shape[0]*3, self.output_shape[1], self.output_shape[2])
        # x = torch.cat([x, x1, x2], 1).view(x1.shape[0], self.output_shape[0]*3, self.output_shape[1], self.output_shape[2])

        layer_mix = np.random.randint(0, 8)
        indices = np.random.permutation(x.size(0))
        shuffled = x[indices]

        if(layer_mix == 0):
            x, target_reweighted = mixup_process(x, y, indices, lam)
        conv1 = self.conv1(x)
        pool1 = self.maxPool(conv1)

        if(layer_mix == 1):
            pool1, target_reweighted = mixup_process(pool1, y, indices, lam)
        conv2 = self.conv2(pool1)
        pool2 = self.maxPool(conv2)

        if(layer_mix == 2):
            pool2, target_reweighted = mixup_process(pool2, y, indices, lam)
        conv3 = self.conv3(pool2)
        pool3 = self.maxPool(conv3)

        if(layer_mix == 3):
            pool3, target_reweighted = mixup_process(pool3, y, indices, lam)
        conv4 = self.conv4(pool3)
        pool4 = self.maxPool(conv4)

        if(layer_mix == 4):
            pool4, target_reweighted = mixup_process(pool4, y, indices, lam)
        bottleneck = self.deconv11(pool4) if pool3.shape[2]%2 == 0 else self.deconv12(pool4)
        concat1 = self.crop_and_concat(bottleneck, conv4)

        if(layer_mix == 5):
            concat1, target_reweighted = mixup_process(concat1, y, indices, lam)
        deconv1 = self.deconv21(concat1) if pool2.shape[2]%2 == 0 else self.deconv22(concat1)
        concat2 = self.crop_and_concat(deconv1, conv3)

        if(layer_mix == 6):
            concat2, target_reweighted = mixup_process(concat2, y, indices, lam)
        deconv2 = self.deconv31(concat2) if pool1.shape[2]%2 == 0 else self.deconv32(concat2)
        concat3 = self.crop_and_concat(deconv2, conv2)

        if(layer_mix == 7):
            concat3, target_reweighted = mixup_process(concat3, y, indices, lam)
        deconv3 = self.deconv41(concat3) if conv1.shape[2]%2 == 0 else self.deconv42(concat3)
        concat4 = self.crop_and_concat(deconv3, conv1)

        # conv1 = self.conv1(x)
        # pool1 = self.maxPool(conv1)
        # conv2 = self.conv2(pool1)
        # pool2 = self.maxPool(conv2)
        # # conv3 = self.conv3(pool2)
        # # pool3 = self.maxPool(conv3)
        # # conv4 = self.conv4(pool3)
        # # pool4 = self.maxPool(conv4)

        # bottleneck = self.deconv31(pool2) if pool1.shape[2]%2 == 0 else self.deconv31(pool2)
        # concat1 = self.crop_and_concat(bottleneck, conv2)
        # deconv3 = self.deconv41(concat1) if conv1.shape[2]%2 == 0 else self.deconv42(concat1)
        # concat4 = self.crop_and_concat(deconv3, conv1)

        # output = concat4
        if(layer_mix == 8):
            concat4, target_reweighted = mixup_process(concat4, y, indices, lam)
        output = torch.tanh(self.finalLayer(concat4))

        return output, shuffled, target_reweighted