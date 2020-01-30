import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DoubleConv(nn.Module):
	"""(convolution => [BN] => ReLU) * 2"""

	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.double_conv = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		return self.double_conv(x)


class Down(nn.Module):
	"""Downscaling with maxpool then double conv"""

	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.maxpool_conv = nn.Sequential(
			nn.MaxPool2d(2),
			DoubleConv(in_channels, out_channels)
		)

	def forward(self, x):
		return self.maxpool_conv(x)


class Up(nn.Module):
	"""Upscaling then double conv"""

	def __init__(self, in_channels, out_channels, bilinear=True):
		super().__init__()

		# if bilinear, use the normal convolutions to reduce the number of channels
		if bilinear:
			self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
		else:
			self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

		self.conv = DoubleConv(in_channels, out_channels)

	def forward(self, x1, x2):
		x1 = self.up(x1)
		# input is CHW
		diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
		diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

		x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
						diffY // 2, diffY - diffY // 2])
		# if you have padding issues, see
		# https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
		# https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
		x = torch.cat([x2, x1], dim=1)
		return self.conv(x)


class OutConv(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(OutConv, self).__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

	def forward(self, x):
		return self.conv(x)

def mixup_process(x, y, lam, indices):
	x_out = x * lam + x[indices] * (1 - lam)
	y_out = y * lam + y[indices] * (1 - lam)

	return x_out, y_out

class UNet(nn.Module):
	def __init__(self, input_shape, output_shape, num_classes, bilinear=True):
		super(UNet, self).__init__()
		self.input_shape = input_shape
		self.output_shape = output_shape
		self.num_classes = num_classes
		self.bilinear = bilinear

		self.inc = DoubleConv(input_shape[0], 64)
		self.down1 = Down(64, 128)
		self.down2 = Down(128, 256)
		self.down3 = Down(256, 512)
		self.down4 = Down(512, 512)
		self.up1 = Up(1024, 256, bilinear)
		self.up2 = Up(512, 128, bilinear)
		self.up3 = Up(256, 64, bilinear)
		self.up4 = Up(128, 64, bilinear)
		self.outc = OutConv(64, output_shape[0])

	def forward(self, x, y, lam):
		layer_mix = np.random.randint(0, 4)
		indices = np.random.permutation(x.size(0))
		x_shuffled = x[indices]

		if(layer_mix == 0):
			x, y_mix = mixup_process(x, y, lam, indices)
		x1 = self.inc(x)
		x2 = self.down1(x1)

		if(layer_mix == 1):
			x2, y_mix = mixup_process(x2, y, lam, indices)
		x3 = self.down2(x2)

		if(layer_mix == 2):
			x3, y_mix = mixup_process(x3, y, lam, indices)
		x4 = self.down3(x3)

		if(layer_mix == 3):
			x4, y_mix = mixup_process(x4, y, lam, indices)
		x5 = self.down4(x4)

		if(layer_mix == 4):
			x5, y_mix = mixup_process(x5, y, lam, indices)
		x = self.up1(x5, x4)

		if(layer_mix == 5):
			x, y_mix = mixup_process(x, y, lam, indices)
		x = self.up2(x, x3)

		if(layer_mix == 6):
			x, y_mix = mixup_process(x, y, lam, indices)
		x = self.up3(x, x2)

		if(layer_mix == 7):
			x, y_mix = mixup_process(x, y, lam, indices)
		x = self.up4(x, x1)

		if(layer_mix == 8):
			x, y_mix = mixup_process(x, y, lam, indices)
		x_mix = self.outc(x)

		return x_mix, x_shuffled, y_mix