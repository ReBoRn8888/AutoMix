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
			# nn.ReLU(inplace=True),
			nn.LeakyReLU(0.2, True),
			nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
			nn.BatchNorm2d(out_channels),
			# nn.ReLU(inplace=True)
			nn.LeakyReLU(0.2, True),
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

	def __init__(self, in_channels, out_channels, bilinear=False):
		super().__init__()

		# if bilinear, use the normal convolutions to reduce the number of channels
		if bilinear:
			self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
		else:
			self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
			# self.up = nn.ConvTranspose2d(in_channels // 3, in_channels // 3, kernel_size=2, stride=2)

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
		self.conv = nn.Sequential(
				nn.Conv2d(in_channels, out_channels, kernel_size=1),
				# nn.Sigmoid(),
				nn.Tanh(),
			)

	def forward(self, x):
		return self.conv(x)

class Reshape(nn.Module):
	def __init__(self, shape):
		super(Reshape, self).__init__()
		self.shape = shape

	def forward(self, x):
		return x.view(self.shape)

class SEModule(nn.Module):
	def __init__(self, channel, reduction=16):
		super(SEModule, self).__init__()
		self.globalAvgPool = nn.AdaptiveAvgPool2d(1)
		self.fc = nn.Sequential(
				nn.Linear(channel, channel//reduction, bias=False),
				# nn.ReLU(True),
				nn.LeakyReLU(0.2, True),
				nn.Linear(channel//reduction, channel, bias=False),
				nn.Sigmoid()
			)

	def forward(self, x):
		batchSize, channel, _, _ = x.size()
		y = self.globalAvgPool(x).view(batchSize, channel)
		y = self.fc(y).view(batchSize, channel, 1, 1)
		return y

def mixup_process(x, y, lam, indices):
	x_out = x * lam + x[indices] * (1 - lam)
	y_out = y * lam + y[indices] * (1 - lam)

	return x_out, y_out

class UNet(nn.Module):
	def __init__(self, input_shape, output_shape, num_classes, bilinear=False):
		super(UNet, self).__init__()
		self.input_shape = input_shape
		self.output_shape = output_shape
		self.num_classes = num_classes
		self.bilinear = bilinear
		self.input_size = input_shape[0] * input_shape[1] * input_shape[2]
		self.output_size = output_shape[0] * output_shape[1] * output_shape[2]

		# linearSize = 1024
		channelSize = 32

		self.merge5 = DoubleConv(channelSize*16, channelSize*8)
		self.merge4 = DoubleConv(channelSize*16, channelSize*8)
		self.merge3 = DoubleConv(channelSize*8, channelSize*4)
		self.merge2 = DoubleConv(channelSize*4, channelSize*2)
		self.merge1 = DoubleConv(channelSize*2, channelSize)

		# self.dense1 = nn.Sequential(
		# 			nn.Linear(channelSize*input_shape[1]*input_shape[2], channelSize),
		# 			nn.BatchNorm1d(channelSize),
		# 			nn.ReLU(True),
		# 		)
		# self.dense2 = nn.Sequential(
		# 			nn.Linear(channelSize*2*round(input_shape[1]//2)*round(input_shape[2]//2), channelSize*2),
		# 			nn.BatchNorm1d(channelSize*2),
		# 			nn.ReLU(True),
		# 		)
		# self.dense3 = nn.Sequential(
		# 			nn.Linear(channelSize*4*round(input_shape[1]//4)*round(input_shape[2]//4), channelSize*4),
		# 			nn.BatchNorm1d(channelSize*4),
		# 			nn.ReLU(True),
		# 		)
		# self.dense4 = nn.Sequential(
		# 			nn.Linear(channelSize*8*round(input_shape[1]//8)*round(input_shape[2]//8), channelSize*8),
		# 			nn.BatchNorm1d(channelSize*8),
		# 			nn.ReLU(True),
		# 		)
		# self.dense5 = nn.Sequential(
		# 			nn.Linear(channelSize*8*round(input_shape[1]//16)*round(input_shape[2]//16), channelSize*8),
		# 			nn.BatchNorm1d(channelSize*8),
		# 			nn.ReLU(True),
		# 		)

		self.inc = DoubleConv(input_shape[0], channelSize)
		# self.inc = DoubleConv(input_shape[0]*2+1, channelSize)
		self.down1 = Down(channelSize, channelSize*2)
		self.down2 = Down(channelSize*2, channelSize*4)
		self.down3 = Down(channelSize*4, channelSize*8)
		self.down4 = Down(channelSize*8, channelSize*8)
		self.up1 = Up(channelSize*16, channelSize*4, bilinear)
		self.up2 = Up(channelSize*8, channelSize*2, bilinear)
		self.up3 = Up(channelSize*4, channelSize, bilinear)
		self.up4 = Up(channelSize*2, channelSize, bilinear)
		self.outc = OutConv(channelSize, output_shape[0])

		# self.globalAvgPool = nn.AdaptiveAvgPool2d((1, 1))
		self.SEmodule1 = SEModule(channelSize)
		self.SEmodule2 = SEModule(channelSize*2)
		self.SEmodule3 = SEModule(channelSize*4)
		self.SEmodule4 = SEModule(channelSize*8)
		self.SEmodule5 = SEModule(channelSize*8)

		# self.SEmodule2 = nn.Sequential(
		# 			nn.AdaptiveAvgPool2d((1, 1)),
		# 			Reshape((-1, channelSize*2)),
		# 			nn.Linear(channelSize*2, channelSize*2//16),
		# 			nn.ReLU(True),
		# 			nn.Linear(channelSize*2//16, channelSize*2),
		# 			nn.Sigmoid(),
		# 		)
		# self.SEmodule3 = nn.Sequential(
		# 			nn.AdaptiveAvgPool2d((1, 1)),
		# 			Reshape((-1, channelSize*4)),
		# 			nn.Linear(channelSize*4, channelSize*4//16),
		# 			nn.ReLU(True),
		# 			nn.Linear(channelSize*4//16, channelSize*4),
		# 			nn.Sigmoid(),
		# 		)
		# self.SEmodule4 = nn.Sequential(
		# 			nn.AdaptiveAvgPool2d((1, 1)),
		# 			Reshape((-1, channelSize*8)),
		# 			nn.Linear(channelSize*8, channelSize*8//16),
		# 			nn.ReLU(True),
		# 			nn.Linear(channelSize*8//16, channelSize*8),
		# 			nn.Sigmoid(),
		# 		)
		# self.SEmodule5 = nn.Sequential(
		# 			nn.AdaptiveAvgPool2d((1, 1)),
		# 			Reshape((-1, channelSize*8)),
		# 			nn.Linear(channelSize*8, channelSize*8//16),
		# 			nn.ReLU(True),
		# 			nn.Linear(channelSize*8//16, channelSize*8),
		# 			nn.Sigmoid(),
		# 		)

		# Backup
		# self.inc = DoubleConv(input_shape[0], channelSize)
		# self.down1 = Down(channelSize, channelSize*2)
		# self.down2 = Down(channelSize*2, channelSize*4)
		# self.down3 = Down(channelSize*4, channelSize*8)
		# self.down4 = Down(channelSize*8, channelSize*8)
		# self.up1 = Up(channelSize*16, channelSize*4, bilinear)
		# self.up2 = Up(channelSize*8, channelSize*2, bilinear)
		# self.up3 = Up(channelSize*4, channelSize, bilinear)
		# self.up4 = Up(channelSize*2, channelSize, bilinear)
		# self.outc = OutConv(channelSize, output_shape[0])

	def clean(self, features, logits, indices, label):
		r1, r2 = label
		# avg = self.globalAvgPool(logits).view(logits.shape[0], logits.shape[1])
		# num1 = round(logits.shape[1] * r1)
		# num2 = logits.shape[1] - num1
		# topk1 = torch.topk(logits, k=max(num1, num2), dim=1).indices
		# x = torch.zeros(features.shape).cuda()
		# for i, idx in enumerate(topk1[:, :num1]):
		#     x[i, idx] = 1
		# a = features * x

		# topk2 = topk1[indices]
		# x = torch.zeros(features.shape).cuda()
		# for i, idx in enumerate(topk2[:, :num2]):
		#     x[i, idx] = 1
		# b = features[indices] * x

		ones = torch.ones(logits.shape).cuda()
		zeros = torch.zeros(logits.shape).cuda()
		maxx = torch.max(logits, 1).values.view(logits.shape[0], 1, 1, 1)
		minn = torch.min(logits, 1).values.view(logits.shape[0], 1, 1, 1)
		logits = (logits - minn) / (maxx - minn)
		a = features * torch.where(logits >= r2, ones, zeros).expand_as(features)
		b = features[indices] * torch.where(logits[indices] >= r1, ones, zeros).expand_as(features)

		return a, b

	def forward(self, a, indices, label):
		# By concat
		# label = self.labelDense(label).view(x1.shape[0], 1, self.input_shape[1], self.input_shape[2])
		# x = torch.cat([x1, label, x2], 1)

		# By Linear1
		# a = self.xDense(x1.view(x1.shape[0], -1))
		# b = self.xDense(x2.view(x2.shape[0], -1))
		# x = torch.cat([a, label, b], 1)
		# x = self.outDense(x).view(x1.shape[0], x1.shape[1], x1.shape[2], x1.shape[3])

		# By Linear2
		# x = torch.cat([a.view(a.shape[0], -1), b.view(b.shape[0], -1), label], 1)
		# x = self.fc(x).view(-1, a.shape[1], a.shape[2], a.shape[3])

		# print(self.inc(a).shape) # [50, 32, 28, 28]
		a1 = self.inc(a)
		# print(a1.shape)
		# print(self.SEmodule1(a1).shape)
		a1, b1 = self.clean(a1, self.SEmodule1(a1), indices, label)
		a2 = self.down1(a1)
		a2, b2 = self.clean(a2, self.SEmodule2(a2), indices, label)
		a3 = self.down2(a2)
		a3, b3 = self.clean(a3, self.SEmodule3(a3), indices, label)
		a4 = self.down3(a3)
		a4, b4 = self.clean(a4, self.SEmodule4(a4), indices, label)
		a5 = self.down4(a4)
		a5, b5 = self.clean(a5, self.SEmodule5(a5), indices, label)

		cated = self.merge5(torch.cat([a5, b5], 1))
		u1 = self.up1(cated, self.merge4(torch.cat([a4, b4], 1)))
		u2 = self.up2(u1, self.merge3(torch.cat([a3, b3], 1)))		
		u3 = self.up3(u2, self.merge2(torch.cat([a2, b2], 1)))
		u4 = self.up4(u3, self.merge1(torch.cat([a1, b1], 1)))

		x_mix = self.outc(u4)

		return x_mix

# ------------------------------------------------------------------------------
		# a1, b1 = self.clean(self.inc(a), indices, label)
		# a2, b2 = self.clean(self.down1(a1), indices, label)
		# a3, b3 = self.clean(self.down2(a2), indices, label)
		# a4, b4 = self.clean(self.down3(a3), indices, label)
		# a5, b5 = self.clean(self.down4(a4), indices, label)

# -----------------------------------------------------------------------------
		# cleanLayer = np.random.randint(0, 5)
		# if(cleanLayer == 0):
		# 	a5 = self.clean(a5, label[0])
		# 	b5 = self.clean(b5, label[1])
		# cated = self.merge5(torch.cat([a5, b5], 1))

		# if(cleanLayer == 1):
		# 	a4 = self.clean(a4, label[0])
		# 	b4 = self.clean(b4, label[1])
		# u1 = self.up1(cated, self.merge4(torch.cat([a4, b4], 1)))
		
		# if(cleanLayer == 2):
		# 	a3 = self.clean(a3, label[0])
		# 	b3 = self.clean(b3, label[1])
		# u2 = self.up2(u1, self.merge3(torch.cat([a3, b3], 1)))
		
		# if(cleanLayer == 3):
		# 	a2 = self.clean(a2, label[0])
		# 	b2 = self.clean(b2, label[1])
		# u3 = self.up3(u2, self.merge2(torch.cat([a2, b2], 1)))
		
		# if(cleanLayer == 4):
		# 	a1 = self.clean(a1, label[0])
		# 	b1 = self.clean(b1, label[1])
		# u4 = self.up4(u3, self.merge1(torch.cat([a1, b1], 1)))

# ---------------------------------------------------------------------------------------------------------------

		# # label5 = self.labelDense5(label).view(a.shape[0], 1, a5.shape[2], a5.shape[3])
		# cated = self.merge5(torch.cat([a5*label[0][0], b5*label[0][1]], 1))
		# u1 = self.up1(cated, self.merge4(torch.cat([a4*label[0][0], b4*label[0][1]], 1)))
		# u2 = self.up2(u1, self.merge3(torch.cat([a3*label[0][0], b3*label[0][1]], 1)))
		# u3 = self.up3(u2, self.merge2(torch.cat([a2*label[0][0], b2*label[0][1]], 1)))
		# u4 = self.up4(u3, self.merge1(torch.cat([a1*label[0][0], b1*label[0][1]], 1)))

# ------------------------------------------------------------------------------------------------------
		
		# label = self.labelDense(label)
		# cated = self.merge(torch.cat([a5.view(a5.shape[0], -1), b5.view(b5.shape[0], -1), label], 1)).view(a5.shape[0], a5.shape[1], a5.shape[2], a5.shape[3])
		# # label4 = self.labelDense4(label).view(a.shape[0], 1, a4.shape[2], a4.shape[3])
		# # u1 = self.up1(cated, self.merge4(torch.cat([a4.view(a4.shape[0], -1), b4.view(b4.shape[0], -1), label4], 1)))
		# u1 = self.up1(cated, self.merge4(torch.cat([a4, b4], 1)))
		# u2 = self.up2(u1, self.merge3(torch.cat([a3, b3], 1)))
		# u3 = self.up3(u2, self.merge2(torch.cat([a2, b2], 1)))
		# u4 = self.up4(u3, self.merge1(torch.cat([a1, b1], 1)))

# ------------------------------------------------------------------------------------------------------

		# label5 = self.labelDense5(label).view(a.shape[0], 1, a5.shape[2], a5.shape[3])
		# cated = self.merge5(torch.cat([a5, b5, label5], 1))

		# label4 = self.labelDense4(label).view(a.shape[0], 1, a4.shape[2], a4.shape[3])
		# u1 = self.up1(cated, self.merge4(torch.cat([a4, b4, label4], 1)))
		
		# label3 = self.labelDense3(label).view(a.shape[0], 1, a3.shape[2], a3.shape[3])
		# u2 = self.up2(u1, self.merge3(torch.cat([a3, b3, label3], 1)))
		
		# label2 = self.labelDense2(label).view(a.shape[0], 1, a2.shape[2], a2.shape[3])
		# u3 = self.up3(u2, self.merge2(torch.cat([a2, b2, label2], 1)))
		
		# label1 = self.labelDense1(label).view(a.shape[0], 1, a1.shape[2], a1.shape[3])
		# u4 = self.up4(u3, self.merge1(torch.cat([a1, b1, label1], 1)))
		
		# x_mix = self.outc(u4)



		# # Backup
		# x1 = self.inc(x)
		# x2 = self.down1(x1)
		# x3 = self.down2(x2)
		# x4 = self.down3(x3)
		# x5 = self.down4(x4)
		# u1 = self.up1(x5, x4)
		# u2 = self.up2(u1, x3)
		# u3 = self.up3(u2, x2)
		# u4 = self.up4(u3, x1)
		# x_mix = self.outc(u4)

		# return x_mix






		# print('x1.shape = {}'.format(x1.shape))
		# print('x2.shape = {}'.format(x2.shape))
		# print('x3.shape = {}'.format(x3.shape))
		# print('x4.shape = {}'.format(x4.shape))
		# print('x5.shape = {}'.format(x5.shape))
		# print('up1.shape = {}'.format(u1.shape))
		# print('up2.shape = {}'.format(u2.shape))
		# print('up3.shape = {}'.format(u3.shape))
		# print('up4.shape = {}'.format(u4.shape))

		# layer_mix = np.random.randint(0, 4)
		# if(layer_mix == 0):
		# 	x, y_mix = mixup_process(x, y, lam, indices)
		# x1 = self.inc(x)
		# x2 = self.down1(x1)

		# if(layer_mix == 1):
		# 	x2, y_mix = mixup_process(x2, y, lam, indices)
		# x3 = self.down2(x2)

		# if(layer_mix == 2):
		# 	x3, y_mix = mixup_process(x3, y, lam, indices)
		# x4 = self.down3(x3)

		# if(layer_mix == 3):
		# 	x4, y_mix = mixup_process(x4, y, lam, indices)
		# x5 = self.down4(x4)

		# if(layer_mix == 4):
		# 	x5, y_mix = mixup_process(x5, y, lam, indices)
		# x = self.up1(x5, x4)

		# if(layer_mix == 5):
		# 	x, y_mix = mixup_process(x, y, lam, indices)
		# x = self.up2(x, x3)

		# if(layer_mix == 6):
		# 	x, y_mix = mixup_process(x, y, lam, indices)
		# x = self.up3(x, x2)

		# if(layer_mix == 7):
		# 	x, y_mix = mixup_process(x, y, lam, indices)
		# x = self.up4(x, x1)

		# if(layer_mix == 8):
		# 	x, y_mix = mixup_process(x, y, lam, indices)
		# x_mix = self.outc(x)