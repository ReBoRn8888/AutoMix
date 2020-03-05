import torch
import torch.utils.model_zoo as model_zoo
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from util import get_lambda, mixup_process, to_one_hot

__all__ = ['ResNet', 'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101',
		   'ResNet152']


model_urls = {
	'ResNet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
	'ResNet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
	'ResNet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
	'ResNet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
	'ResNet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
	"""3x3 convolution with padding"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
					 padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
	"""1x1 convolution"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(BasicBlock, self).__init__()
		self.conv1 = conv3x3(inplanes, planes, stride)
		self.bn1 = nn.BatchNorm2d(planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(planes, planes)
		self.bn2 = nn.BatchNorm2d(planes)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample is not None:
			identity = self.downsample(x)

		out += identity
		out = self.relu(out)

		return out


class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(Bottleneck, self).__init__()
		self.conv1 = conv1x1(inplanes, planes)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = conv3x3(planes, planes, stride)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv3 = conv1x1(planes, planes * self.expansion)
		self.bn3 = nn.BatchNorm2d(planes * self.expansion)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			identity = self.downsample(x)

		out += identity
		out = self.relu(out)

		return out


class ResNet(nn.Module):

	def __init__(self, block, layers, num_classes, input_shape, zero_init_residual=False):
		super(ResNet, self).__init__()
		self.inplanes = 64
		# self.conv1 = nn.Conv2d(input_shape[0], 64, kernel_size=7, stride=2, padding=3, bias=False)
		self.conv1 = nn.Conv2d(input_shape[0], 64, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
		self.fc = nn.Linear(512 * block.expansion, num_classes)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

		# Zero-initialize the last BN in each residual branch,
		# so that the residual branch starts with zeros, and each residual block behaves like an identity.
		# This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
		if zero_init_residual:
			for m in self.modules():
				if isinstance(m, Bottleneck):
					nn.init.constant_(m.bn3.weight, 0)
				elif isinstance(m, BasicBlock):
					nn.init.constant_(m.bn2.weight, 0)

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				conv1x1(self.inplanes, planes * block.expansion, stride),
				nn.BatchNorm2d(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for _ in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)

	def forward(self, x, y=None, manifoldMixup=False, mixup_alpha=2.0, layer_mix=None):
		if(manifoldMixup):
			if(layer_mix == None):
				layer_mix = random.randint(0, 2)

			if mixup_alpha is not None:
				lam = get_lambda(mixup_alpha)
				lam = torch.from_numpy(np.array([lam]).astype('float32')).cuda()

			if(layer_mix == 0):
				x, target_reweighted = mixup_process(x, y, lam=lam)
			
			x = self.conv1(x)
			x = self.bn1(x)
			x = self.relu(x)
			# x = self.maxpool(x)
			x = self.layer1(x)
	
			if(layer_mix == 1):
				x, target_reweighted = mixup_process(x, y, lam=lam)

			x = self.layer2(x)
	
			if(layer_mix == 2):
				x, target_reweighted = mixup_process(x, y, lam=lam)

			x = self.layer3(x)
			
			if(layer_mix == 3):
				x, target_reweighted = mixup_process(x, y, lam=lam)

			x = self.layer4(x)
			
			if(layer_mix == 4):
				x, target_reweighted = mixup_process(x, y, lam=lam)

			x = self.avgpool(x)
			# x = F.avg_pool2d(x, 4)
			x = x.view(x.size(0), -1)
			x = self.fc(x)
			
			if(layer_mix == 5):
				x, target_reweighted = mixup_process(x, y, lam=lam)
			
			return x, target_reweighted

		else:
			x = self.conv1(x)
			x = self.bn1(x)
			x = self.relu(x)
			# x = self.maxpool(x)

			x = self.layer1(x)
			x = self.layer2(x)
			x = self.layer3(x)
			x = self.layer4(x)

			x = self.avgpool(x)
			x = x.view(x.size(0), -1)
			x = self.fc(x)

			return x


def ResNet18(num_classes, input_shape, pretrained=False):
	model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes, input_shape)
	if pretrained:
		model.load_state_dict(model_zoo.load_url(model_urls['ResNet18']))
	return model

def ResNet34(num_classes, input_shape, pretrained=False):
	model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes, input_shape)
	if pretrained:
		model.load_state_dict(model_zoo.load_url(model_urls['ResNet34']))
	return model

def ResNet50(num_classes, input_shape, pretrained=False):
	model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes, input_shape)
	if pretrained:
		model.load_state_dict(model_zoo.load_url(model_urls['ResNet50']))
	return model

def ResNet101(num_classes, input_shape, pretrained=False):
	model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes, input_shape)
	if pretrained:
		model.load_state_dict(model_zoo.load_url(model_urls['ResNet101']))
	return model

def ResNet152(num_classes, input_shape, pretrained=False):
	model = ResNet(Bottleneck, [3, 8, 36, 3], num_classes, input_shape)
	if pretrained:
		model.load_state_dict(model_zoo.load_url(model_urls['ResNet152']))
	return model
