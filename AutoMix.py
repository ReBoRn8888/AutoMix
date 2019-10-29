import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets, models
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.optim import lr_scheduler
from torchsummary import summary

import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image
from scipy.io import loadmat
import numpy as np
import argparse
import pickle
import copy
import pprint
import time
import sys
import os
import ot
import cv2
import datetime
import logging

def add_path(path):
	if path not in sys.path:
		sys.path.insert(0, path)
lib_path = os.path.abspath('pytorch_models')
add_path(lib_path)

from pytorch_models import *

# python AutoMix.py --method=baseline \
# 				  --arch=resnet18  \
# 				  --dataset=IMAGENET \
# 				  --data_dir=/media/reborn/Others2/ImageNet \
# 				  --batch_size=32 \
# 				  --lr=0.01 \
# 				  --gpu=0,1 \
# 				  --num_workers=8 \
# 				  --parallel=True \
# 				  --log_path=./automix.log
def parse_args():
	"""Parse input arguments."""
	parser = argparse.ArgumentParser(description='AutoMix: Mixup Networks for Sample Interpolation via Cooperative Barycenter Learning')
	parser.add_argument('--method', dest='method', default='baseline', type=str, choices=['baseline', 'bc', 'mixup', 'automix', 'adamixup'], help='Method : [baseline, bc, mixup, automix, adamixup]')
	parser.add_argument('--arch', dest='arch', default='resnet18', type=str, choices=['mynet', 'resnet18'], help='Backbone architecture : [mynet, resnet18]')
	parser.add_argument('--dataset', dest='dataset', default='IMAGENET', type=str, choices=['IMAGENET', 'CIFAR10', 'CIFAR100', 'MNIST', 'FASHION-MNIST', 'GTSRB', 'MIML'], help='Dataset to be trained : [IMAGENET, CIFAR10, CIFAR100, MNIST, FASHION-MNIST, GTSRB, MIML]')
	parser.add_argument('--data_dir', dest='data_dir', default=None, type=str, help='Path to the dataset')
	parser.add_argument('--batch_size', dest='batch_size', default=25, type=int, help='Batch_size for training')
	parser.add_argument('--gpu', dest='gpu', default='', type=str, help='GPU lists can be used')
	parser.add_argument('--lr', dest='lr', default=0.05, type=float, help='Learning rate')
	parser.add_argument('--num_workers', dest='num_workers', default=8, type=int, help='Num of multiple threads')
	parser.add_argument('--momentum', dest='momentum', default=0.9, type=float, help='Momentum for optimizer')
	parser.add_argument('--weight_decay', dest='weight_decay', default=5e-4, type=float, help='Weight_decay for optimizer')
	parser.add_argument('--parallel', dest='parallel', default=False, type=bool, help='Train parallelly with multi-GPUs?')
	parser.add_argument('--log_path', dest='log_path', default=os.path.join(os.getcwd(),'{}.log'.format(datetime.datetime.now().strftime('%Y-%m-%d|%H:%M:%S'))), type=str, help='Path to the dataset')

	args = parser.parse_args()
	return args

class myLoss(nn.Module):
	def __init__(self, mtype='baseline'):
		super(myLoss, self).__init__()
		self.type = mtype.lower()
		
	def forward(self, pred, truth):
#         pred += 1e-9
#         truth += 1e-9
		if(self.type in ['bc', 'automix']):
#             entropy = - torch.sum(truth[truth > 0] * torch.log(truth[truth > 0]), 1)
			entropy = - torch.sum((truth+1e-9) * torch.log((truth+1e-9)), 1)
			crossEntropy = - torch.sum((truth+1e-9) * F.log_softmax((pred+1e-9), 1), 1)
			loss = crossEntropy - entropy
		else:
			pred = F.log_softmax((pred+1e-9), 1)
			loss = -torch.sum((pred+1e-9) * (truth+1e-9), 1)
		return loss

class myDataset(Dataset):
	def __init__(self, images, labels, classes=None, transform=None, mtype='baseline', dtype='mnist', onehot=True):
		self.images = images
		self.labels = labels
		self.transform = transform
		self.classes = classes
		self.num_classes = len(self.classes)
		self.mtype = mtype.lower()
		self.dtype = dtype.lower()
		self.onehot = onehot
		
	def to_onehot(self, label):
		label = torch.unsqueeze(torch.unsqueeze(label, 0), 1)
		label = torch.zeros(1, self.num_classes).scatter_(1, label, 1)
		label = torch.squeeze(label)
		return label
		
	def __len__(self):
		return len(self.images)
	
	def __getitem__(self, index):
		if(self.mtype == 'bc'):
			while True:  # Select two training examples
				id1 = index
				image1, label1 = self.images[id1], self.labels[id1]
				id2 = np.random.randint(0, self.__len__() - 1)
				image2, label2 = self.images[id2], self.labels[id2]
				if label1 != label2:
					break
			if(self.transform):
				image1 = self.transform(Image.fromarray(np.uint8(image1)))
				image2 = self.transform(Image.fromarray(np.uint8(image2)))
			# Mix two images
			r = torch.rand(1)
			g1 = torch.std(image1)
			g2 = torch.std(image2)
			p = 1.0 / (1 + g1 / g2 * (1 - r) / r)
			image = ((image1 * p + image2 * (1 - p)) / np.sqrt(p ** 2 + (1 - p) ** 2))
			
			# Mix two labels
			label1 = self.to_onehot(label1)
			label2 = self.to_onehot(label2)
			label = (label1 * r + label2 * (1 - r)).float()
			return image, label
		else:
			image, label = self.images[index], self.labels[index]
			if(self.dtype == 'imagenet'):
				image = cv2.imread(image[0])
			if(self.onehot):
				label = self.to_onehot(label)
			if(self.transform):
				image = self.transform(Image.fromarray(np.uint8(image)))
			return image, label

def zero_mean(tensor, mean, std):
	if not torchvision.transforms.functional._is_tensor_image(tensor):
		raise TypeError('tensor is not a torch image.')
	# TODO: make efficient
	for t, m, s in zip(tensor, mean, std):
		t.sub_(m).sub_(torch.mean(t)).div_(s)
	return tensor

class ZeroMean(object):
	def __init__(self, mean, std):
		self.mean = mean
		self.std = std

	def __call__(self, tensor):
		return zero_mean(tensor, self.mean, self.std)

	def __repr__(self):
		return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def get_dataset(dataType, methodType, dataPath):
	if(dataType == 'IMAGENET'):
		shape = (3, 224, 224)
		lrDecayStep = [50, 75]
		n_epochs = 90
		# Data loading code
		if(methodType == 'bc'):
			normalize = ZeroMean
		else:
			normalize = transforms.Normalize
		transform_train = transforms.Compose([
			transforms.RandomResizedCrop(224),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		])

		transform_test = transforms.Compose([
			transforms.RandomResizedCrop(224),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		])

		traindir = os.path.join(dataPath, 'train')
		valdir = os.path.join(dataPath, 'validation')

		oriTrainDataset = datasets.ImageFolder(traindir)
		classes = list(oriTrainDataset.class_to_idx.keys())
		clsId2clsName = get_imageNet_classId2Name()
		num_classes = len(classes)
		images = oriTrainDataset.imgs
		labels = torch.Tensor(oriTrainDataset.targets).long()
		trainDataset = myDataset(images, labels, classes, transform_train, mtype=methodType, dtype=dataType)
		trainLoader = DataLoader(trainDataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

		oriTestDataset = datasets.ImageFolder(valdir)
		testImages = oriTestDataset.imgs
		testLabels = torch.Tensor(oriTestDataset.targets).long()
		testDataset = myDataset(testImages, testLabels, classes, transform_test, mtype=methodType, dtype=dataType)
		testLoader = DataLoader(testDataset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
	elif(dataType == 'CIFAR10'):
		classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
		num_classes = len(classes)
		shape = (3, 32, 32)
		lrDecayStep = [150, 225]
		n_epochs = 300

		if(methodType == 'bc'):
			normalize = ZeroMean
		else:
			normalize = transforms.Normalize
		transform_train = transforms.Compose([
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
		])

		transform_test = transforms.Compose([
			transforms.ToTensor(),
			normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
		])

		oriTrainDataset = datasets.CIFAR10(root=dataPath, 
												train=True, download=True, transform=transform_train)
		images = oriTrainDataset.data
		labels = torch.Tensor(oriTrainDataset.targets).long()
		trainDataset = myDataset(images, labels, classes, transform_train, mtype=methodType)
		trainLoader = DataLoader(trainDataset, batch_size=train_batch_size, shuffle=True, num_workers=1)

		oriTestDataset = datasets.CIFAR10(root=dataPath, 
											   train=False, download=False, transform=transform_test)
		testImages = oriTestDataset.data
		testLabels = torch.Tensor(oriTestDataset.targets).long()
		testDataset = myDataset(testImages, testLabels, classes, transform_test)
		testLoader = DataLoader(testDataset, batch_size=test_batch_size, shuffle=False, num_workers=1)
	if(dataType == 'CIFAR100'):
		classes = ['{}'.format(i) for i in range(100)]
		num_classes = len(classes)
		shape = (3, 32, 32)
		lrDecayStep = [150, 225]
		n_epochs = 300

		if(methodType == 'bc'):
			normalize = ZeroMean
		else:
			normalize = transforms.Normalize
		transform_train = transforms.Compose([
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
		])

		transform_test = transforms.Compose([
			transforms.ToTensor(),
			normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
		])

		oriTrainDataset = datasets.CIFAR100(root=dataPath, 
												train=True, download=True, transform=transform_train)
		images = oriTrainDataset.data
		labels = torch.Tensor(oriTrainDataset.targets).long()
		trainDataset = myDataset(images, labels, classes, transform_train, mtype=methodType)
		trainLoader = DataLoader(trainDataset, batch_size=train_batch_size, shuffle=True, num_workers=1)

		oriTestDataset = datasets.CIFAR100(root=dataPath, 
											   train=False, download=False, transform=transform_test)
		testImages = oriTestDataset.data
		testLabels = torch.Tensor(oriTestDataset.targets).long()
		testDataset = myDataset(testImages, testLabels, classes, transform_test)
		testLoader = DataLoader(testDataset, batch_size=test_batch_size, shuffle=False, num_workers=1)
	elif(dataType == 'GTSRB'):
		classes = ['{}'.format(i) for i in range(43)]
		num_classes = len(classes)
		shape = (3, 28, 28)
		lrDecayStep = [50, 75]
		n_epochs = 100

		if(methodType == 'bc'):
			normalize = ZeroMean
		else:
			normalize = transforms.Normalize
		transform_train = transforms.Compose([
			transforms.RandomCrop(28, padding=4),
			transforms.ToTensor(),
			normalize(mean=[0.3352, 0.3173, 0.3584], std=[0.2662, 0.2563, 0.2727]),
		])
		transform_test = transforms.Compose([
			transforms.ToTensor(),
			normalize(mean=[0.3352, 0.3173, 0.3584], std=[0.2662, 0.2563, 0.2727]),
		])

		with open('{}/39209-all/images.pkl'.format(dataPath), 'rb') as f:
			images = torch.from_numpy(pickle.load(f)).float()
		with open('{}/39209-all/labels.pkl'.format(dataPath), 'rb') as f:
			labels = torch.from_numpy(pickle.load(f))
			labels = torch.argmax(labels, 1)
		with open('{}/39209-all/testImages.pkl'.format(dataPath), 'rb') as f:
			testImages = torch.from_numpy(pickle.load(f)).float()
		with open('{}/39209-all/testLabels.pkl'.format(dataPath), 'rb') as f:
			testLabels = torch.from_numpy(pickle.load(f))
			testLabels = torch.argmax(testLabels, 1)
		trainDataset = myDataset(images, labels, classes, transform_train, mtype=methodType)
		trainLoader = DataLoader(trainDataset, batch_size=train_batch_size, shuffle=True, num_workers=1)

		testDataset = myDataset(testImages, testLabels, classes, transform_test)
		testLoader = DataLoader(testDataset, batch_size=test_batch_size, shuffle=False, num_workers=1)
	elif(dataType == 'MNIST'):
		classes = ['{}'.format(i) for i in range(10)]
		num_classes = len(classes)
		shape = (1, 28, 28)
		lrDecayStep = [50, 75]
		n_epochs = 100

		if(methodType == 'bc'):
			normalize = ZeroMean
		else:
			normalize = transforms.Normalize
		transform_train = transforms.Compose([
			transforms.RandomCrop(28, padding=4),
			transforms.ToTensor(),
			normalize(mean=[0.1307,], std=[0.3081,]),
		])
		transform_test = transforms.Compose([
			transforms.ToTensor(),
			normalize(mean=[0.1307,], std=[0.3081,]),
		])

		oriTrainDataset = datasets.MNIST(dataPath, 
									  train=True, download=False, transform=transform_train)
		images = oriTrainDataset.data
		labels = oriTrainDataset.targets
		trainDataset = myDataset(images, labels, classes, transform_train, mtype=methodType)
		trainLoader = DataLoader(trainDataset, batch_size=train_batch_size, shuffle=True, num_workers=1)

		oriTestDataset = datasets.MNIST(dataPath, 
									 train=False, download=False, transform=transform_test)
		testImages = oriTestDataset.data
		testLabels = oriTestDataset.targets
		testDataset = myDataset(testImages, testLabels, classes, transform_test)
		testLoader = DataLoader(testDataset, batch_size=test_batch_size, shuffle=False, num_workers=1)
	elif(dataType == 'FASHION-MNIST'):
		classes = ['{}'.format(i) for i in range(10)]
		num_classes = len(classes)
		shape = (1, 28, 28)
		lrDecayStep = [50, 75]
		n_epochs = 100

		if(methodType == 'bc'):
			normalize = ZeroMean
		else:
			normalize = transforms.Normalize
		transform_train = transforms.Compose([
			transforms.RandomCrop(28, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			normalize(mean=[0.2860,], std=[0.3530,]),
		])
		transform_test = transforms.Compose([
			transforms.ToTensor(),
			normalize(mean=[0.2860,], std=[0.3530,]),
		])

		oriTrainDataset = datasets.FashionMNIST(dataPath, 
									  train=True, download=True, transform=transform_train)
		images = oriTrainDataset.data
		labels = oriTrainDataset.targets
		trainDataset = myDataset(images, labels, classes, transform_train, mtype=methodType)
		trainLoader = DataLoader(trainDataset, batch_size=train_batch_size, shuffle=True, num_workers=1)

		oriTestDataset = datasets.FashionMNIST(dataPath, 
									 train=False, download=False, transform=transform_test)
		testImages = oriTestDataset.data
		testLabels = oriTestDataset.targets
		testDataset = myDataset(testImages, testLabels, classes, transform_test)
		testLoader = DataLoader(testDataset, batch_size=test_batch_size, shuffle=False, num_workers=1)
	
	return trainDataset, trainLoader, testDataset, testLoader, classes, num_classes, shape, lrDecayStep, n_epochs

def get_imageNet_classId2Name():
	m = loadmat('Dataset/ImageNet-meta.mat')
	clsId2clsName = dict()
	for i in range(len(m['synsets'])):
		clsId2clsName[m['synsets'][i][0][1][0]] = m['synsets'][i][0][2][0]
	return clsId2clsName

def shuffle(x, y):
	perm = np.random.permutation(len(x))
	x_shuffle = x[perm]
	y_shuffle = y[perm]
	return x_shuffle, y_shuffle

def get_cls_result(net, dataLoader):
	y_score, y_true = [], []
	for i, data in enumerate(dataLoader, 0):
		inputs, labels = data
		y_true.extend(labels.numpy())
		inputs = inputs.to(device)
		labels = labels.to(device)
		optimizer.zero_grad()
		with torch.set_grad_enabled(False):
			outputs = net(inputs)
			y_score.extend(outputs.cpu().detach().numpy())
	y_score = np.array(y_score)
	y_true = np.array(y_true)
	return y_true, y_score

def get_roc_data(y_true, y_score, n_classes, rocType='micro'):
	fpr = dict() 
	tpr = dict() 
	thres = dict()
	roc_auc = dict() 
	for i in range(n_classes): 
		fpr[i], tpr[i], thres[i] = roc_curve(y_true[:, i], y_score[:, i]) 
		roc_auc[i] = auc(fpr[i], tpr[i])
	if(rocType == "micro"):
		fprOut, tprOut, _ = roc_curve(y_true.ravel(), y_score.ravel())
	elif(rocType == "macro"):
		all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)])) 
		mean_tpr = np.zeros_like(all_fpr) 
		for i in range(n_classes): 
			mean_tpr += interp(all_fpr, fpr[i], tpr[i]) 
		mean_tpr /= n_classes 
		fprOut = all_fpr 
		tprOut = mean_tpr 
	aucOut = auc(fprOut, tprOut)
	ROC = dict()
	ROC['fpr'], ROC['tpr'], ROC['auc'] = fprOut, tprOut, aucOut
	return ROC

def ROC_plot(sorted_ROC):
	lw=2
	plt.figure(figsize=(10, 10)) 
	title = "{} AUC : ".format(dataset)
	for i, ROC in enumerate(sorted_ROC):
		if(i == len(sorted_ROC) - 1):
			title += "{}".format(ROC[0])
		else:
			title += "{} > ".format(ROC[0])
		plt.plot(ROC[1]["fpr"], ROC[1]["tpr"], 
				 label='{}\'s AUC = {}'.format(ROC[0], ROC[1]["auc"]), linestyle='--', linewidth=lw) 
	plt.plot([0, 1], [0, 1], 'k--', lw=lw) 
	plt.xlim([-0.02, 1.02])
	plt.ylim([-0.02, 1.02]) 
	plt.xlabel('False Positive Rate') 
	plt.ylabel('True Positive Rate') 
	plt.title(title) 
	plt.legend(loc="lower right") 
#     plt.savefig("{}/{}-{}{}-ROC.jpg".format(path, dataset, num_examples, suffix))
	plt.show()

def distribution(labels, type='normal'):
	result = {}
	if(type == 'onehot'):
		ll = np.argmax(labels, axis=1)
	else:
		ll = labels
	for i in set(ll):
		result[i] = ll.tolist().count(i)
	print("Label distribution: {}".format(result))
	logger.info("Label distribution: {}".format(result))

def cal_mean_std(dataset):
	dataLoader = DataLoader(dataset, batch_size=dataset.__len__(), shuffle=True, num_workers=1)
	it = iter(dataLoader)
	images, labels = it.next()
	images = np.array(images)
	mean = np.round(np.mean(images, axis=(0, 2, 3)), 4)
	std = np.round(np.std(images, axis=(0, 2, 3)), 4)
	return mean, std

def eval_total(net, dataLoader):
	start = time.time()
	correct = 0
	total = 0
	with torch.no_grad():
		for data in dataLoader:
			images, labels = data
			images = images.to(device)
			labels = labels.to(device)
			outputs = net(images)
			predicted = torch.argmax(outputs.data, 1)
			total += labels.size(0)
			correct += predicted.eq(torch.argmax(labels, 1)).cpu().sum().float().item()

	accTotal = 100 * correct / total
	duration = time.time() - start
	print('Accuracy of the network on the 10000 test images: {:.2f}% ({:.0f}mins {:.2f}s)'.format(accTotal, duration // 60, duration % 60))
	logger.info('Accuracy of the network on the 10000 test images: {:.2f}% ({:.0f}mins {:.2f}s)'.format(accTotal, duration // 60, duration % 60))
	return accTotal

def eval_per_class(net, dataLoader, classes):
	num_classes = len(classes)
	start = time.time()
	class_correct = list(0. for i in range(num_classes))
	class_total = list(0. for i in range(num_classes))
	with torch.no_grad():
		for data in dataLoader:
			images, labels = data
			images = images.to(device)
			labels = labels.to(device)
			outputs = net(images)
			predicted = torch.argmax(outputs, 1)
			c = predicted.eq(torch.argmax(labels, 1)).cpu().squeeze()
			for i in range(len(c)):
				label = torch.argmax(labels[i]).item()
				class_correct[label] += c[i].item()
				class_total[label] += 1
	print('class_correct\t:\t{}'.format(class_correct))
	logger.info('class_correct\t:\t{}'.format(class_correct))
	print('class_total\t:\t{}'.format(class_total))
	logger.info('class_total\t:\t{}'.format(class_total))
	accPerClass = dict()
	for i in range(num_classes):
		accPerClass[classes[i]] = 0 if class_correct[i] == 0 else '{:.2f}%'.format(100 * class_correct[i] / class_total[i])
	duration = time.time() - start
	print('Per class accuracy :')
	logger.info('Per class accuracy :')
	pprint.pprint(accPerClass)
	print('Duration for accPerClass : {:.0f}mins {:.2f}s'.format(duration // 60, duration % 60))
	logger.info('Duration for accPerClass : {:.0f}mins {:.2f}s'.format(duration // 60, duration % 60))
	return accPerClass

def plot_acc_loss(log, type, prefix='', suffix=''):
	trainAcc = log['acc']['train']
	trainLoss = log['loss']['train']
	valAcc = log['acc']['val']
	valLoss = log['loss']['val']
	if(type == 'loss'):
		plt.figure(figsize=(7, 5))
		plt.plot(trainLoss, label='Train_loss')
		plt.plot(valLoss, label='Test_loss')
		plt.title('Epoch - Loss')
		plt.xlabel('Epoch')
		plt.ylabel('Loss')
		plt.legend()
		figName = '{}loss{}.png'.format(prefix, suffix)
	elif(type == 'accuracy'):
		plt.figure(figsize=(7, 5))
		plt.plot(trainAcc, label='Train_acc')
		plt.plot(valAcc, label='Test_acc')
		plt.title('Epoch - Acuracy')
		plt.xlabel('Epoch')
		plt.ylabel('Accuracy')
#         plt.ylim(0, 1.01)
		plt.legend()
		figName = '{}accuracy{}.png'.format(prefix, suffix)
	elif(type == 'both'):
		fig, ax1 = plt.subplots(figsize=(7, 5))
		ax2 = plt.twinx()
		ax1.set_xlabel('Epoch')
		ax1.set_ylabel('Loss')
		ax2.set_ylabel('Accuracy')
		plt.ylim(0, 1.01)
		plt.title('Loss & Accuracy')

		l_trainLoss, = ax1.plot(trainLoss)
		l_testLoss, = ax1.plot(valLoss)
		l_trainAcc, = ax2.plot(trainAcc)
		l_testAcc, = ax2.plot(valAcc)
		plt.legend([l_trainLoss, l_testLoss, l_trainAcc, l_testAcc],
				  ['Train_loss', 'Test_loss', 'Train_acc', 'Test_acc'])
		figName = '{}loss_accuracy{}.png'.format(prefix, suffix)

	plt.grid(linewidth=1, linestyle='-.')
	plt.savefig(os.path.join(modelPath, figName), dpi=200, bbox_inches='tight')
	plt.show()
	
def mixup_data(x, y, alpha=1.0):
	'''Returns mixed inputs, pairs of targets, and lambda'''
	if alpha > 0:
		lam = np.random.beta(alpha, alpha)
	else:
		lam = 1
	batch_size = x.size()[0]
	index = torch.randperm(batch_size)

	mixed_x = lam * x + (1 - lam) * x[index, :]
	y_a, y_b = y, y[index]
	
	return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
	return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def get_shuffled_data(x, y):
	index = torch.randperm(x.shape[0])
	x_a, x_b = x, x[index]
	y_a, y_b = y, y[index]
	return x_a, x_b, y_a, y_b

def cal_ssim(img1, img2):
	img1 = img1.cpu().detach().numpy()
	img2 = img2.cpu().detach().numpy()
	k1 = 0.01
	k2 = 0.03
	l = 255
	C1 = (k1 * l) ** 2
	C2 = (k2 * l) ** 2
	C3 = C2 / 2
	mean1 = np.mean(img1, axis=(1, 2, 3))
	mean2 = np.mean(img2, axis=(1, 2, 3))
	var1 = np.var(img1, axis=(1, 2, 3))
	var2 = np.var(img2, axis=(1, 2, 3))
	std1 = np.sqrt(var1)
	std2 = np.sqrt(var2)
	cov = np.mean((img1 - mean1[:, np.newaxis, np.newaxis, np.newaxis]) * 
				  (img2 - mean2[:, np.newaxis, np.newaxis, np.newaxis]), axis=(1, 2, 3))
	L = (2 * mean1 * mean2 + C1) / (mean1 ** 2 + mean2 ** 2 + C1)
	C = (2 * std1 * std2 + C2) / (std1 ** 2 + std2 ** 2 + C2)
	S = (cov + C3) / (std1 * std2 + C3)
	ssim = L * C * S
	return torch.from_numpy(ssim).to(device)

def Euclidean_distance(P, Q):
	P = F.softmax(P.view(P.shape[0], -1), 1) + 1e-9
	Q = F.softmax(Q.view(Q.shape[0], -1), 1) + 1e-9
	return ((P - Q) ** 2).sum(1).sqrt()

def KL_divergence(P, Q):
	P = F.softmax(P.view(P.shape[0], -1), 1) + 1e-9
	Q = F.softmax(Q.view(Q.shape[0], -1), 1) + 1e-9
	return torch.sum(P * P.log() - P * Q.log(), 1)

def JS_divergence(P, Q):
	return 0.5 * KL_divergence(P, (P + Q) / 2) + 0.5 * KL_divergence(Q, (P + Q) / 2)

def EMD(P, Q):
	a = np.ones(shape[0] * shape[1] * shape[2]) / (shape[0] * shape[1] * shape[2])
	b = np.ones(shape[0] * shape[1] * shape[2]) / (shape[0] * shape[1] * shape[2])
	dis = []
	for i in range(len(P)):
		M = ot.dist(P[i].view(-1, 1).cpu().detach().numpy(), Q[i].view(-1, 1).cpu().detach().numpy(), 'euclidean')
		d = ot.emd2(a, b, M)
		dis.append(d)
	return torch.Tensor(dis)

def sinkhorn(P, Q):
	a = np.ones(shape[0] * shape[1] * shape[2]) / (shape[0] * shape[1] * shape[2])
	b = np.ones(shape[0] * shape[1] * shape[2]) / (shape[0] * shape[1] * shape[2])
	dis = []
	for i in range(len(P)):
		M = ot.dist(P[i].view(-1, 1).cpu().detach().numpy(), Q[i].view(-1, 1).cpu().detach().numpy(), 'euclidean')
		d = ot.sinkhorn2(a, b, M, 1, numItermax=100)
		dis.append(d)
	return torch.Tensor(dis)

def norm1(P, Q):
#     P = P.view(P.shape[0], -1)
#     Q = Q.view(Q.shape[0], -1)
	return (P - Q).abs().mean(1).mean(1).mean(1)
#     return torch.norm(P - Q, p=1, dim=[1, 2, 3])


def train_val(optimizer, n_epochs, trainDataset, trainLoader, valDataset, valLoader):
	lossLog = dict({'train': [], 'val': []})
	accLog = dict({'train': [], 'val': []})
	dataSet = {'train': trainDataset, 'val': valDataset}
	dataLoader = {'train': trainLoader, 'val': valLoader}
	dataSize = {x: dataSet[x].__len__() for x in ['train', 'val']}
	batchSize = {'train': train_batch_size, 'val': test_batch_size}
	iterNum = {x: np.ceil(dataSize[x] / batchSize[x]).astype('int32') for x in ['train', 'val']}
	print('dataSize: {}'.format(dataSize))
	logger.info('dataSize: {}'.format(dataSize))
	print('batchSize: {}'.format(batchSize))
	logger.info('batchSize: {}'.format(batchSize))
	print('iterNum: {}'.format(iterNum))
	logger.info('iterNum: {}'.format(iterNum))
	best_acc = 0.0
	start = time.time()
	for epoch in tqdm(range(n_epochs), desc='Epoch'):  # loop over the dataset multiple times
		print('Epoch {}/{} lr = {}'.format(epoch+1, n_epochs, optimizer.param_groups[0]['lr']))
		logger.info('Epoch {}/{} lr = {}'.format(epoch+1, n_epochs, optimizer.param_groups[0]['lr']))
		print('-' * 10)
		logger.info('-' * 10)
		epochStart = time.time()
		for phase in ['train', 'val']:
			if(phase == 'train'):
				net.train()  # Set model to training mode
			else:
				net.eval()   # Set model to evaluate mode
			running_loss = []
			running_corrects = 0
			running_cnt = 0
			for i, data in enumerate(dataLoader[phase], 0):
				# sys.stdout.flush()
				inputs, labels = data
				inputs = inputs.to(device)
				labels = labels.to(device)
				if(method == 'mixup' and phase == 'train'):
					inputs, labels_a, labels_b, lam = mixup_data(inputs, labels)
				elif(method in ['automix', 'adamixup'] and phase == 'train'):
					inputs_a, inputs_b, labels_a, labels_b = get_shuffled_data(inputs, labels)
					
				optimizer.zero_grad()
				with torch.set_grad_enabled(phase == 'train'):
					if(method == 'adamixup' and phase == 'train'):
						midIdx = int(len(inputs_a) / 2)
						inputs_unseen = torch.cat([inputs_a[midIdx:]], 0)
						labels_unseen = torch.cat([labels_a[midIdx:]], 0)
						inputs_a, inputs_b = inputs_a[:midIdx], inputs_b[:midIdx]
						labels_a, labels_b = labels_a[:midIdx], labels_b[:midIdx]
						m1, m2 = torch.randn(inputs_a.shape).to(device), torch.randn(inputs_b.shape).to(device)
						inputs = (m1 * inputs_a + m2 * inputs_b) / 2

						gate = PRG_net(inputs)
						gate = torch.softmax(gate, 1)
						alpha, alpha2, _ = torch.split(gate, [1, 1, 1], 1)
						uni = torch.rand([len(inputs_a), 1]).to(device)
						weight = alpha + uni * alpha2
						
						x_weight = weight.view(len(inputs_a), 1, 1, 1)
						y_weight = weight.view(len(inputs_a), 1)
						x_mix = inputs_a * x_weight + inputs_b * (1 - x_weight)
						y_mix = labels_a * y_weight + labels_b * (1 - y_weight)
						x_all = torch.cat([x_mix, inputs_a, inputs_b], 0)
						y_all = torch.cat([y_mix, labels_a, labels_b], 0)
						inputs, labels = shuffle(x_all, y_all)
					elif(method == 'automix' and phase == 'train'):
						midIdx = int(len(inputs_a) / 2)
						inputs_unseen = torch.cat([inputs_a[midIdx:]], 0)
						labels_unseen = torch.cat([labels_a[midIdx:]], 0)
						inputs_a, inputs_b = inputs_a[:midIdx], inputs_b[:midIdx]
						labels_a, labels_b = labels_a[:midIdx], labels_b[:midIdx]
						
						lam = torch.rand(len(labels_a), 1).to(device)
						y_mix = lam * labels_a + (1 - lam) * labels_b
						x_mix = unet([inputs_a, inputs_b, y_mix])
						
						x_all = torch.cat([x_mix, inputs_a], 0)
						y_all = torch.cat([y_mix, labels_a], 0)
						inputs, labels = shuffle(x_all, y_all)

						if(i in [0, 1]):
							cmap = 'gray'
							print(lam[0].item())
							plt.subplot(131)
							plt.imshow(inputs_a[0].permute(1, 2, 0).squeeze().cpu().detach().numpy(), cmap=cmap)
							plt.subplot(132)
							plt.imshow(inputs_b[0].permute(1, 2, 0).squeeze().cpu().detach().numpy(), cmap=cmap)
							plt.subplot(133)
							plt.title(y_mix[0].cpu())
							plt.imshow(x_mix[0].permute(1, 2, 0).squeeze().cpu().detach().numpy(), cmap=cmap)
							plt.show()
					outputs = net(inputs)
					preds = torch.argmax(outputs, 1)
					if(method in ['mixup'] and phase == 'train'):
						clsLoss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
					else:
						clsLoss = criterion(outputs, labels)
#                         clsLoss = criterion(outputs, torch.argmax(labels, 1))
					if(method == 'automix' and phase == 'train'):
						d1 = norm1(inputs_a, x_mix)
						d2 = norm1(inputs_b, x_mix)
#                         disLoss = lam[:int(lam.shape[0]/3)] * d1 + (1 - lam[:int(lam.shape[0]/3)]) * d2
						disLoss = lam * d1 + (1 - lam) * d2
#                         disLoss = d1 + d2
						loss = clsLoss.mean() + 1.5 * disLoss.mean()
					elif(method == 'adamixup' and phase == 'train'):
						x_pos = torch.cat([inputs_unseen, inputs_a, inputs_b], 0)
						y_pos = torch.zeros(len(x_pos), 2).scatter_(1, torch.ones(len(x_pos), 1).long(), 1).to(device)
						x_neg = torch.cat([x_mix], 0)
						y_neg = torch.zeros(len(x_neg), 2).scatter_(1, torch.zeros(len(x_neg), 1).long(), 1).to(device)
						x_bin = torch.cat([x_pos, x_neg], 0)
						y_bin = torch.cat([y_pos, y_neg], 0)
						x_bin, y_bin = shuffle(x_bin, y_bin)
						linear = nn.Linear(num_classes, 2)
						extra = net(x_bin)
						logits = net.linear2(extra)
						disLoss = criterion(logits, y_bin)
						
						loss = clsLoss.mean() + disLoss.mean()
					else:
						loss = clsLoss.mean()
					if(phase == 'train'):
						loss.backward()
						optimizer.step()

				running_loss.append(loss.cpu().item())
				if(method in ['mixup'] and phase == 'train'):
					num_correct = (lam.cpu().item() * preds.eq(torch.argmax(labels_a, 1)).cpu().sum().float().item() + (1 - lam.cpu().item()) * preds.eq(torch.argmax(labels_b, 1)).cpu().sum().float().item())
				else:
					num_correct = preds.eq(torch.argmax(labels, 1)).cpu().sum().float().item()
				running_corrects += num_correct
				running_cnt += inputs.shape[0]

				sys.stdout.write('                                                                                                 \r')
				sys.stdout.flush()
				sys.stdout.write('Iter: {} / {} ({:.0f}s)\tLoss= {:.4f}\tAcc= {:.2f}% ({}/{})\r'
					 .format(i+1, iterNum[phase], time.time() - epochStart, 
							 running_loss[-1], 100 * num_correct / len(inputs), int(running_corrects), running_cnt))
				sys.stdout.flush()
			if(phase == 'train'):
				exp_lr_scheduler.step()
			epoch_loss = np.mean(running_loss)
			epoch_acc = running_corrects / running_cnt
			accLog[phase].append(epoch_acc)
			lossLog[phase].append(epoch_loss)
			epochDuration = time.time() - epochStart
			epochStart = time.time()
			sys.stdout.write('                                                                                                  \r')
			sys.stdout.flush()
			print('[ {} ] Loss: {:.4f} Acc: {:.2f}% ({:.0f}mins {:.2f}s)'.format(phase, epoch_loss, 100 * epoch_acc, epochDuration // 60, epochDuration % 60))
			logger.info('[ {} ] Loss: {:.4f} Acc: {:.2f}% ({:.0f}mins {:.2f}s)'.format(phase, epoch_loss, 100 * epoch_acc, epochDuration // 60, epochDuration % 60))

			if(phase == 'val' and epoch_acc > best_acc):
				print('Saving best model to {}'.format(os.path.join(modelPath, modelName)))
				logger.info('Saving best model to {}'.format(os.path.join(modelPath, modelName)))
				if(method == 'automix'):
					state = {'net': [net, unet], 'opt': optimizer, 'acc': epoch_acc, 'epoch': epoch}
				elif(method == 'adamixup'):
					state = {'net': [net, PRG_net], 'opt': optimizer, 'acc': epoch_acc, 'epoch': epoch}
				else:
					state = {'net': net, 'opt': optimizer, 'acc': epoch_acc, 'epoch': epoch}
				if not os.path.isdir(modelPath):
					os.makedirs(modelPath)
				torch.save(state, os.path.join(modelPath, modelName))
				best_acc = epoch_acc
			if(phase == 'val' and epoch == n_epochs - 1):
				finalModelName = 'final-{}'.format(modelName)
				print('Saving final model to {}'.format(os.path.join(modelPath, finalModelName)))
				logger.info('Saving final model to {}'.format(os.path.join(modelPath, finalModelName)))
				if(method == 'automix'):
					state = {'net': [net, unet], 'opt': optimizer, 'acc': epoch_acc, 'epoch': epoch}
				elif(method == 'adamixup'):
					state = {'net': [net, PRG_net], 'opt': optimizer, 'acc': epoch_acc, 'epoch': epoch}
				else:
					state = {'net': net, 'opt': optimizer, 'acc': epoch_acc, 'epoch': epoch}
				if not os.path.isdir(modelPath):
					os.makedirs(modelPath)
				torch.save(state, os.path.join(modelPath, finalModelName))
		print()
		logger.info()
	duration = time.time() - start
	print('Training complete in {:.0f}h {:.0f}m {:.2f}s'.format(duration // 3600, (duration % 3600) // 60, duration % 60))
	logger.info('Training complete in {:.0f}h {:.0f}m {:.2f}s'.format(duration // 3600, (duration % 3600) // 60, duration % 60))
	print('Best val Acc: {:4f}'.format(best_acc))
	logger.info('Best val Acc: {:4f}'.format(best_acc))

	log = dict({'acc': accLog, 'loss': lossLog})
	with open(os.path.join(modelPath, '{}-log-{}.pkl'.format(method, fold+1)), 'wb') as f:
		pickle.dump(log, f)

	return best_acc, log

def get_model(netType, methodType, num_classes, shape):
	if(netType == 'mynet'):
		net = MyNet(input_shape=shape, 
					num_classes=num_classes)
	elif(netType == 'resnet18'):
		net = ResNet18(input_shape=shape, 
					   num_classes=num_classes)
	if(methodType == 'adamixup'):
		net.linear2 = nn.Linear(num_classes, 2)    
	net = net.to(device)
	outputNet = [net]
	parameters = [{'params': net.parameters()}]
#     summary(model=net, input_size=shape)

	if(methodType == 'automix'):
		unet = UNet(input_shape=(shape[0], shape[1], shape[2]), output_shape=shape, num_classes=num_classes)
		unet = unet.to(device)
		outputNet.append(unet)
		parameters.append({'params': unet.parameters()})
	if(methodType == 'adamixup'):
		PRG_net = ResNet18(num_classes=3)
		PRG_net = PRG_net.to(device)
		outputNet.append(PRG_net)
		parameters.append({'params': PRG_net.parameters()})
#         summary(model=PRG_net, input_size=shape)

	if(device.type == 'cuda'):
		cudnn.benchmark = True
		if(args.parallel):
			print('DataParallel : {}'.format(args.parallel))
			logger.info('DataParallel : {}'.format(args.parallel))
			for i in range(len(outputNet)):
				outputNet[i] = torch.nn.DataParallel(outputNet[i])

	return outputNet, parameters

def get_logging(log_path):
	logging.basicConfig(
		level=logging.DEBUG,
		format='%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s\t%(message)s',
		datefmt='%Y-%m-%d %H:%M:%S',
		filename=log_path,
		filemode='w')
	# console = logging.StreamHandler()
	logger = logging.getLogger(__name__)

	return logger

if(__name__ == '__main__'):
	# torch.autograd.set_detect_anomaly(True)

	args = parse_args()

	method = str(args.method).lower()
	arch = str(args.arch).lower()
	dataset = str(args.dataset).upper()
	train_batch_size = int(args.batch_size)
	test_batch_size = 10
	num_workers = int(args.num_workers)
	learning_rate = float(args.lr)
	momentum = float(args.momentum)
	weight_decay = float(args.weight_decay)
	log_path = str(args.log_path)
	if(not args.data_dir):
		dataPathDict = {
				'IMAGENET'		: '/media/reborn/Others2/ImageNet',
				'CIFAR10'		: 'Dataset/cifar10',
				'CIFAR100'		: 'Dataset/cifar100',
				'MNIST'			: 'Dataset/mnist',
				'FASHION-MNIST'	: 'Dataset/fashion-mnist',
				'GTSRB'			: 'Dataset/GTSRB',
				}
		args.data_dir = dataPathDict[dataset]
	data_dir = str(args.data_dir)
	logger = get_logging(log_path)

	print(args)
	logger.info(args)
	
	if(len(args.gpu) > 0):
		os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
		print('CUDA_VISIBLE_DEVICES : {}'.format(os.environ["CUDA_VISIBLE_DEVICES"]))
		logger.info('CUDA_VISIBLE_DEVICES : {}'.format(os.environ["CUDA_VISIBLE_DEVICES"]))
	print('torch.cuda.is_available : {}'.format(torch.cuda.is_available()))
	logger.info('torch.cuda.is_available : {}'.format(torch.cuda.is_available()))
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	# netList = ['mynet', 'resnet18']
	# netIndex = 1
	# dataName = ['IMAGENET', 'CIFAR10', 'CIFAR100', 'MNIST', 'FASHION-MNIST', 'GTSRB', 'MIML']
	# dataIndex = 0
	# methodList = ['baseline', 'bc', 'mixup', 'automix', 'adamixup']
	# methodIndex = 0
	# print(arch, dataset, method)

	trainDataset, trainLoader, testDataset, testLoader, classes, num_classes, shape, lrDecayStep, n_epochs = get_dataset(dataset, method, data_dir)
	print('Dataset [{}] loaded!'.format(dataset))
	logger.info('Dataset [{}] loaded!'.format(dataset))

	modelPath = 'pytorch_model_learnt/{}/{}/{}'.format(dataset, arch, method)
	for fold in tqdm(range(1), desc='Fold'):
		modelName = '{}-{}.ckpt'.format(method, fold+1)
		nets, parameters = get_model(arch, method, num_classes, shape)
		if(method == 'automix'):
			net, unet = nets
		elif(method == 'adamixup'):
			net, PRG_net = nets
		else:
			net = nets[0]

		criterion = myLoss(method).to(device)
	#     criterion = nn.CrossEntropyLoss().to(device)
		optimizer = optim.SGD(parameters, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
	#     optimizer = optim.Adam(parameters, lr=0.0002, betas=(0.5, 0.999))
		exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, lrDecayStep, gamma=0.1)

		best_acc, log = train_val(optimizer, 
								  n_epochs, 
								  trainDataset, 
								  trainLoader, 
								  testDataset, 
								  testLoader)

		plot_acc_loss(log, 'both', '{}-'.format(method), '-{}'.format(fold+1))
		plot_acc_loss(log, 'loss', '{}-'.format(method), '-{}'.format(fold+1))
		plot_acc_loss(log, 'accuracy', '{}-'.format(method), '-{}'.format(fold+1))
		if(method == 'automix'):
			net, unet = torch.load(os.path.join(modelPath, modelName))['net']
		elif(method == 'adamixup'):
			net, PRG_net = torch.load(os.path.join(modelPath, modelName))['net']
		else:
			net = torch.load(os.path.join(modelPath, modelName))['net']
		accTotal = eval_total(net, testLoader)
		accPerClass = eval_per_class(net, testLoader, classes)
