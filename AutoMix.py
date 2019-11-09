import _init_paths
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import argparse
import pickle
import time
import sys
import os
# import ot
import cv2
import datetime

from summary import summary
from distance import *
from utils import *
from load_data import *
from pytorch_models import *

import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets, models
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.optim import lr_scheduler


# python AutoMix.py \
# --method=baseline \
# --arch=resnet18  \
# --dataset=IMAGENET \
# --data_dir=/media/reborn/Others2/ImageNet \
# --batch_size=32 \
# --lr=0.01 \
# --gpu=0 \
# --num_workers=8 \
# --parallel=True \
# --log_path=./automix.log

def parse_args():
	"""Parse input arguments."""
	parser = argparse.ArgumentParser(description='AutoMix: Mixup Networks for Sample Interpolation via Cooperative Barycenter Learning')
	parser.add_argument('--method', dest='method', default='baseline', type=str, choices=['baseline', 'bc', 'mixup', 'automix', 'adamixup', 'manifoldmixup'], help='Method : [baseline, bc, mixup, automix, adamixup]')
	parser.add_argument('--arch', dest='arch', default='resnet18', type=str, choices=['mynet', 'resnet18'], help='Backbone architecture : [mynet, resnet18]')
	parser.add_argument('--dataset', dest='dataset', default='IMAGENET', type=str, choices=['IMAGENET', 'CIFAR10', 'CIFAR100', 'MNIST', 'FASHION-MNIST', 'GTSRB', 'MIML'], help='Dataset to be trained : [IMAGENET, CIFAR10, CIFAR100, MNIST, FASHION-MNIST, GTSRB, MIML]')
	parser.add_argument('--data_dir', dest='data_dir', default=None, type=str, help='Path to the dataset')
	parser.add_argument('--epoch', dest='epoch', default=None, type=int, help='Training epochs')
	parser.add_argument('--kfold', dest='kfold', default=1, type=int, help='K-fold cross validation')
	parser.add_argument('--criterion', dest='criterion', default='myloss', type=str, choices=['bceloss', 'myloss', 'celoss', 'mseloss'], help='Loss criterion')
	parser.add_argument('--batch_size', dest='batch_size', default=25, type=int, help='Batch_size for training')
	parser.add_argument('--gpu', dest='gpu', default='', type=str, help='GPU lists can be used')
	parser.add_argument('--lr', dest='lr', default=0.05, type=float, help='Learning rate')
	parser.add_argument('--lr_schedule', dest='lr_schedule', default=None, type=str, help='Learning rate decay schedule')
	parser.add_argument('--num_workers', dest='num_workers', default=8, type=int, help='Num of multiple threads')
	parser.add_argument('--momentum', dest='momentum', default=0.9, type=float, help='Momentum for optimizer')
	parser.add_argument('--weight_decay', dest='weight_decay', default=5e-4, type=float, help='Weight_decay for optimizer')
	parser.add_argument('--parallel', dest='parallel', default=False, type=bool, help='Train parallelly with multi-GPUs?')
	parser.add_argument('--log_path', dest='log_path', default=None, type=str, help='Path to the dataset')

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
	print_log('Accuracy of the network on the {} test images: {:.2f}% ({:.0f}mins {:.2f}s)'.format(total, accTotal, duration // 60, duration % 60), logger, 'info')
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
	print_log('class_correct\t:\t{}'.format(class_correct), logger, 'info')
	print_log('class_total\t:\t{}'.format(class_total), logger, 'info')
	accPerClass = dict()
	for i in range(num_classes):
		accPerClass[classes[i]] = 0 if class_correct[i] == 0 else '{:.2f}%'.format(100 * class_correct[i] / class_total[i])
	duration = time.time() - start
	print_log('Per class accuracy :', logger, 'info')
	print_log(accPerClass, logger, 'info')
	print_log('Duration for accPerClass : {:.0f}mins {:.2f}s'.format(duration // 60, duration % 60), logger, 'info')
	return accPerClass
	
def train_val(optimizer, n_epochs, trainDataset, trainLoader, valDataset, valLoader):
	lossLog = dict({'train': [], 'val': []})
	accLog = dict({'train': [], 'val': []})
	dataSet = {'train': trainDataset, 'val': valDataset}
	dataLoader = {'train': trainLoader, 'val': valLoader}
	dataSize = {x: dataSet[x].__len__() for x in ['train', 'val']}
	batchSize = {'train': trainBS, 'val': testBS}
	iterNum = {x: np.ceil(dataSize[x] / batchSize[x]).astype('int32') for x in ['train', 'val']}
	print_log('dataSize: {}'.format(dataSize), logger, 'info')
	print_log('batchSize: {}'.format(batchSize), logger, 'info')
	print_log('iterNum: {}'.format(iterNum), logger, 'info')
	best_acc = 0.0
	start = time.time()
	for epoch in tqdm(range(n_epochs), desc='Epoch'):  # loop over the dataset multiple times
		print_log('Epoch {}/{} lr = {}'.format(epoch+1, n_epochs, optimizer.param_groups[0]['lr']), logger, 'info')
		print_log('-' * 10, logger, 'info')
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
				lam = None
				if(method == 'mixup' and phase == 'train'):
					inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, 1.0)
				elif(method == 'manifoldmixup' and phase == 'train'):
					inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, 2.0)
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
						# if(i in [0, 1]):
						# 	cmap = 'gray'
						# 	print(lam[0].item())
						# 	plt.subplot(131)
						# 	plt.imshow(inputs_a[0].permute(1, 2, 0).squeeze().cpu().detach().numpy(), cmap=cmap)
						# 	plt.subplot(132)
						# 	plt.imshow(inputs_b[0].permute(1, 2, 0).squeeze().cpu().detach().numpy(), cmap=cmap)
						# 	plt.subplot(133)
						# 	plt.title(y_mix[0].cpu())
						# 	plt.imshow(x_mix[0].permute(1, 2, 0).squeeze().cpu().detach().numpy(), cmap=cmap)
						# 	plt.show()
					elif(method == 'manifoldmixup' and phase == 'train'):
						y_mix = lam * labels_a + (1 - lam) * labels_b
						inputs, labels = shuffle(inputs, y_mix)

					if(phase == 'train'):
						outputs = net(inputs, manifoldMixup=(method=='manifoldmixup'), lam=lam)
					else:
						outputs = net(inputs)
					preds = torch.argmax(outputs, 1)

					if(method in ['mixup'] and phase == 'train'):
						clsLoss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
					else:
						if(criterionType == 'bceloss'):
							clsLoss = criterion(softmax(outputs), labels)
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
					num_correct = (lam * preds.eq(torch.argmax(labels_a, 1)).cpu().sum().float().item() + (1 - lam) * preds.eq(torch.argmax(labels_b, 1)).cpu().sum().float().item())
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
			print_log('[ {} ] Loss: {:.4f} Acc: {:.2f}% ({}/{}) ({:.0f}mins {:.2f}s)'.format(phase, epoch_loss, 100 * epoch_acc, int(running_corrects), running_cnt, epochDuration // 60, epochDuration % 60), logger, 'info')

			if(phase == 'val' and epoch_acc > best_acc):
				print_log('Saving best model to {}'.format(os.path.join(modelPath, modelName)), logger, 'info')
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
				print_log('Saving final model to {}'.format(os.path.join(modelPath, finalModelName)), logger, 'info')
				if(method == 'automix'):
					state = {'net': [net, unet], 'opt': optimizer, 'acc': epoch_acc, 'epoch': epoch}
				elif(method == 'adamixup'):
					state = {'net': [net, PRG_net], 'opt': optimizer, 'acc': epoch_acc, 'epoch': epoch}
				else:
					state = {'net': net, 'opt': optimizer, 'acc': epoch_acc, 'epoch': epoch}
				if not os.path.isdir(modelPath):
					os.makedirs(modelPath)
				torch.save(state, os.path.join(modelPath, finalModelName))
		print_log('', logger, 'info')
	duration = time.time() - start
	print_log('Training complete in {:.0f}h {:.0f}m {:.2f}s'.format(duration // 3600, (duration % 3600) // 60, duration % 60), logger, 'info')
	print_log('Best val Acc: {:4f}'.format(best_acc), logger, 'info')

	log = dict({'acc': accLog, 'loss': lossLog})
	with open(os.path.join(modelPath, '{}-log-{}.pkl'.format(expName, fold+1)), 'wb') as f:
		pickle.dump(log, f)
		print_log("Training logs saved to : {}".format(os.path.join(modelPath, '{}-log-{}.pkl'.format(expName, fold+1))), logger, 'info')

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
	print_log('Backbone : [{}]\n{}'.format(netType, summary(model=net, input_size=shape)), logger, 'info')
	
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
			print_log('DataParallel : {}'.format(args.parallel), logger, 'info')
			for i in range(len(outputNet)):
				outputNet[i] = torch.nn.DataParallel(outputNet[i])

	return outputNet, parameters

if(__name__ == '__main__'):
	# torch.autograd.set_detect_anomaly(True)
	defaultSetting = {
				'IMAGENET'		: ['/media/reborn/Others2/ImageNet', 90, [50, 75]],
				'CIFAR10'		: ['Dataset/cifar10', 300, [150, 225]],
				'CIFAR100'		: ['Dataset/cifar100', 300, [150, 225]],
				'MNIST'			: ['Dataset/mnist', 100, [50, 75]],
				'FASHION-MNIST'	: ['Dataset/fashion-mnist', 100, [50, 75]],
				'GTSRB'			: ['Dataset/GTSRB', 100, [50, 75]],
				}

	args = parse_args()

	dataset = str(args.dataset).upper()
	method = str(args.method).lower()
	arch = str(args.arch).lower()
	criterionType = str(args.criterion)
	trainBS = int(args.batch_size)
	testBS = 10
	numWorkers = int(args.num_workers)
	learning_rate = float(args.lr)
	args.lr_schedule = list(map(int, args.lr_schedule.split(','))) if args.lr_schedule else defaultSetting[dataset][2]
	lrDecayStep = args.lr_schedule
	momentum = float(args.momentum)
	weight_decay = float(args.weight_decay)
	args.data_dir = str(args.data_dir) if args.data_dir else defaultSetting[dataset][0]
	dataDir = str(args.data_dir)
	args.epoch = int(args.epoch) if args.epoch else defaultSetting[dataset][1]
	epochs = int(args.epoch)
	kfold = int(args.kfold)
	expName = get_exp_name(dataset=dataset,
							arch=arch,
							epochs=epochs,
							batch_size=trainBS,
							lr=learning_rate,
							momentum=momentum,
							decay=weight_decay,
							method=method,
							criterion=criterionType)
	args.log_path = str(args.log_path) if args.log_path else os.path.join(os.getcwd(),'{}|{}.log'.format(expName, datetime.datetime.now().strftime('%Y-%m-%d|%H:%M:%S')))
	logPath = str(args.log_path)
	logger = get_logging(logPath)

	print_log("Arguments :\n{}".format(args), logger, 'info')
	print_log("Experiment settings :\n{}".format(expName), logger, 'info')
	print_log("python version : {}".format(sys.version.replace('\n', ' ')), logger, 'info')
	print_log("torch  version : {}".format(torch.__version__), logger, 'info')
	print_log("cudnn  version : {}".format(torch.backends.cudnn.version()), logger, 'info')
	if(len(args.gpu) > 0):
		os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
		print_log('CUDA_VISIBLE_DEVICES : {}'.format(os.environ["CUDA_VISIBLE_DEVICES"]), logger, 'info')
	print_log('torch.cuda.is_available : {}'.format(torch.cuda.is_available()), logger, 'info')
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	trainDataset, trainLoader, testDataset, testLoader, classes, num_classes, shape = get_dataset(dataset, method, dataDir, trainBS, testBS, numWorkers)
	if(epochs):
		n_epochs = epochs
	print_log('Dataset [{}] loaded!'.format(dataset), logger, 'info')

	modelPath = 'pytorch_model_learnt/{}/{}/{}'.format(dataset, arch, method)
	for fold in tqdm(range(kfold), desc='Fold'):
		modelName = '{}-{}.ckpt'.format(expName, fold+1)
		nets, parameters = get_model(arch, method, num_classes, shape)
		if(method == 'automix'):
			net, unet = nets
		elif(method == 'adamixup'):
			net, PRG_net = nets
		else:
			net = nets[0]

		softmax = nn.Softmax(dim=1).to(device)
		if(criterionType == 'bceloss'):
			criterion = nn.BCELoss().to(device)
		elif(criterionType == 'myloss'):
			criterion = myLoss(method).to(device)
		elif(criterionType == 'mseloss'):
			criterion = nn.MSELoss().to(device)
		elif(criterionType == 'celoss'):
			criterion = nn.CrossEntropyLoss().to(device)

		optimizer = optim.SGD(parameters, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
	#     optimizer = optim.Adam(parameters, lr=0.0002, betas=(0.5, 0.999))
		exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, lrDecayStep, gamma=0.1)

		best_acc, log = train_val(optimizer, 
								  n_epochs, 
								  trainDataset, 
								  trainLoader, 
								  testDataset, 
								  testLoader)

		plot_acc_loss(log, 'both', modelPath, logger, '{}-'.format(expName), '-{}'.format(fold+1))
		plot_acc_loss(log, 'loss', modelPath, logger, '{}-'.format(expName), '-{}'.format(fold+1))
		plot_acc_loss(log, 'accuracy', modelPath, logger, '{}-'.format(expName), '-{}'.format(fold+1))
		if(method == 'automix'):
			net, unet = torch.load(os.path.join(modelPath, modelName))['net']
		elif(method == 'adamixup'):
			net, PRG_net = torch.load(os.path.join(modelPath, modelName))['net']
		else:
			net = torch.load(os.path.join(modelPath, modelName))['net']
		accTotal = eval_total(net, testLoader)
		accPerClass = eval_per_class(net, testLoader, classes)
