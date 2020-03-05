import _init_paths
import torch
from torch.utils.data import DataLoader
import numpy as np
import logging
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import interp
from scipy.io import loadmat

class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.sum += val
		self.count += n
		self.avg = self.sum / self.count

def get_exp_name(dataset='cifar10',
					arch='',
					epochs=400,
					batch_size=64,
					lr=0.01,
					momentum=0.5,
					decay=0.0005,
					method='baseline',
					criterion='bceloss',
					sampleNum=None):
	splitChar = '|'
	exp_name = 'dataset_' + str(dataset)
	if(sampleNum):
		exp_name += '({}-pc)'.format(sampleNum)
	exp_name += splitChar + 'arch_' + str(arch)
	exp_name += splitChar + 'method_' + str(method)
	exp_name += splitChar + 'epoch_' + str(epochs)
	exp_name += splitChar + 'batchsize_' + str(batch_size)
	exp_name += splitChar + 'lr_' + str(lr)
	exp_name += splitChar + 'momentum_' + str(momentum)
	exp_name += splitChar + 'decay_' + str(decay)
	exp_name += splitChar + 'criterion_' + str(criterion)

	return exp_name

def get_logging(log_path):
	logging.basicConfig(
		level=logging.DEBUG,
		format='%(asctime)s %(filename)s %(levelname)s\t%(message)s',
		datefmt='%Y-%m-%d %H:%M:%S',
		filename=log_path,
		filemode='w')
	# console = logging.StreamHandler()
	logger = logging.getLogger(__name__)

	return logger

def print_log(print_string, logger, log_type):
	print("{}".format(print_string))
	if(log_type == 'info'):
		logger.info("{}".format(print_string))

def convert_secs2time(epoch_time):
	need_hour = int(epoch_time / 3600)
	need_mins = int((epoch_time - 3600*need_hour) / 60)
	need_secs = int(epoch_time - 3600*need_hour - 60*need_mins)
	return need_hour, need_mins, need_secs

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

def distribution(labels, type='normal'):
	result = {}
	if(type == 'onehot'):
		ll = np.argmax(labels, axis=1)
	else:
		ll = labels
	for i in set(ll):
		result[i] = ll.tolist().count(i)
	print_log("Label distribution: {}".format(result), logger, 'info')

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

def ROC_plot(sorted_ROC, dataset):
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
	# plt.show()
	plt.close()

def plot_acc_loss(log, type, modelPath, logger, prefix='', suffix='', printFlag=False):
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
		l_trainAcc, = ax2.plot(trainAcc, marker='x')
		l_testAcc, = ax2.plot(valAcc, marker='x')
		plt.legend([l_trainLoss, l_testLoss, l_trainAcc, l_testAcc],
				  ['Train_loss', 'Test_loss', 'Train_acc', 'Test_acc'])
		figName = '{}loss_accuracy{}.png'.format(prefix, suffix)

	plt.grid(linewidth=1, linestyle='-.')
	plt.savefig(os.path.join(modelPath, figName), dpi=200, bbox_inches='tight')
	if(printFlag):
		print_log("Figure saved to : {}".format(os.path.join(modelPath, figName)), logger, 'info')
	# plt.show()
	plt.close()

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

def mAP_evaluation(PR_results, PR_labels, numClasses):
	classPR = []
	for i in range(numClasses):
		classPR.append(np.concatenate([PR_results[:, i][:, np.newaxis], PR_labels[:, i][:, np.newaxis]], 1))
	classPR = np.array(classPR)
	APList = []
	for i in range(len(classPR)):
	# for i in range(2):
		sortIdx = np.argsort(-classPR[i][:, 0])
		sortClassPR = classPR[i][sortIdx]
		P, R = [], []
		P_unique = []
		for n in range(1, len(sortClassPR) + 1):
			TP = np.ceil(np.sum(sortClassPR[:n-1, 1]))
			precision = TP / n
			recall = TP / np.sum(sortClassPR[:, 1])
			if(n > 1 and recall == R[n - 2]):
				P.append(P[n - 2])
			else:
				P.append(precision)
				P_unique.append(precision)
			R.append(recall)
		P = np.array(P)
		R = np.array(R)
		AP = np.mean(P_unique)
		APList.append(AP)
	#     plot_PR(P, R, title = "P-R for class@ {} (AP = {:.4f})".format(i, AP))
	mAP = np.mean(APList)
	return mAP

def min_max_norm(tensor):
	mmin = torch.min(tensor)
	mmax = torch.max(tensor)
	tensor = (tensor - mmin) / (mmax - mmin)
	return tensor

def mean_std_norm(tensor, mean, std):
	tensor = (tensor - mean / std)
	return tensor