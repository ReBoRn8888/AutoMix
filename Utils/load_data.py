from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import torch
import cv2
import os
import pickle
import numpy as np
from PIL import Image
import glob

class myDataset(Dataset):
	def __init__(self, images, labels, classes=None, transform=None, mtype='baseline', dtype='MNIST', onehot=True):
		self.images = images
		self.labels = labels
		self.transform = transform
		self.classes = classes
		self.num_classes = len(self.classes)
		self.mtype = mtype.lower()
		self.dtype = dtype.upper()
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
			if(self.dtype in ['IMAGENET', 'TINY-IMAGENET']):
				image1 = cv2.imread(image1[0])
				image2 = cv2.imread(image2[0])
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
			if(self.dtype in ['IMAGENET', 'TINY-IMAGENET']):
				image = cv2.imread(image[0])
			if(self.onehot):
				label = self.to_onehot(label)
			if(self.transform):
				image = self.transform(Image.fromarray(np.uint8(image)))
			return image, label

def zero_mean(tensor, mean, std):
	if not transforms.functional._is_tensor_image(tensor):
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

def cal_mean_std(images):
	rgbMean = []
	rgbStd = []
	for i in range(len(images)):
		if(len(images[i].shape) == 2):
			M, S = np.mean(images[i]), np.std(images[i])
			rgbMean.append([M])
			rgbStd.append([S])
		else:
			rM, rS = np.mean(images[i][:, :, 0]), np.std(images[i][:, :, 0])
			gM, gS = np.mean(images[i][:, :, 1]), np.std(images[i][:, :, 1])
			bM, bS = np.mean(images[i][:, :, 2]), np.std(images[i][:, :, 2])
			rgbMean.append([rM, gM, bM])
			rgbStd.append([rS, gS, bS])
	rgbMean = np.array(rgbMean)
	rgbStd = np.array(rgbStd)
	return np.round(np.mean(rgbMean, 0), 4), np.round(np.mean(rgbStd, 0), 4)

def get_dataset(dataType, methodType, dataPath, trainBS, testBS, numWorkers, sampleNum=None):
	if(dataType == 'IMAGENET'):
		shape = (3, 224, 224)

		# Data loading code
		if(methodType == 'bc'):
			normalize = ZeroMean
		else:
			normalize = transforms.Normalize
		transform_train = transforms.Compose([
			# transforms.RandomResizedCrop(224),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		])

		transform_test = transforms.Compose([
			# transforms.RandomResizedCrop(224),
			# transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		])

		traindir = os.path.join(dataPath, 'train')
		valdir = os.path.join(dataPath, 'validation')

		oriTrainDataset = datasets.ImageFolder(traindir)
		classes = list(oriTrainDataset.class_to_idx.keys())
		# clsId2clsName = get_imageNet_classId2Name()
		num_classes = len(classes)
		images = oriTrainDataset.imgs
		labels = torch.Tensor(oriTrainDataset.targets).long()
		trainDataset = myDataset(images, labels, classes, transform_train, mtype=methodType, dtype=dataType)
		trainLoader = DataLoader(trainDataset, batch_size=trainBS, shuffle=True, num_workers=numWorkers, pin_memory=True)

		oriTestDataset = datasets.ImageFolder(valdir)
		testImages = oriTestDataset.imgs
		testLabels = torch.Tensor(oriTestDataset.targets).long()
		testDataset = myDataset(testImages, testLabels, classes, transform_test, mtype=methodType, dtype=dataType)
		testLoader = DataLoader(testDataset, batch_size=testBS, shuffle=False, num_workers=numWorkers, pin_memory=True)
	elif(dataType == 'TINY-IMAGENET'):
		shape = (3, 64, 64)
		# shape = (3, 224, 224)

		# Data loading code
		if(methodType == 'bc'):
			normalize = ZeroMean
		else:
			normalize = transforms.Normalize
		transform_train = transforms.Compose([
			# transforms.RandomResizedCrop(224),
			# transforms.RandomResizedCrop(64),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			normalize(mean=[0.3975, 0.4481, 0.4802], std=[0.2255, 0.2262, 0.2295]),
		])

		transform_test = transforms.Compose([
			# transforms.RandomResizedCrop(224),
			# transforms.RandomResizedCrop(64),
			# transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			normalize(mean=[0.3975, 0.4481, 0.4802], std=[0.2255, 0.2262, 0.2295]),
		])

		traindir = os.path.join(dataPath, 'train')
		valdir = os.path.join(dataPath, 'validation')

		oriTrainDataset = datasets.ImageFolder(traindir)
		classes = list(oriTrainDataset.class_to_idx.keys())
		# clsId2clsName = get_imageNet_classId2Name()
		num_classes = len(classes)
		images = oriTrainDataset.imgs
		labels = torch.Tensor(oriTrainDataset.targets).long()
		trainDataset = myDataset(images, labels, classes, transform_train, mtype=methodType, dtype=dataType)
		trainLoader = DataLoader(trainDataset, batch_size=trainBS, shuffle=True, num_workers=numWorkers, pin_memory=True)

		oriTestDataset = datasets.ImageFolder(valdir)
		testImages = oriTestDataset.imgs
		testLabels = torch.Tensor(oriTestDataset.targets).long()
		testDataset = myDataset(testImages, testLabels, classes, transform_test, mtype=methodType, dtype=dataType)
		testLoader = DataLoader(testDataset, batch_size=testBS, shuffle=False, num_workers=numWorkers, pin_memory=True)
	elif(dataType == 'CIFAR10'):
		classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
		num_classes = len(classes)
		shape = (3, 32, 32)

		if(methodType == 'bc'):
			normalize = ZeroMean
		else:
			normalize = transforms.Normalize
		transform_train = transforms.Compose([
			transforms.RandomHorizontalFlip(),
			transforms.RandomCrop(32, padding=2),
			transforms.ToTensor(),
			normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
		])

		transform_test = transforms.Compose([
			transforms.ToTensor(),
			normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
		])

		# Load training data
		oriTrainDataset = datasets.CIFAR10(root=dataPath, train=True, download=True, transform=transform_train)
		images = oriTrainDataset.data
		labels = torch.Tensor(oriTrainDataset.targets).long()

		# Sample if needed
		if(sampleNum):
			classSet = list(set(labels.numpy()))
			idxOutput = []
			for cls in classSet:
				idx = np.where(labels == cls)[0][:sampleNum]
				idxOutput.extend(idx)
			images = images[idxOutput]
			labels = labels[idxOutput]

		# Load testing data
		oriTestDataset = datasets.CIFAR10(root=dataPath, train=False, download=True, transform=transform_test)
		testImages = oriTestDataset.data
		testLabels = torch.Tensor(oriTestDataset.targets).long()

		# Create Dataset and DataLoader
		trainDataset = myDataset(images, labels, classes, transform_train, mtype=methodType)
		trainLoader = DataLoader(trainDataset, batch_size=trainBS, shuffle=True, num_workers=numWorkers)
		testDataset = myDataset(testImages, testLabels, classes, transform_test)
		testLoader = DataLoader(testDataset, batch_size=testBS, shuffle=False, num_workers=numWorkers)
	if(dataType == 'CIFAR100'):
		classes = ['{}'.format(i) for i in range(100)]
		num_classes = len(classes)
		shape = (3, 32, 32)

		if(methodType == 'bc'):
			normalize = ZeroMean
		else:
			normalize = transforms.Normalize
		transform_train = transforms.Compose([
			transforms.RandomHorizontalFlip(),
			transforms.RandomCrop(32, padding=2),
			transforms.ToTensor(),
			normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
		])

		transform_test = transforms.Compose([
			transforms.ToTensor(),
			normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
		])

		# Load training data
		oriTrainDataset = datasets.CIFAR100(root=dataPath, train=True, download=True, transform=transform_train)
		images = oriTrainDataset.data
		labels = torch.Tensor(oriTrainDataset.targets).long()

		# Sample if needed
		if(sampleNum):
			classSet = list(set(labels.numpy()))
			idxOutput = []
			for cls in classSet:
				idx = np.where(labels == cls)[0][:sampleNum]
				idxOutput.extend(idx)
			images = images[idxOutput]
			labels = labels[idxOutput]

		# Load testing data
		oriTestDataset = datasets.CIFAR100(root=dataPath, train=False, download=True, transform=transform_test)
		testImages = oriTestDataset.data
		testLabels = torch.Tensor(oriTestDataset.targets).long()

		# Create Dataset and DataLoader
		trainDataset = myDataset(images, labels, classes, transform_train, mtype=methodType)
		trainLoader = DataLoader(trainDataset, batch_size=trainBS, shuffle=True, num_workers=numWorkers)
		testDataset = myDataset(testImages, testLabels, classes, transform_test)
		testLoader = DataLoader(testDataset, batch_size=testBS, shuffle=False, num_workers=numWorkers)
	elif(dataType == 'GTSRB'):
		classes = ['{}'.format(i) for i in range(43)]
		num_classes = len(classes)
		shape = (3, 28, 28)

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

		# Load training data
		with open('{}/39209-all/images.pkl'.format(dataPath), 'rb') as f:
			images = torch.from_numpy(pickle.load(f)).float()
		with open('{}/39209-all/labels.pkl'.format(dataPath), 'rb') as f:
			labels = torch.from_numpy(pickle.load(f))
			labels = torch.argmax(labels, 1)

		# Load testing data
		with open('{}/39209-all/testImages.pkl'.format(dataPath), 'rb') as f:
			testImages = torch.from_numpy(pickle.load(f)).float()
		with open('{}/39209-all/testLabels.pkl'.format(dataPath), 'rb') as f:
			testLabels = torch.from_numpy(pickle.load(f))
			testLabels = torch.argmax(testLabels, 1)

		# Sample if needed
		if(sampleNum):
			classSet = list(set(labels.numpy()))
			idxOutput = []
			for cls in classSet:
				idx = np.where(labels == cls)[0][:sampleNum]
				idxOutput.extend(idx)
			images = images[idxOutput]
			labels = labels[idxOutput]
		print()
		# Create Dataset and DataLoader
		trainDataset = myDataset(images, labels, classes, transform_train, mtype=methodType)
		trainLoader = DataLoader(trainDataset, batch_size=trainBS, shuffle=True, num_workers=numWorkers)
		testDataset = myDataset(testImages, testLabels, classes, transform_test)
		testLoader = DataLoader(testDataset, batch_size=testBS, shuffle=False, num_workers=numWorkers)
	elif(dataType == 'MIML'):
		classes = ['{}'.format(i) for i in range(5)]
		num_classes = len(classes)
		shape = (3, 50, 50)

		if(methodType == 'bc'):
			normalize = ZeroMean
		else:
			normalize = transforms.Normalize
		
		def get_MIML_dataset():
			imagePath = "{}/images/".format(dataPath)
			labelPath = "{}/labels/".format(dataPath)
			classFile = "{}/classes.txt".format(dataPath)
			images, labels = [], []
			with open(classFile, "r") as f:
				classes = np.array(f.read().splitlines())
			for root, dirs, files in os.walk(imagePath):
				for file in files:
					absPath = os.path.join(root, file)
					image = cv2.imread(absPath, 1)
					image = cv2.resize(image, (shape[1], shape[2]))
					labelFile = os.path.join(labelPath, file + ".txt")
					with open(labelFile, "r") as f:
						label = f.read().splitlines()
					if(len(label) == 1):
						label = np.eye(len(classes))[int(np.where(classes == label)[0])]
					else:
						oriLabel = label
						for i in range(len(oriLabel)):
							label[i] = np.eye(len(classes))[int(np.where(classes == oriLabel[i])[0])]
						label = np.sum(label, 0)
					images.append(image)
					labels.append(label)
			images = np.array(images)
			labels = np.array(labels)
			perm = np.random.permutation(len(images))
			images = images[perm]
			labels = labels[perm]
			trainImages = images[:1000]
			trainLabels = labels[:1000]
			testImages = images[1000:]
			testLabels = labels[1000:]
			return trainImages, trainLabels, testImages, testLabels

		images, labels, testImages, testLabels = get_MIML_dataset()
		mean, std = cal_mean_std(np.concatenate([images, testImages]))
		transform_train = transforms.Compose([
			transforms.RandomHorizontalFlip(),
			transforms.RandomCrop(shape[1], padding=4),
			transforms.ToTensor(),
			normalize(mean=mean, std=std),
			# normalize(mean=[0.4007, 0.4100, 0.4252], std=[0.2082, 0.2140, 0.2402]),
		])
		transform_test = transforms.Compose([
			transforms.ToTensor(),
			normalize(mean=mean, std=std),
			# normalize(mean=[0.4007, 0.4100, 0.4252], std=[0.2082, 0.2140, 0.2402]),
		])
		# Load training data
		images = torch.from_numpy(images).float()
		labels = torch.from_numpy(labels).float()
		# Load testing data
		testImages = torch.from_numpy(testImages).float()
		testLabels = torch.from_numpy(testLabels).float()

		# Sample if needed
		if(sampleNum):
			classSet = list(set(labels.numpy()))
			idxOutput = []
			for cls in classSet:
				idx = np.where(labels == cls)[0][:sampleNum]
				idxOutput.extend(idx)
			images = images[idxOutput]
			labels = labels[idxOutput]

		# Create Dataset and DataLoader
		trainDataset = myDataset(images, labels, classes, transform_train, mtype=methodType, onehot=False)
		trainLoader = DataLoader(trainDataset, batch_size=trainBS, shuffle=True, num_workers=numWorkers)
		testDataset = myDataset(testImages, testLabels, classes, transform_test, onehot=False)
		testLoader = DataLoader(testDataset, batch_size=testBS, shuffle=False, num_workers=numWorkers)
	elif(dataType == 'MNIST'):
		classes = ['{}'.format(i) for i in range(10)]
		num_classes = len(classes)
		shape = (1, 28, 28)

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

		# Load training data
		oriTrainDataset = datasets.MNIST(dataPath, train=True, download=True, transform=transform_train)
		images = oriTrainDataset.data
		labels = oriTrainDataset.targets

		# Sample if needed
		if(sampleNum):
			classSet = list(set(labels.numpy()))
			idxOutput = []
			for cls in classSet:
				idx = np.where(labels == cls)[0][:sampleNum]
				idxOutput.extend(idx)
			images = images[idxOutput]
			labels = labels[idxOutput]

		# Load testing data
		oriTestDataset = datasets.MNIST(dataPath, train=False, download=True, transform=transform_test)
		testImages = oriTestDataset.data
		testLabels = oriTestDataset.targets

		# Create Dataset and DataLoader
		trainDataset = myDataset(images, labels, classes, transform_train, mtype=methodType)
		trainLoader = DataLoader(trainDataset, batch_size=trainBS, shuffle=True, num_workers=numWorkers)
		testDataset = myDataset(testImages, testLabels, classes, transform_test)
		testLoader = DataLoader(testDataset, batch_size=testBS, shuffle=False, num_workers=numWorkers)
	elif(dataType == 'FASHION-MNIST'):
		classes = ['{}'.format(i) for i in range(10)]
		num_classes = len(classes)
		shape = (1, 28, 28)

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

		# Load training data
		oriTrainDataset = datasets.FashionMNIST(dataPath, train=True, download=True, transform=transform_train)
		images = oriTrainDataset.data
		labels = oriTrainDataset.targets

		# Sample if needed
		if(sampleNum):
			classSet = list(set(labels.numpy()))
			idxOutput = []
			for cls in classSet:
				idx = np.where(labels == cls)[0][:sampleNum]
				idxOutput.extend(idx)
			images = images[idxOutput]
			labels = labels[idxOutput]

		# Load testing data
		oriTestDataset = datasets.FashionMNIST(dataPath, train=False, download=True, transform=transform_test)
		testImages = oriTestDataset.data
		testLabels = oriTestDataset.targets

		# Create Dataset and DataLoader
		trainDataset = myDataset(images, labels, classes, transform_train, mtype=methodType)
		trainLoader = DataLoader(trainDataset, batch_size=trainBS, shuffle=True, num_workers=numWorkers)
		testDataset = myDataset(testImages, testLabels, classes, transform_test)
		testLoader = DataLoader(testDataset, batch_size=testBS, shuffle=False, num_workers=numWorkers)
	
	return trainDataset, trainLoader, testDataset, testLoader, classes, num_classes, shape