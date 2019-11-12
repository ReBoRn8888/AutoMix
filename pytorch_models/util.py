import numpy as np
import torch
from torch.autograd import Variable

def mix(x, y, alpha=1.0):
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

def mixup_process(out, target_reweighted, lam):
	indices = np.random.permutation(out.size(0))
	out = out*lam + out[indices]*(1-lam)
	target_shuffled_onehot = target_reweighted[indices]
	target_reweighted = target_reweighted * lam + target_shuffled_onehot * (1 - lam)

	return out, target_reweighted

def get_lambda(alpha=1.0):
	'''Return lambda'''
	if alpha > 0.:
		lam = np.random.beta(alpha, alpha)
	else:
		lam = 1.
	return lam

def to_one_hot(inp,num_classes):
	y_onehot = torch.FloatTensor(inp.size(0), num_classes)
	y_onehot.zero_()

	y_onehot.scatter_(1, inp.unsqueeze(1).data.cpu(), 1)
	
	return Variable(y_onehot.cuda(),requires_grad=False)