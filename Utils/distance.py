import torch
import torch.nn.functional as F
import numpy as np
# import ot


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

# def EMD(P, Q):
# 	a = np.ones(shape[0] * shape[1] * shape[2]) / (shape[0] * shape[1] * shape[2])
# 	b = np.ones(shape[0] * shape[1] * shape[2]) / (shape[0] * shape[1] * shape[2])
# 	dis = []
# 	for i in range(len(P)):
# 		M = ot.dist(P[i].view(-1, 1).cpu().detach().numpy(), Q[i].view(-1, 1).cpu().detach().numpy(), 'euclidean')
# 		d = ot.emd2(a, b, M)
# 		dis.append(d)
# 	return torch.Tensor(dis)

# def sinkhorn(P, Q):
# 	a = np.ones(shape[0] * shape[1] * shape[2]) / (shape[0] * shape[1] * shape[2])
# 	b = np.ones(shape[0] * shape[1] * shape[2]) / (shape[0] * shape[1] * shape[2])
# 	dis = []
# 	for i in range(len(P)):
# 		M = ot.dist(P[i].view(-1, 1).cpu().detach().numpy(), Q[i].view(-1, 1).cpu().detach().numpy(), 'euclidean')
# 		d = ot.sinkhorn2(a, b, M, 1, numItermax=100)
# 		dis.append(d)
# 	return torch.Tensor(dis)

def norm1(P, Q):
	return (P - Q).abs().mean(1).mean(1).mean(1)