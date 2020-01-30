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
    parser.add_argument('--arch', dest='arch', default='resnet18', type=str, choices=['mynet', 'resnet18', 'preactresnet18'], help='Backbone architecture : [mynet, resnet18, preactresnet18]')
    parser.add_argument('--dataset', dest='dataset', default='IMAGENET', type=str, choices=['IMAGENET', 'TINY-IMAGENET', 'CIFAR10', 'CIFAR100', 'MNIST', 'FASHION-MNIST', 'GTSRB', 'MIML'], help='Dataset to be trained : [IMAGENET, TINY-IMAGENET, CIFAR10, CIFAR100, MNIST, FASHION-MNIST, GTSRB, MIML]')
    parser.add_argument('--sample_num', dest='sample_num', default=None, type=int, help='The number of images sampled per class from dataset')
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
    parser.add_argument('--parallel', dest='parallel', default=True, type=bool, help='Train parallelly with multi-GPUs?')
    parser.add_argument('--log_path', dest='log_path', default=None, type=str, help='Path to the dataset')
    parser.add_argument('--pretrain', dest='pretrain', default=False, type=bool, help='Whether to use pretrained weights for ResNets.')

    args = parser.parse_args()
    return args

class myLoss(nn.Module):
    def __init__(self, mtype='baseline'):
        super(myLoss, self).__init__()
        self.type = mtype.lower()
        
    def forward(self, pred, truth):
#         pred += 1e-9
#         truth += 1e-9
        pred = F.log_softmax(pred, 1)
        if(self.type in ['bc', 'automix']):
#             entropy = - torch.sum(truth[truth > 0] * torch.log(truth[truth > 0]), 1)
            entropy = - torch.sum(truth * torch.log((truth+1e-9)), 1)
            crossEntropy = - torch.sum(truth * pred, 1)
            loss = crossEntropy - entropy
        else:
            loss = -torch.sum(pred * truth, 1)
        return loss.mean()

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k)
    return res

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

def load_pretrained_model(net, weightPath):
    # Load pretrained dict
    pretrained_dict = torch.load(weightPath)
    # Current state dict
    model_dict = net.state_dict()
    # Select params that need to load
    pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict and k.find('fc') < 0}
    # Update state dict
    model_dict.update(pretrained_dict)
    # Load new state dict
    net.load_state_dict(model_dict)

    return net

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
        print_log('Epoch {}/{}, lr = {}  [best_acc = {:.4f}%]'.format(epoch+1, n_epochs, optimizer.param_groups[0]['lr'], best_acc), logger, 'info')
        print_log('-' * 10, logger, 'info')
        epochStart = time.time()
        for phase in ['train', 'val']:
            if(phase == 'train'):
                # Update learning_rate
                exp_lr_scheduler.step()
                net.train()  # Set model to training mode
            else:
                net.eval()   # Set model to evaluate mode
            losses = AverageMeter()
            if(dataset == 'MIML'):
                # Init PR records for MIML
                PR_labels, PR_results = np.array([]), np.array([])
            else:
                # Init AverageMeter for single-label classification
                top1 = AverageMeter()
                top5 = AverageMeter()   

            for i, data in enumerate(dataLoader[phase], 0):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                targets = labels
                lam = None
                with torch.set_grad_enabled(phase == 'train'):
                    if(method == 'mixup' and phase == 'train'):
                        inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, 1.0)
                    elif(method == 'adamixup' and phase == 'train'):
                        inputs_a, inputs_b, labels_a, labels_b = get_shuffled_data(inputs, labels)
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
                        inputs = torch.cat([x_mix, inputs_a], 0)
                        labels = torch.cat([y_mix, labels_a], 0)
                        inputs, targets = shuffle(inputs, labels)
                    elif(method == 'automix' and phase == 'train'):
                        lam = np.random.beta(1.0, 1.0)

                        midIdx = int(len(inputs) / 2)
                        inputs_a = inputs[:midIdx]
                        x_mix, inputs_b, y_mix = unet(inputs_a, labels[:midIdx], lam)
                        inputs = torch.cat([x_mix, inputs[midIdx:]], 0)
                        targets = torch.cat([y_mix, labels[midIdx:]], 0)

                        # inputs_a = inputs
                        # x_mix, inputs_b, y_mix = unet(inputs_a, labels, lam)
                        # inputs = x_mix
                        # targets = y_mix

                        if(i in [0]):
                            cmap = 'gray'
                            plt.subplot(221)
                            plt.imshow(inputs_a[0].permute(1, 2, 0).squeeze().cpu().detach().numpy(), cmap=cmap)
                            plt.subplot(222)
                            plt.imshow(inputs_b[0].permute(1, 2, 0).squeeze().cpu().detach().numpy(), cmap=cmap)
                            plt.subplot(223)
                            plt.title('{:.4f}-unet'.format(lam))
                            plt.imshow(x_mix[0].permute(1, 2, 0).squeeze().cpu().detach().numpy(), cmap=cmap)
                            plt.subplot(224)
                            plt.title('{:.4f}-mixup'.format(lam))
                            plt.imshow((inputs_a[0] * lam + inputs_b[0] * (1 - lam)).permute(1, 2, 0).squeeze().cpu().detach().numpy(), cmap=cmap)
                            plt.savefig("out/epoch-{}[{}].jpg".format(epoch, i))
                            plt.show()

                    # Feed forward
                    if(phase == 'train' and method == 'manifoldmixup'):
                        outputs, targets = net(inputs, labels, manifoldMixup=True)
                    else:
                        outputs = net(inputs)

                    # Calculate [classification] loss
                    if(method in ['mixup'] and phase == 'train'):
                        if(criterionType == 'celoss'):
                            labels_a = torch.argmax(labels_a, 1)
                            labels_b = torch.argmax(labels_b, 1)
                        clsLoss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                    else:
                        if(criterionType == 'celoss'):
                            targets = torch.argmax(targets, 1)
                        clsLoss = criterion(outputs, targets)

                    # Calculate [reconstruction] loss
                    if(method == 'automix' and phase == 'train'):
                        d1 = norm1(inputs_a, x_mix)
                        d2 = norm1(inputs_b, x_mix)
#                         disLoss = lam[:int(lam.shape[0]/3)] * d1 + (1 - lam[:int(lam.shape[0]/3)]) * d2
                        disLoss = (lam * d1 + (1 - lam) * d2).mean()
#                         disLoss = d1 + d2
                        loss = clsLoss + 1.5 * disLoss
                    elif(method == 'adamixup' and phase == 'train'):
                        x_pos = torch.cat([inputs_unseen], 0)
                        y_pos = torch.zeros(len(x_pos), 2).scatter_(1, torch.ones(len(x_pos), 1).long(), 1).to(device)
                        x_neg = torch.cat([x_mix], 0)
                        y_neg = torch.zeros(len(x_neg), 2).scatter_(1, torch.zeros(len(x_neg), 1).long(), 1).to(device)
                        x_bin = torch.cat([x_pos, x_neg], 0)
                        y_bin = torch.cat([y_pos, y_neg], 0)
                        x_bin, y_bin = shuffle(x_bin, y_bin)
                        extra = net(x_bin)
                        logits = net.linear2(extra)
                        disLoss = criterion(logits, y_bin)
                        loss = clsLoss + disLoss
                    else:
                        loss = clsLoss

                    # Feed backward
                    if(phase == 'train'):
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                losses.update(loss.item()*inputs.size(0), inputs.size(0))

                sys.stdout.write('                                                                                                 \r')
                sys.stdout.flush()
                if(dataset == 'MIML'):
                    # Calculate mAP for multi-label classification
                    outputs = sigmoid(outputs)
                    PR_results = np.concatenate([PR_results, outputs.cpu().detach().numpy()], 0)
                    PR_labels = np.concatenate([PR_labels, labels.cpu().detach().numpy()], 0)
                    precision = mAP_evaluation(PR_results, PR_labels, num_classes)
                    sys.stdout.write('Iter: {} / {} ({:.0f}s)\tLoss= {:.4f} ({:.4f})\tAcc= {:.2f}%\r'
                         .format(i+1, iterNum[phase], time.time() - epochStart, loss.item(), losses.avg, precision*100))
                else:
                    # Calculate top1/5 accuracy for single-label classification
                    prec1, prec5 = accuracy(outputs, torch.argmax(labels, 1), topk=(1, 5))
                    top1.update(prec1.item(), inputs.size(0))
                    top5.update(prec5.item(), inputs.size(0))
                    sys.stdout.write('Iter: {} / {} ({:.0f}s)\tLoss= {:.4f} ({:.4f})\tAcc= {:.2f}% ({:.0f}/{:.0f})\r'
                         .format(i+1, iterNum[phase], time.time() - epochStart, loss.item(), losses.avg, prec1/inputs.size(0)*100, top1.sum, top1.count))
                sys.stdout.flush()
            sys.stdout.write('                                                                                                  \r')
            sys.stdout.flush()

            epoch_loss = losses.avg
            epoch_acc = mAP_evaluation(PR_results, PR_labels, num_classes)*100 if dataset == 'MIML' else top1.avg*100
            accLog[phase].append(epoch_acc/100)
            lossLog[phase].append(epoch_loss)
            epochDuration = time.time() - epochStart
            epochStart = time.time()
            hour, minute, second = convert_secs2time(epochDuration)
            if(dataset == 'MIML'):
                print_log('[ {} ]  Loss: {:.4f} Acc: {:.3f}% ({:.0f}h {:.0f}m {:.2f}s)'
                    .format(phase, epoch_loss, epoch_acc, hour, minute, second), logger, 'info')
            else:
                print_log('[ {} ]  Loss: {:.4f} Acc: {:.3f}% ({:.0f}/{:.0f}) ({:.0f}h {:.0f}m {:.2f}s)'
                    .format(phase, epoch_loss, epoch_acc, top1.sum, top1.count, hour, minute, second), logger, 'info')
            
            if(phase == 'val' and epoch_acc > best_acc):
                print_log('Saving best model to {}'.format(os.path.join(modelPath, modelName)), logger, 'info')
                if(method == 'automix'):
                    state = {'net': [net, unet], 'opt': optimizer, 'acc': epoch_acc, 'epoch': epoch}
                elif(method == 'adamixup'):
                    state = {'net': [net, PRG_net], 'opt': optimizer, 'acc': epoch_acc, 'epoch': epoch}
                else:
                    state = {'net': net, 'opt': optimizer, 'acc': epoch_acc, 'epoch': epoch}
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
                torch.save(state, os.path.join(modelPath, finalModelName))
        print_log('', logger, 'info')

        log = dict({'acc': accLog, 'loss': lossLog})
        with open(os.path.join(modelPath, '{}-log-{}.pkl'.format(expName, fold+1)), 'wb') as f:
            pickle.dump(log, f)
            if(epoch + 1 == n_epochs):
                print_log("Training logs saved to : {}".format(os.path.join(modelPath, '{}-log-{}.pkl'.format(expName, fold+1))), logger, 'info')
        plot_acc_loss(log, 'both', modelPath, logger, '{}-'.format(expName), '-{}'.format(fold+1), (epoch + 1 == n_epochs))
        plot_acc_loss(log, 'loss', modelPath, logger, '{}-'.format(expName), '-{}'.format(fold+1), (epoch + 1 == n_epochs))
        plot_acc_loss(log, 'accuracy', modelPath, logger, '{}-'.format(expName), '-{}'.format(fold+1), (epoch + 1 == n_epochs))
    duration = time.time() - start
    print_log('Training complete in {:.0f}h {:.0f}m {:.2f}s'.format(duration // 3600, (duration % 3600) // 60, duration % 60), logger, 'info')
    print_log('Best val Acc: {:4f}'.format(best_acc), logger, 'info')

    return best_acc, log

def get_model(netType, methodType, num_classes, shape, pretrain):
    if(netType == 'mynet'):
        net = MyNet(input_shape=shape, 
                    num_classes=num_classes)
    elif(netType == 'resnet18'):
        net = ResNet18(input_shape=shape, 
                       num_classes=num_classes,
                       pretrained=pretrain)
        if(pretrain):
            print_log('Pretrained weight loaded!', logger, 'info')
    elif(netType == 'preactresnet18'):
        net = PreActResNet18(input_shape=shape, 
                             num_classes=num_classes)

    net = net.to(device)
    outputNet = [net]
    parameters = [{'params': net.parameters()}]
    print_log('Backbone : [{}]\n{}'.format(netType, summary(model=net, input_size=shape)), logger, 'info')
    
    if(methodType == 'automix'):
        unet = UNet(input_shape=(shape[0], shape[1], shape[2]), output_shape=shape, num_classes=num_classes)
        unet = unet.to(device)
        # print_log('UNet : {}'.format(summary(model=unet, input_size=(shape[0], shape[1], shape[2]))), logger, 'info')
        outputNet.append(unet)
        parameters.append({'params': unet.parameters()})
    if(methodType == 'adamixup'):
        if(netType == 'mynet'):
            PRG_net = MyNet(input_shape=shape, 
                            num_classes=3)
        elif(netType == 'resnet18'):
            PRG_net = ResNet18(input_shape=shape, 
                               num_classes=3,
                               pretrained=pretrain)
            if(pretrain):
                print_log('Pretrained weight loaded!', logger, 'info')
        elif(netType == 'preactresnet18'):
            PRG_net = PreActResNet18(input_shape=shape, 
                                     num_classes=3)
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
                'IMAGENET'      : ['/data/ImageNet', 300, [75, 150, 225, 275]],
                'TINY-IMAGENET' : ['/data/tiny-imagenet/tiny-imagenet-200', 300, [75, 150, 225, 275]],
                'CIFAR10'       : ['Dataset/cifar10', 300, [75, 150, 225]],
                'CIFAR100'      : ['Dataset/cifar100', 300, [75, 150, 225]],
                'MNIST'         : ['Dataset/mnist', 100, [50, 75]],
                'FASHION-MNIST' : ['Dataset/fashion-mnist', 100, [50, 75]],
                'GTSRB'         : ['Dataset/GTSRB', 100, [50, 75]],
                'MIML'          : ['Dataset/MIML', 100, [50, 75]],
                }

    args = parse_args()

    dataset = str(args.dataset).upper()
    method = str(args.method).lower()
    arch = str(args.arch).lower()
    sampleNum = int(args.sample_num) if args.sample_num else args.sample_num
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
    n_epochs = int(args.epoch)
    kfold = int(args.kfold)
    pretrain = args.pretrain
    expName = get_exp_name(dataset=dataset,
                            arch=arch,
                            epochs=n_epochs,
                            batch_size=trainBS,
                            lr=learning_rate,
                            momentum=momentum,
                            decay=weight_decay,
                            method=method,
                            criterion=criterionType,
                            sampleNum=sampleNum)
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

    trainDataset, trainLoader, testDataset, testLoader, classes, num_classes, shape = get_dataset(dataset, method, dataDir, trainBS, testBS, numWorkers, sampleNum)
    
    # it = iter(trainLoader)
    # a, b = it.next()
    # print(a.shape, b.shape)
    print_log('Dataset [{}] loaded!'.format(dataset), logger, 'info')
    print_log('Training : {} images - {} labels'.format(len(trainDataset.images), len(trainDataset.labels)), logger, 'info')
    print_log('Testing  : {} images - {} labels'.format(len(testDataset.images), len(testDataset.labels)), logger, 'info')

    modelPath = 'pytorch_model_learnt/{}/{}/{}'.format(dataset, arch, method)
    if not os.path.isdir(modelPath):
        os.makedirs(modelPath)

    for fold in tqdm(range(kfold), desc='Fold'):
        print_log('=====================================================', logger, 'info')
        print_log('====================  Fold #{}  ====================='.format(fold), logger, 'info')
        print_log('=====================================================', logger, 'info')
        modelName = '{}-{}.ckpt'.format(expName, fold+1)
        nets, parameters = get_model(arch, method, num_classes, shape, pretrain)
        if(method == 'automix'):
            net, unet = nets
        elif(method == 'adamixup'):
            net, PRG_net = nets
        else:
            net = nets[0]

        softmax = nn.Softmax(dim=1).to(device)
        sigmoid = nn.Sigmoid().to(device)
        if(criterionType == 'bceloss'):
            criterion = nn.BCELoss().to(device)
        elif(criterionType == 'myloss'):
            criterion = myLoss(method).to(device)
        elif(criterionType == 'mseloss'):
            criterion = nn.MSELoss().to(device)
        elif(criterionType == 'celoss'):
            criterion = nn.CrossEntropyLoss().to(device)

        optimizer = optim.SGD(parameters, lr=learning_rate, momentum=momentum, weight_decay=weight_decay, nesterov=True)
    #     optimizer = optim.Adam(parameters, lr=0.0002, betas=(0.5, 0.999))
        exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, lrDecayStep, gamma=0.1)
        # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        best_acc, log = train_val(optimizer, 
                                  n_epochs, 
                                  trainDataset, 
                                  trainLoader, 
                                  testDataset, 
                                  testLoader)

        if(method == 'automix'):
            net, unet = torch.load(os.path.join(modelPath, modelName))['net']
        elif(method == 'adamixup'):
            net, PRG_net = torch.load(os.path.join(modelPath, modelName))['net']
        else:
            net = torch.load(os.path.join(modelPath, modelName))['net']
        if(dataset != 'MIML'):
            accTotal = eval_total(net, testLoader)
            accPerClass = eval_per_class(net, testLoader, classes)
