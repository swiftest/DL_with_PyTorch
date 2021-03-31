import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import pandas as pd
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse
import torch.multiprocessing as mp
import os


parser = argparse.ArgumentParser(description='PyTorch MNIST')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', 
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', 
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N', 
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR', 
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', 
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--seed', type=int, default=1, metavar='S', 
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N', 
                    help='how many batches to wait before logging training status')
parser.add_argument('--num-processes', type=int, default=6, metavar='N', 
                    help='how many training processes to use (default: 6)')
parser.add_argument('--cuda', action='store_true', default=True, 
                    help='enables CUDA training')


def train(rank, args, model, device, dataloader_kwargs):
    # 手动设置随机种子
    torch.manual_seed(args.seed + rank)
    # 加载训练数据
    train_loader = DataLoader(datasets.MNIST('./pytorch_dataset/', train=True, download=True, 
                                             transform=Compose([ToTensor(), Normalize((0.,), (1,))])), 
                              batch_size=args.batch_size, shuffle=True, num_workers=1, **dataloader_kwargs)
    # 使用随机梯度下降进行优化
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    # 开始训练，训练epochs次
    for epoch in range(1, args.epochs + 1):
        train_epoch(epoch, args, model, device, train_loader, optimizer)       
        
        
def train_epoch(epoch, args, model, device, data_loader, optimizer):
    # 模型转换为训练模式
    model.train()
    pid = os.getpid()
    for batch_idx, (data, target) in enumerate(data_loader):
        # 优化器梯度设置为0
        optimizer.zero_grad()
        # 输入特征预测值
        output = model(data.to(device=device))
        # 用预测值与标准值计算损失
        loss = F.nll_loss(output, target.to(device=device))
        # 计算梯度
        loss.backward()
        # 更新梯度
        optimizer.step()
        # 每100个小批次打印一下日至
        if batch_idx % 100 == 0:
            print('{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format
                  (pid, epoch, batch_idx * len(data), len(data_loader.dataset), 
                   100. * batch_idx / len(data_loader), loss.item()))


def test(args, model, device, dataloader_kwargs):
    # 设置随机种子
    torch.manual_seed(args.seed)
    # 加载测试数据
    test_loader = DataLoader(datasets.MNIST('./pytorch_dataset/', train=False, 
                                            transform=Compose([ToTensor(), Normalize((0.,), (1.,))])), 
                             batch_size=args.test_batch_size, shuffle=True, num_workers=1, 
                             **dataloader_kwargs)
    # 运行测试
    test_epoch(model, device, test_loader)
    
    
def test_epoch(model, device, data_loader):
    # 将模型转换为测试模式
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            output = model(data.to(device))
            test_loss += F.nll_loss(output, target.to(device), reduction='sum').item()
            # 得到概率最大的索引
            pred = output.max(1)[1]
            # 如果预测的索引和目标索引相同，则认为预测正确
            correct += pred.eq(target.to(device)).sum().item()
    test_loss /= len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format
          (test_loss, correct, len(data_loader.dataset), 100. * correct / len(data_loader.dataset)))
    
    
class CNN_Net(nn.Module):
    def __init__(self):
        super(CNN_Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=7)
        self.conv2 = nn.Conv2d(8, 32, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(288, 100)
        self.fc2 = nn.Linear(100, 10)
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 288)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(x, dim=1)
    

class CNN_Dilation_Net(nn.Module):
    def __init__(self):
        super(CNN_Dilation_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=7, dilation=2)
        self.conv2 = nn.Conv2d(32, 128, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(512, 100)
        self.fc2 = nn.Linear(100, 10)
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 512)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(x, dim=1)
    

if __name__ == '__main__':
    # 解析参数
    args = parser.parse_args()
    # 判断是否使用GPU设备
    use_cuda = args.cuda and torch.cuda.is_available()
    # 运行时设备
    device = torch.device("cuda" if use_cuda else "cpu")
    # 使用固定缓冲区
    dataloader_kwargs = {'pin_memory': True} if use_cuda else {}
    # 多进程训练，Windows环境使用'spawn'
    mp.set_start_method('spawn')
    # 将模型复制到GPU
    model = CNN_Dilation_Net().to(device)
    # 多进程共享模型参数
    model.share_memory()
    
    processes = []
    for rank in range(args.num_processes):
        p = mp.Process(target=train, args=(rank, args, model, device, dataloader_kwargs))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    
    # 测试模型
    test(args, model, device, dataloader_kwargs)