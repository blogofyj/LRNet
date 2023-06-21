import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.models import alexnet
# import numpy as np
# import matplotlib.pyplot as plt
device = torch.device('cuda')

transform = transforms.Compose(
    [transforms.ToTensor(),  # 转为Tensor
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]  # 归一化分别为RGB三通道的均值和标准差
)

train_data = datasets.CIFAR10(root='/remote-home/cs_igps_yangjin/MyVae3/cifar10/train_data', train=True, download=False, transform=transform)
test_data = datasets.CIFAR10(root='/remote-home/cs_igps_yangjin/MyVae3/cifar10/test_data', train=False, download=False, transform=transform)

# 通过train_loader把数据传入网络
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128, shuffle=True)



# class BasicBlock(nn.Module):  # nn.Module是所有神经网络的基类，自己定义的任何神经网络都要继承nn.Module
#     expansion = 1
#
#     def __init__(self, in_planes, planes, stride=1):
#         super(BasicBlock, self).__init__()  # 第四、五行都是python类继承的基本操作,此写法应该是python2.7的继承格式,但python3里写这个好像也可以
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,  # 输入in_planes个通道，输出planes的通道即planes个卷积核
#                                stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
#                                stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#
#         self.shortcut = nn.Sequential()
#         # 经过处理后的x 要与 x 的维度相同（尺寸和深度）
#         # 如果不相同，需要添加卷积+BN来变换为同一纬度
#         if stride != 1 or in_planes != self.expansion * planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion * planes,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(self.expansion * planes)
#             )
#
#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out
#
# class Bottleneck(nn.Module):
#     # 前面1x1和3x3卷积的filter个数相等，最后1x1卷积是其expansion倍
#     expansion = 4
#
#     def __init__(self, in_planes, planes, stride=1):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
#                                stride=stride, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#
#         self.conv3 = nn.Conv2d(planes, self.expansion * planes,
#                                kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(self.expansion * planes)
#
#         self.shorcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion * planes:
#             self.shorcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion * planes,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(self.expansion * planes)
#             )
#
#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = F.relu(self.bn2(self.conv2(out)))
#         out = self.bn3(self.conv3(out))
#         out += self.shorcut(x)
#         out = F.relu(out)
#
# class ResNet(nn.Module):
#     def __init__(self, block, num_blocks, num_classes=10):
#         super(ResNet, self).__init__()
#         self.in_planes = 64
#
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 池化层
#         self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
#         self.linear = nn.Linear(512 * block.expansion, num_classes)  # 线性层
#
#     def _make_layer(self, block, planes, num_blocks, stride=1):
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride))
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         out = F.relu(self.maxpool(self.bn1(self.conv1(x))))
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         # out = F.avg_pool2d(out, 7, stride=1)  # [(7 - 7 + 0) / 1]  + 1 = 1. 所以最后输出1x1
#         out = F.avg_pool2d(out, 1, stride=1)  # [(1 - 1 + 0) / 1]  + 1 = 1. 所以最后输出1x1
#         out = out.view(out.size(0), -1)  # .view( )是一个tensor的方法，使得tensor改变size但是元素的总数是不变的。
#         out = self.linear(out)
#         return out
#
# def ResNet18():
#     return ResNet(BasicBlock, [2, 2, 2, 2])
#
# def ResNet34():
#     return ResNet(BasicBlock, [3, 4, 6, 3])
#
# def ResNet50():
#     return ResNet(Bottleneck, [3, 4, 6, 3])
#
# def ResNet101():
#     return ResNet(Bottleneck, [3, 4, 23, 3])
#
# def ResNet152():
#     return ResNet(Bottleneck, [3, 8, 36, 3])

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Block') == -1:
        m.weight.data.normal_(0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1, 0.02)
        m.bias.data.fill_(0)

# net = ResNet50().cuda()
# net.apply(weights_init).cuda()

net = alexnet()
net.classifier[6] = nn.Linear(4096, 10)
net.apply(weights_init).cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# optimizer = optim.Adam(net.parameters(), lr=2e-3)

def train(epoch):
    net.train()
    loss_list, batch_list = [], []
    for epoch in range(1, epoch + 1):
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()

            output = net(images)

            loss = criterion(output, labels)

            loss_list.append(loss.detach().cpu().item())
            batch_list.append(i+1)

            if i % 10 == 0:
                print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.detach().cpu().item()))

            # Update Visualization
            loss.backward()
            optimizer.step()
        if epoch % 2 == 0:
            test()
        if epoch % 2 == 0:
            torch.save(net.state_dict(), '{}/params_{}.pt'.format(model_dir, epoch))
        torch.save(net.state_dict(), '{}/params_pause.pt'.format(model_dir))
        print("Current epoch saved!")


def test():
    net.eval()
    total_correct = 0
    avg_loss = 0.0
    for i, (images, labels) in enumerate(test_loader):
        images = images.cuda()
        labels = labels.cuda()
        output = net(images)
        avg_loss += criterion(output, labels).sum()
        pred = output.detach().max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()

    avg_loss /= len(test_data)
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.detach().cpu().item(), float(total_correct) / len(test_data)))


if __name__ == '__main__':
    model_dir = './cifar_black_test_resnet50'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    train(100)
