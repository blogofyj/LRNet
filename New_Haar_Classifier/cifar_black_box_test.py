import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
device = torch.device('cuda')

transform = transforms.Compose(
    [transforms.ToTensor(),  # 转为Tensor
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]  # 归一化分别为RGB三通道的均值和标准差
)
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

train_data = datasets.CIFAR10(root='/remote-home/cs_igps_yangjin/MyVae3/cifar10/train_data', train=True, download=False, transform=transform)
# test_data = datasets.CIFAR10(root='/remote-home/cs_igps_yangjin/MyVae3/cifar10/test_data', train=False, download=False, transform=transform)

# 通过train_loader把数据传入网络
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
# test_loader = DataLoader(test_data, batch_size=32, shuffle=True)

class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=10):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Sequential(
            nn.Linear(in_features=512, out_features=4096),  # in_features = 1x1x512
            nn.ReLU(True),
            nn.Dropout(),

            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(True),
            nn.Dropout(),

            nn.Linear(in_features=4096, out_features=num_classes)
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)  # 1x1x512
        out = self.classifier(out)
        return F.log_softmax(out, dim=1)


    def _make_layers(self, cfg):
        layers = []
        in_channels = 3  # 最开始输入通道为RGB图像，所以3通道
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:  # 一次性向列表中加入三个操作即为一个块
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]  # 当inplace=True的时候，会改变输入数据；当inplace=False的时候，不会改变输入数据
                in_channels = x
        return nn.Sequential(*layers)


net = VGG('VGG19').cuda()
# net.load_state_dict(torch.load('/remote-home/cs_igps_yangjin/MyVae3/New_Haar_Classifier/cifar_black_test_vgg16/params_56.pt'))

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
        # if epoch % 2 == 0:
        #     test()
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
        avg_loss = avg_loss + criterion(output, labels).sum()
        pred = output.detach().max(1)[1]
        total_correct = total_correct + pred.eq(labels.view_as(pred)).sum()

    avg_loss /= len(test_data)
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.detach().cpu().item(), float(total_correct) / len(test_data)))


if __name__ == '__main__':
    model_dir = './cifar_black_test_vgg19'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    train(100)
