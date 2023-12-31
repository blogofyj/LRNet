import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from train_isic_mode_1l import classifier,test_loader

print('model', classifier)

class Net(nn.Module):

    def __init__(self, num_classes=7):
        super(Net, self).__init__()
        self.num_classes = num_classes
        self.resnet = models.resnet50(pretrained=True)

        for i, param in enumerate(self.resnet.parameters()):
            param.requires_grad = False

        self.a_convT2d = nn.ConvTranspose2d(in_channels=2048, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.b_convT2d = nn.ConvTranspose2d(in_channels=1280, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.c_convT2d = nn.ConvTranspose2d(in_channels=640, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.convT2d3 = nn.ConvTranspose2d(in_channels=320, out_channels=self.num_classes, kernel_size=4, stride=4,
                                           padding=0)

    def setTrainableLayers(self, trainable_layers):
        for name, node in self.resnet.named_children():
            unlock = name in trainable_layers
            for param in node.parameters():
                param.requires_grad = unlock

    def forward(self, x):

        skipConnections = {}
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        skipConnections[1] = x = self.resnet.layer1(x)  # [10, 256, 56, 56]

        skipConnections[2] = x = self.resnet.layer2(x)  # [10, 512, 28, 28]

        skipConnections[3] = x = self.resnet.layer3(x)  # [10, 1024, 14, 14]

        skipConnections[4] = x = self.resnet.layer4(x)  # [10, 2048, 7, 7]

        x = self.a_convT2d(x)  # [10, 256, 14, 14]

        x = torch.cat((x, skipConnections[3]), 1)

        x = self.b_convT2d(x)  # [10, 128, 28, 28]

        x = torch.cat((x, skipConnections[2]), 1)

        x = self.c_convT2d(x)  # [10, 64, 56, 56]

        x = torch.cat((x, skipConnections[1]), 1)

        x = self.convT2d3(x)  # [10, num_classes, 224, 224]

        x = nn.Sigmoid()(x)
        x = x.view(x.size()[0], -1, self.num_classes)

        return x

    def load(self, filename):
        self.load_state_dict(torch.load(filename))


net = Net().cuda()
net.load('./lesions.pth')

def test(epoch, net, test_loader):
    net.eval()
    total = 0
    correct = 0

    for i, (images, labels) in enumerate(test_loader):
        inputs = images.cuda()
        y = labels.cuda()
        with torch.no_grad():
            out = net(inputs)
            out = torch.transpose(out, 2, 1)
            print('out.shape', out.shape)
            out = out.reshape((32,1,224,224))

            logit = classifier(out)
            prediction = torch.max(logit, 1)[1]
            correct = correct + torch.eq(prediction, y).float().sum().item()

            total = total + inputs.size(0)
    accuracy = correct / total
    print()
    print('TEST *TOP* ACC:{:.4f} at e:{:03d}'.format(accuracy, epoch))
    print()

if __name__ == '__main__':
    test(100, net, test_loader)