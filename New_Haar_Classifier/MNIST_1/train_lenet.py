import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
device = torch.device('cuda')

# class C1(nn.Module):
#     def __init__(self):
#         super(C1, self).__init__()
#
#         self.c1 = nn.Sequential(OrderedDict([
#             ('c1', nn.Conv2d(1, 6, kernel_size=(5, 5))),
#             ('relu1', nn.ReLU()),
#             ('s1', nn.MaxPool2d(kernel_size=(2, 2), stride=2))
#         ]))
#
#     def forward(self, img):
#         output = self.c1(img)
#         return output
#
#
# class C2(nn.Module):
#     def __init__(self):
#         super(C2, self).__init__()
#
#         self.c2 = nn.Sequential(OrderedDict([
#             ('c2', nn.Conv2d(6, 16, kernel_size=(5, 5))),
#             ('relu2', nn.ReLU()),
#             ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2))
#         ]))
#
#     def forward(self, img):
#         output = self.c2(img)
#         return output
#
#
# class C3(nn.Module):
#     def __init__(self):
#         super(C3, self).__init__()
#
#         self.c3 = nn.Sequential(OrderedDict([
#             ('c3', nn.Conv2d(16, 120, kernel_size=(5, 5))),
#             ('relu3', nn.ReLU())
#         ]))
#
#     def forward(self, img):
#         output = self.c3(img)
#         return output
#
#
# class F4(nn.Module):
#     def __init__(self):
#         super(F4, self).__init__()
#
#         self.f4 = nn.Sequential(OrderedDict([
#             ('f4', nn.Linear(120, 84)),
#             ('relu4', nn.ReLU())
#         ]))
#
#     def forward(self, img):
#         output = self.f4(img)
#         return output
#
#
# class F5(nn.Module):
#     def __init__(self):
#         super(F5, self).__init__()
#
#         self.f5 = nn.Sequential(OrderedDict([
#             ('f5', nn.Linear(84, 10)),
#             ('sig5', nn.LogSoftmax(dim=-1))
#         ]))
#
#     def forward(self, img):
#         output = self.f5(img)
#         return output


# class LeNet5(nn.Module):
#     """
#     Input - 1x32x32
#     Output - 10
#     """
#     def __init__(self):
#         super(LeNet5, self).__init__()
#
#         self.c1 = C1()
#         self.c2_1 = C2()
#         self.c2_2 = C2()
#         self.c3 = C3()
#         self.f4 = F4()
#         self.f5 = F5()
#
#     def forward(self, img):
#         output = self.c1(img)
#
#         x = self.c2_1(output)
#         output = self.c2_2(output)
#
#         output += x
#
#         output = self.c3(output)
#         output = output.view(img.size(0), -1)
#         output = self.f4(output)
#         output = self.f5(output)
#         return output

class LeNet5(torch.nn.Module):
    def __init__(self, num_classes):
        super(LeNet5, self).__init__()
        self.layer1 = torch.nn.Sequential(torch.nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
                                          torch.nn.BatchNorm2d(6),
                                          torch.nn.ReLU(),
                                          torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = torch.nn.Sequential(torch.nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
                                          torch.nn.BatchNorm2d(16),
                                          torch.nn.ReLU(),
                                          torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = torch.nn.Sequential(torch.nn.Linear(4 * 4 * 16, 120),
                                       torch.nn.ReLU())
        self.fc2 = torch.nn.Sequential(torch.nn.Linear(120, 84),
                                       torch.nn.ReLU())
        self.fc3 = torch.nn.Linear(84, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Block') == -1:
        m.weight.data.normal_(0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1, 0.02)
        m.bias.data.fill_(0)

data_train = MNIST('./MNIST_data/mnist',download=False,transform=transforms.Compose([transforms.ToTensor()]))
data_test = MNIST('./MNIST_data/mnist',train=False,download=False,transform=transforms.Compose([transforms.ToTensor()]))
data_train_loader = DataLoader(data_train, batch_size=128, shuffle=True, num_workers=8)
data_test_loader = DataLoader(data_test, batch_size=128, num_workers=8)

learning_rate = 0.001

net = LeNet5(10)
net.apply(weights_init)
net = net.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)



def train(epoch):
    net.train()
    loss_list, batch_list = [], []
    for epoch in range(1, epoch + 1):
        for i, (images, labels) in enumerate(data_train_loader):
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
    for i, (images, labels) in enumerate(data_test_loader):
        images = images.cuda()
        labels = labels.cuda()
        output = net(images)
        avg_loss += criterion(output, labels).sum()
        pred = output.detach().max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()

    avg_loss /= len(data_test)
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.detach().cpu().item(), float(total_correct) / len(data_test)))


if __name__ == '__main__':
    model_dir = './train_lenet5_params_new'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    train(100)