import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

data_train = MNIST('./MNIST_data/mnist',download=True,transform=transforms.Compose([transforms.Resize((28, 28)),transforms.ToTensor()]))
data_test = MNIST('./MNIST_data/mnist',train=False,download=True,transform=transforms.Compose([transforms.Resize((28, 28)),transforms.ToTensor()]))
data_train_loader = DataLoader(data_train, batch_size=128, shuffle=True, num_workers=8)
data_test_loader = DataLoader(data_test, batch_size=128, num_workers=8)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(784, 512)
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.l5 = torch.nn.Linear(64, 10)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)

net = Net()
criterion = torch.nn.CrossEntropyLoss()
#优化器选择SGD
optimizer = optim.Adam(net.parameters(), lr=2e-3)
# optimizer = optim.SGD(net.parameters(), lr=0.002, momentum=0.5)

def train(epoch):
    net.train()
    loss_list, batch_list = [], []
    for epoch in range(1, epoch + 1):
        for i, (images, labels) in enumerate(data_train_loader):
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
        output = net(images)
        avg_loss += criterion(output, labels).sum()
        pred = output.detach().max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()

    avg_loss /= len(data_test)
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.detach().cpu().item(), float(total_correct) / len(data_test)))


if __name__ == '__main__':
    model_dir = './train_full_FC_params'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    train(100)