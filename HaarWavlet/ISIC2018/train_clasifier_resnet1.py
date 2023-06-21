import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import torch
device = torch.device('cuda')
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from torchvision.models import resnet50
# from torchvision.models import alexnet
from torchvision import transforms
import numpy as np
from tqdm import tqdm
# from torch.optim.lr_scheduler import ExponentialLR
from torch.backends import cudnn
from isic_dataset1 import Skin7

# class MinExponentialLR(ExponentialLR):
#     def __init__(self, optimizer, gamma, minimum, last_epoch=-1):
#         self.min = minimum
#         super(MinExponentialLR, self).__init__(optimizer, gamma, last_epoch=-1)
#
#     def get_lr(self):
#         return [
#             max(base_lr * self.gamma**self.last_epoch, self.min)
#             for base_lr in self.base_lrs
#         ]

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
train_transform = transforms.Compose([
    transforms.Resize(300),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(0.02, 0.02, 0.02, 0.01),
    transforms.RandomRotation([-180, 180]),
    transforms.RandomAffine([-180, 180], translate=[0.1, 0.1],
                            scale=[0.7, 1.3]),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
    ])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize(mean, std)
    ])

transform_test = transforms.Compose([transforms.ToTensor()])

trainset = Skin7(root="/remote-home/cs_igps_yangjin/MyVae2/FFC_Model/ISIC2018/data", iter_fold=1, train=True,transform=train_transform)
valset = Skin7(root="/remote-home/cs_igps_yangjin/MyVae2/FFC_Model/ISIC2018/data", iter_fold=1, train=False,transform=val_transform)
print('trainset', len(trainset))
print('valset', len(valset))

train_dataloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4, drop_last=True)
test_dataloader = DataLoader(valset, batch_size=32, shuffle=False, num_workers=4)

# net = resnet50(pretrained=True)
# net.fc = nn.Linear(2048, 7)
# net = net.cuda()

net = resnet50()
net.fc = nn.Linear(2048, 7)
net.load_state_dict(torch.load('/remote-home/cs_igps_yangjin/MyVae2/FFC_Model/ISIC2018/isic_resnet/params_186.pt'))
net = net.cuda()


criterion = nn.CrossEntropyLoss()
optimizer_G = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer_G, step_size=20, gamma=0.1)

# optimizer_G = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-8, betas=(0.9, 0.999))#0.4bigbudget 6e-5
# scheduler = MinExponentialLR(optimizer_G, gamma=0.998, minimum=1e-5)

def train(net):
    cudnn.benchmark
    train_accuracies = []
    train_losses = []
    current_step = 0
    for epoch in range(200):
        net.train()  # Sets module in training mode
        running_corrects_train = 0
        running_loss_train = 0.0

        for images, labels in train_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer_G.zero_grad()

            with torch.set_grad_enabled(True):
                outputs_train = net(images)

                _, preds = torch.max(outputs_train, 1)

                loss = criterion(outputs_train, labels)

                # Log loss
                if current_step % 100 == 0:
                    print('Step {}, Loss {}'.format(current_step, loss.item()))

                # Compute gradients for each layer and update weights
                loss.backward()  # backward pass: computes gradients
                optimizer_G.step()  # update weights based on accumulated gradients

            current_step += 1

        # store loss and accuracy values
        running_corrects_train += torch.sum(preds == labels.data).data.item()
        running_loss_train += loss.item() * images.size(0)

        train_acc = running_corrects_train / float(len(trainset))
        train_loss = running_loss_train / float(len(valset))

        train_accuracies.append(train_acc)
        train_losses.append(train_loss)
        print(f'train_accuracies {np.sum(train_accuracies)/len(train_accuracies)}, train_losses is {np.sum(train_losses)/len(train_losses)}')

        torch.save(net.state_dict(), '{}/params_{}.pt'.format(model_dir, epoch))
        if epoch % 1 == 0:
            test(net)

        scheduler.step()



def test(net):
    net.eval()
    running_corrects = 0
    for images, labels in tqdm(test_dataloader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward Pass
        outputs = net(images)

        # Get predictions
        _, preds = torch.max(outputs.data, 1)

        # Update Corrects
        running_corrects += torch.sum(preds == labels.data).data.item()

    # Calculate Accuracy
    accuracy = running_corrects / float(len(valset))

    print()
    print(f"Test Accuracy: {accuracy}")


if __name__ == '__main__':
    model_dir = './isic_resnet'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    # train(net)
    test(net)