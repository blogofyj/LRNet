import os, sys
sys.path.append(os.pardir)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch,torchvision
import torch.nn as nn
import numpy as np
from attack_model import InvNet
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR
import torch.fft as fft
from HaarWavlet.resnet_model import ResNet18
from torch.utils import data
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import torch.nn.functional as F
device = torch.device('cuda')


classifier = ResNet18()
classifier.load_state_dict(torch.load('/remote-home/cs_igps_yangjin/DC-VAE-main/models/Result/ResNet18/params_finished.pt'))
classifier = classifier.cuda()
classifier.eval()


class Dataset(data.Dataset):
    def __init__(self, data, labels, adv_data):
        self.data = torch.from_numpy(data)
        self.labels = torch.from_numpy(labels)
        self.adv_data = torch.from_numpy(adv_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X = self.data[index]
        y = self.labels[index]
        z = self.adv_data[index]

        return X, y, z

transform = transforms.Compose([transforms.ToTensor()])
train_data = torchvision.datasets.CIFAR10(root='../cifar10/train_data', train=True, transform=transform, download=False)
dataloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=6)

test_data = torchvision.datasets.CIFAR10(root='../cifar10/test_data', train=False, transform=transform, download=False)
testloader = torch.utils.data.DataLoader(dataset=test_data, batch_size=32, shuffle=True)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Block') == -1:
        m.weight.data.normal_(0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1, 0.02)
        m.bias.data.fill_(0)


# block_num = [8, 8]
block_num = [4]
netG = InvNet(block_num=block_num)
netG.apply(weights_init)
netG = netG.cuda()
netG.train()


class MinExponentialLR(ExponentialLR):
    def __init__(self, optimizer, gamma, minimum, last_epoch=-1):
        self.min = minimum
        super(MinExponentialLR, self).__init__(optimizer, gamma, last_epoch=-1)

    def get_lr(self):
        return [
            max(base_lr * self.gamma**self.last_epoch, self.min)
            for base_lr in self.base_lrs
        ]

# optim_params = []
# for k, v in netG.named_parameters():
#     if v.requires_grad:
#         optim_params.append(v)

optimizer_G = torch.optim.Adam(netG.parameters(), lr=2e-3, weight_decay=1e-8, betas=(0.9, 0.999))
# scheduler = MultiStepLR(optimizer_G, milestones=[50.0, 100.0], gamma=0.5)
scheduler = MinExponentialLR(optimizer_G, gamma=0.998, minimum=1e-5)



class SSIM_Loss(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM_Loss, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


Rec_back_SSIM = SSIM_Loss()

class ReconstructionLoss(nn.Module):
    def __init__(self, losstype='l2', eps=1e-3):
        super(ReconstructionLoss, self).__init__()
        self.losstype = losstype
        self.eps = eps

    def forward(self, x, target):
        if self.losstype == 'l2':
            # print(x.shape,target.shape)
            return torch.mean(torch.sum((x - target)**2, (1, 2, 3)))
        elif self.losstype == 'l1':
            diff = x - target
            return torch.mean(torch.sum(torch.sqrt(diff * diff + self.eps), (1, 2, 3)))
        elif self.losstype == 'l_log':
            diff = x - target
            eps = 1e-6
            return torch.mean(torch.sum(-torch.log(1-diff.abs()+eps), (1, 2, 3)))
        else:
            print("reconstruction loss type error!")
            return 0


crossLoss = nn.CrossEntropyLoss()
crossLoss.cuda()
# Reconstruction_forw = ReconstructionLoss(losstype='l2')
# Reconstruction_back = ReconstructionLoss(losstype='l1')
Reconstruction_back = nn.MSELoss()
Reconstruction_back.cuda()


h, w = 32,32
lpf = torch.zeros((h, w))
R = (h+w)//8  #或其他
for x in range(w):
    for y in range(h):
        if ((x-(w-1)/2)**2 + (y-(h-1)/2)**2) < (R**2):
            lpf[y,x] = 1
hpf = 1-lpf
hpf, lpf = hpf.cuda(), lpf.cuda()

def pinglvyu(data):
    f = fft.fftn(data, dim=(2,3)).cuda()
    f = torch.roll(f, (h // 2, w // 2), dims=(2, 3))  # 移频操作,把低频放到中央
    f_l = f * lpf
    f_h = f * hpf
    X_l = torch.abs(fft.ifftn(f_l, dim=(2, 3)))
    X_h = torch.abs(fft.ifftn(f_h, dim=(2, 3)))
    return X_l, X_h


def gaussian_batch(dims):
    return torch.randn(tuple(dims)).to(device)


def My_loss_backward(x, y, model, label):
    x_samples = netG(x=y, rev=True)
    x_samples_image = x_samples[:, :3, :, :]

    l_back_rec = Reconstruction_back(x, x_samples_image)
    l_back_SSIM = Rec_back_SSIM(x, x_samples_image).mean()

    logit = model(x_samples_image)
    logit = torch.sigmoid(logit)
    cla_err = crossLoss(logit, label)

    total_loss = 2*l_back_rec + l_back_SSIM - cla_err
    return total_loss, l_back_rec, l_back_SSIM, cla_err


def train(epochs, dataloader, loader_test):
    low_F = list()
    rec_loss = list()
    loss1 = list()
    loss2 = list()
    loss3 = list()
    loss4 = list()
    for epoch in range(1, epochs + 1):
        for data, label in dataloader:
            data = data.cuda()
            label = label.cuda()


            optimizer_G.zero_grad()
            output = netG(data)
            LR, _ = pinglvyu(data)
            LR = netG.low_conv(LR)

            l_forw_fit = Reconstruction_back(output[:, :3, :, :], LR)

            y_ = torch.cat((output[:, :3, :, :], gaussian_batch(output[:, 3:, :, :].shape)), dim=1)

            total_loss, l_back_rec, l_back_SSIM, cla_err = My_loss_backward(data, y_, classifier, label)

            #total loss
            loss = l_forw_fit + total_loss
            loss.backward()

            nn.utils.clip_grad_norm_(netG.parameters(), 10)
            optimizer_G.step()

            low_F.append(l_forw_fit.cpu().item())
            rec_loss.append(l_back_rec.cpu().item())
            # loss1.append(l_grad_back_rec.cpu().item())
            # loss2.append(l_back_SSIM.cpu().item())
            # loss3.append(pertual_loss.cpu().item())
            loss4.append(cla_err.cpu().item())

        scheduler.step()
        print("epoch {}'s: low_F:{:.5f}, rec_loss:{:.5f}, loss1:{:.5f}, loss2:{:.5f}, loss3:{:.5f}, loss4:{:.5f}".format(epoch, np.sum(low_F) / len(low_F), np.sum(rec_loss) / len(rec_loss), \
                                                                 np.sum(loss1) / len(loss1), np.sum(loss2) / len(loss2),\
                                                                 np.sum(loss3) / len(loss3), np.sum(loss4) / len(loss4)))


        if epoch % 2 == 0:
            torch.save(netG.state_dict(), '{}/params_{}.pt'.format(model_dir, epoch))
        torch.save(netG.state_dict(), '{}/params_pause.pt'.format(model_dir))
        print("Current epoch saved!")

        if epoch % 1 == 0:
            test(epoch, netG, loader_test)

def show_images(x, x_recon):
    fig, axes = plt.subplots(2, 5, figsize=(10, 6))
    for i in range(5):
        axes[0, i].axis("off"), axes[1,i].axis("off")
        axes[0, i].imshow(x[i].cpu().numpy().transpose((1,2,0)))
        axes[0, i].set_title("Clean")

        axes[1, i].imshow(x_recon[i].cpu().numpy().transpose((1,2,0)))
        axes[1, i].set_title("Recon")
    plt.axis("off")
    plt.savefig('./test_2.png')
    print('picture already saved!')

def test(epoch, netG, loader_test):
    netG.eval()
    total = 0
    correct = 0

    for i, (images, labels) in enumerate(loader_test):
        inputs = images.cuda()
        y = labels.cuda()
        with torch.no_grad():
            final_pred = netG(inputs)
            y_forw = torch.cat((final_pred[:, :3, :, :], gaussian_batch(final_pred[:, 3:, :, :].shape)),dim=1)
            fake_H = netG(x=y_forw, rev=True)[:, :3, :, :]
            # show_images(inputs.data, fake_H.data)
            logit = classifier(fake_H)
            prediction = torch.max(logit, 1)[1]
            correct = correct + torch.eq(prediction, y).float().sum().item()

            total = total + inputs.size(0)
    accuracy = correct / total
    print()
    print('TEST *TOP* ACC:{:.4f} at e:{:03d}'.format(accuracy, epoch))
    print()

if __name__ == '__main__':

    model_dir = './attack_result_two'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    epochs = 100
    # test(1, netG, testloader)
    train(epochs, dataloader, testloader)