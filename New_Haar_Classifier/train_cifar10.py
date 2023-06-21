import os, sys
sys.path.append(os.pardir)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch,torchvision
import torch.nn as nn
import numpy as np
from cifar_model import InvNet
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR
import torch.fft as fft
from HaarWavlet.resnet_model import ResNet18
from torch.utils import data
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from HaarWavlet.Inv_VGG_loss import VGGLoss
import time
device = torch.device('cuda')

vggloss = VGGLoss(3,1, False).cuda()

classifier = ResNet18()
classifier.load_state_dict(torch.load('/remote-home/cs_igps_yangjin/DC-VAE-main/models/Result/ResNet18/params_finished.pt'))
classifier = classifier.cuda()


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

data = np.load('/remote-home/cs_igps_yangjin/cifar10_data/xs_cifar10.npy')  # image data in npy file
labels = np.load('/remote-home/cs_igps_yangjin/cifar10_data/ys_cifar10.npy')  # labels data in npy file
adv_data = np.load('/remote-home/cs_igps_yangjin/cifar10_data/advs_cifar10.npy')  # adversarial image data in npy file
dataset = Dataset(data, labels, adv_data)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=6)

transform = transforms.Compose([transforms.ToTensor()])
test_data = torchvision.datasets.CIFAR10(root='../cifar10/test_data', train=False, transform=transform, download=False)
testloader = torch.utils.data.DataLoader(dataset=test_data, batch_size=32, shuffle=True)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Block') == -1:
        m.weight.data.normal_(0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1, 0.02)
        m.bias.data.fill_(0)


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


optimizer_G = torch.optim.Adam(netG.parameters(), lr=2e-3, weight_decay=1e-8, betas=(0.9, 0.999))
scheduler = MinExponentialLR(optimizer_G, gamma=0.998, minimum=1e-5)


class Gradient_Loss(nn.Module):
    def __init__(self, losstype='l2'):
        super(Gradient_Loss, self).__init__()
        a = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False, groups=3)
        a = torch.from_numpy(a).float().unsqueeze(0)
        a = torch.stack((a, a, a))
        conv1.weight = nn.Parameter(a, requires_grad=False)
        self.conv1 = conv1.cuda()

        b = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        conv2 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False, groups=3)
        b = torch.from_numpy(b).float().unsqueeze(0)
        b = torch.stack((b, b, b))
        conv2.weight = nn.Parameter(b, requires_grad=False)
        self.conv2 = conv2.cuda()

        # self.Loss_criterion = ReconstructionLoss(losstype)
        self.Loss_criterion = nn.L1Loss()

    def forward(self, x, y):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        # x_total = torch.sqrt(torch.pow(x1, 2) + torch.pow(x2, 2))

        y1 = self.conv1(y)
        y2 = self.conv2(y)
        # y_total = torch.sqrt(torch.pow(y1, 2) + torch.pow(y2, 2))

        l_h = self.Loss_criterion(x1, y1)
        l_v = self.Loss_criterion(x2, y2)
        # l_total = self.Loss_criterion(x_total, y_total)
        return l_h + l_v #+ l_total

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


Rec_back_grad = Gradient_Loss()
Rec_back_SSIM = SSIM_Loss()


Reconstruction_back = nn.MSELoss()
Reconstruction_back.cuda()
cel = nn.CrossEntropyLoss().cuda()

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


def My_loss_backward(x, y):
    x_samples = netG(x=y, rev=True)
    x_samples_image = x_samples

    real = vggloss(x)
    fake = vggloss(x_samples_image)
    pertual_loss = Reconstruction_back(fake, real)

    l_back_rec = Reconstruction_back(x, x_samples_image)
    l_grad_back_rec = 0.1 * Rec_back_grad(x, x_samples_image)
    l_back_SSIM = Rec_back_SSIM(x, x_samples_image).mean()
    # return l_back_rec + l_grad_back_rec + l_back_SSIM
    total_loss = l_back_rec + l_grad_back_rec + l_back_SSIM + pertual_loss
    return total_loss, l_back_rec, l_grad_back_rec, l_back_SSIM, pertual_loss


def train(epochs, dataloader, loader_test):
    low_F = list()
    rec_loss = list()
    loss1 = list()
    loss2 = list()
    loss3 = list()
    for epoch in range(1, epochs + 1):
        # start = time.time()
        for data, label, adv_data in dataloader:
            data = data.cuda()
            label = label.cuda()
            adv_data = adv_data.cuda()

            optimizer_G.zero_grad()
            output = netG(adv_data)
            LR, _ = pinglvyu(data)
            LR = netG.low_conv(LR)
            logit = netG.low_cla(output[:,:3,:,:])
            cla_loss = cel(logit, label)

            l_forw_fit = Reconstruction_back(output[:, :3, :, :], LR)

            y_ = torch.cat((output[:, :3, :, :], gaussian_batch(output[:, 3:, :, :].shape)), dim=1)

            total_loss, l_back_rec, l_grad_back_rec, l_back_SSIM, pertual_loss = My_loss_backward(data, y_)

            #total loss
            loss = l_forw_fit + total_loss + cla_loss
            loss.backward()

            nn.utils.clip_grad_norm_(netG.parameters(), 10)
            optimizer_G.step()

            low_F.append(l_forw_fit.cpu().item())
            rec_loss.append(l_back_rec.cpu().item())
            loss1.append(l_grad_back_rec.cpu().item())
            loss2.append(l_back_SSIM.cpu().item())
            loss3.append(pertual_loss.cpu().item())

        # end = time.time()
        # print('Time:', end-start)
        scheduler.step()
        print("epoch {}'s: low_F:{:.5f}, rec_loss:{:.5f}, loss1:{:.5f}, loss2:{:.5f}, loss3:{:.5f}".format(epoch, np.sum(low_F) / len(low_F), np.sum(rec_loss) / len(rec_loss), \
                                                                 np.sum(loss1) / len(loss1), np.sum(loss2) / len(loss2),\
                                                                 np.sum(loss3) / len(loss3)))


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
            fake_H = netG(x=y_forw, rev=True)
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

    model_dir = './cifar_new_haar_claloss_2e-3_bn_3channel'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    epochs = 100
    train(epochs, dataloader, testloader)