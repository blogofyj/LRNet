import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import torch,torchvision
import torch.nn as nn
import numpy as np
from ISICmodel import InvNet
from torch.optim.lr_scheduler import ExponentialLR
from torchvision.models import resnet50
from torch.utils import data
from Inv_VGG_loss import VGGLoss
from train_clasifier_resnet1 import valset
import torch.fft as fft
device = torch.device('cuda')

vggloss = VGGLoss(3,1, False).cuda()

classifier = resnet50()
classifier.fc = nn.Linear(2048, 7)
classifier.load_state_dict(torch.load('/remote-home/cs_igps_yangjin/MyVae2/FFC_Model/ISIC2018/isic_resnet/params_186.pt'), strict=False)
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


data = np.load('/remote-home/cs_igps_yangjin/MyVae2/FFC_Model/ISIC2018/isicadvdata_0.08_colour/xs_isic.npy')
labels = np.load('/remote-home/cs_igps_yangjin/MyVae2/FFC_Model/ISIC2018/isicadvdata_0.08_colour/ys_isic.npy')
adv_data = np.load('/remote-home/cs_igps_yangjin/MyVae2/FFC_Model/ISIC2018/isicadvdata_0.08_colour/advs_isic.npy')


dataset = Dataset(data, labels, adv_data)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=6) #46

test_loader = torch.utils.data.DataLoader(valset, batch_size=32, shuffle=False, num_workers=6)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Block') == -1:
        m.weight.data.normal_(0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1, 0.02)
        m.bias.data.fill_(0)


block_num = [4, 4]
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


# optimizer_G = torch.optim.SGD(netG.parameters(), lr=2e-3, momentum=0.9,weight_decay=0.01)
# optimizer_G = torch.optim.SGD(netG.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-5)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=20, gamma=0.1)


optimizer_G = torch.optim.Adam(netG.parameters(), lr=2e-3, weight_decay=1e-8, betas=(0.9, 0.999)) #原始1e-8
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
cel = nn.CrossEntropyLoss ().cuda()


def gaussian_batch(dims):
    return torch.randn(tuple(dims)).to(device)


h, w = 224,224
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
    x_samples_image = x_samples[:, :3, :, :]

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
        for data, label, adv_data in dataloader:
            data = data.cuda()
            label = label.cuda()
            adv_data = adv_data.cuda()

            optimizer_G.zero_grad()
            output = netG(adv_data)
            LR, _ = pinglvyu(data)
            LR = netG.low_conv(LR)

            l_forw_fit = Reconstruction_back(output[:, :3, :, :], LR)

            y_ = torch.cat((output[:, :3, :, :], gaussian_batch(output[:, 3:, :, :].shape)), dim=1)

            total_loss, l_back_rec, l_grad_back_rec, l_back_SSIM, pertual_loss = My_loss_backward(data, y_)

            #total loss
            loss = l_forw_fit + total_loss
            loss.backward()

            nn.utils.clip_grad_norm_(netG.parameters(), 10)
            optimizer_G.step()
            if epoch == 10:
                optimizer_G.param_groups[0]['lr'] = 0.001
            elif epoch == 20:
                optimizer_G.param_groups[0]['lr'] = 0.0002

            low_F.append(l_forw_fit.cpu().item())
            rec_loss.append(l_back_rec.cpu().item())
            loss1.append(l_grad_back_rec.cpu().item())
            loss2.append(l_back_SSIM.cpu().item())
            loss3.append(pertual_loss.cpu().item())

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

    model_dir = './ISIC_twoskip_trans_2e-3_nolrop'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    epochs = 1000
    train(epochs, dataloader, test_loader)