import os,sys
sys.path.append(os.pardir)
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import torch,json
import torch.nn as nn
import torch.utils.data as Data
import torchvision  # 数据库
import torch.nn.functional as F
from matplotlib import pyplot as plt
import numpy as np
from HaarWavlet.resnet_model import ResNet18
from torchvision import datasets
from torchvision import transforms
from test_adversarial import add_adv
from cifar_model import InvNet
# from vis_mnist import psnr1, psnr2
from skimage.metrics import peak_signal_noise_ratio as psnr
# from Fu_final_model import Dehaze
# from purifier_network import Dehaze
import time

device = torch.device('cuda')

# torch.set_num_threads(3)


classifier = ResNet18()
classifier.load_state_dict(torch.load('/remote-home/cs_igps_yangjin/DC-VAE-main/models/Result/ResNet18/params_finished.pt'))
classifier = classifier.cuda()



transform = transforms.Compose([transforms.ToTensor()])
test_data = torchvision.datasets.CIFAR10(root='../cifar10/test_data', train=False, transform=transform, download=False)
testloader = torch.utils.data.DataLoader(dataset=test_data, batch_size=64, shuffle=True)


block_num = [4]
netG = InvNet(block_num=block_num)
netG.load_state_dict(torch.load('/remote-home/cs_igps_yangjin/MyVae3/New_Haar_Classifier/cifar_new_haar_claloss_2e-4_bn_3channel/params_98.pt')) #86
# netG.load_state_dict(torch.load('/remote-home/cs_igps_yangjin/MyVae3/New_Haar_Classifier/cifar_new_haar_ablation_no_feature_extract/params_8.pt')) #86
# netG.load_state_dict(torch.load('/remote-home/cs_igps_yangjin/MyVae3/New_Haar_Classifier/cifar_new_haar_ablation_no_lf/params_6.pt')) #86
netG = netG.cuda()



def test_sample_acc(): 
    step = 0
    correct = 0
    total = 0
    netG.eval()
    for batch_idx, (data, label) in enumerate(testloader):
        step += 1
        data = data.cuda()
        label = label.cuda()

        final_pred = netG(data)
        y_forw = torch.cat((final_pred[:, :3, :, :], gaussian_batch(final_pred[:, 3:, :, :].shape)), dim=1)
        fake_H = netG(x=y_forw, rev=True)

        # psnr_result = psnr(data.data.cpu().numpy(), fake_H.data.cpu().numpy())
        # print('psnr_res', psnr_result)

        logit = classifier(fake_H)
        prediction = torch.max(logit, 1)[1]

        correct = correct + torch.eq(prediction, label).float().sum().item()
        total = total + data.size(0)


    accuracy = correct / total
    print(total)
    print('Classifier_ACC: ', accuracy)


def show_images(x, x_adv,x_recon):
    fig, axes = plt.subplots(3, 8, figsize=(22, 12))
    for i in range(8):
        axes[0, i].axis("off"), axes[1,i].axis("off")
        axes[0, i].imshow(x[i].cpu().numpy().transpose((1,2,0)))
        axes[0, i].set_title("Clean")

        axes[1, i].imshow(x_adv[i].cpu().numpy().transpose((1, 2, 0)))
        axes[1, i].set_title("Adv")

        axes[2, i].imshow(x_recon[i].cpu().numpy().transpose((1,2,0)))
        axes[2, i].set_title("Recon")
    plt.axis("off")
    plt.savefig('./new2.png')
    print('picture already saved!')

def show_images_two(x_adv,x_recon):
    fig, axes = plt.subplots(2, 5, figsize=(10, 6))
    for i in range(5):
        axes[0, i].axis("off"), axes[1,i].axis("off")
        axes[0, i].imshow(x_adv[i].cpu().numpy().transpose((1,2,0)), cmap=plt.get_cmap('gray'))
        axes[0, i].set_title("Adv")

        axes[1, i].imshow(x_recon[i].cpu().numpy().transpose((1,2,0)),cmap=plt.get_cmap('gray'))
        axes[1, i].set_title("Recon")

    plt.axis("off")
    plt.savefig('./new4.png')
    print('picture already saved!')

def gaussian_batch(dims):
    return torch.randn(tuple(dims)).to(device)

def test_advsample_acc():
    # adv_accuracy = {'pgd_n': 0, 'pgd_t': 0}
    # adv_accuracy = {'pgd_n': 0}
    # adv_accuracy = {'pgd_t': 0}
    # adv_accuracy = {'ddn_n': 0, 'jsma_t': 0, 'cw_n': 0}
    # adv_accuracy = {'jsma_t': 0}
    # adv_accuracy = {'aa_n': 0}
    # adv_accuracy = {'fgsm': 0}
    adv_accuracy = {'cw_n': 0}
    # adv_accuracy = {'ddn_n': 0}
    netG.eval()
    for adv in adv_accuracy:
        true = 0
        total = len(test_data)
        print('total ', total)
        # start = time.time()
        for image, label in testloader:
            image = image.cuda()
            label = label.cuda()

            # get model output
            output, adv_out = add_adv(classifier, image, label, adv, default=False)
            output_class = classifier(output)
            # adv_out_class = classifier(adv_out)

            final_pred = netG(adv_out)
            y_forw = torch.cat((final_pred[:, :3, :, :], gaussian_batch(final_pred[:, 3:, :, :].shape)), dim=1)
            fake_H = netG(x=y_forw, rev=True)
            adv_out_class = classifier(fake_H)

            # show_images(image.data, adv_out.data, fake_H.data)
            # show_images_two(adv_out.data, fake_H.data)

            # get model predicted class
            true_class = torch.argmax(output_class, 1)
            adversarial_class = torch.argmax(adv_out_class, 1)

            # calculate number of correct classification
            true += torch.sum(torch.eq(true_class, adversarial_class))

            print(int(true) / total)
        # end = time.time()
        # print('Time:', end-start)
        adv_accuracy[adv] = int(true) / total
        print('total: ', total)
        print('int(true): ', int(true))
        print(int(true) / total)
        print('=================================')
    print()

    # with open(f'./accuracy.txt', 'w') as f:
    #     json.dump(adv_accuracy, f)


def test_advsample_acc_one():
    # adv_accuracy = {'pgd_n': 0}
    # adv_accuracy = {'pgd_t': 0}
    # adv_accuracy = {'jsma_t': 0}
    # adv_accuracy = {'aa_n': 0}
    # adv_accuracy = {'fgsm': 0}
    adv_accuracy = {'cw_n': 0}
    # adv_accuracy = {'ddn_n': 0}
    netG.eval()
    correct = 0
    sum = 0
    psnr_sum = 0
    tik = 0
    for adv in adv_accuracy:
        total = len(test_data)
        print('total ', total)
        for image, label in testloader:
            image = image.cuda()
            label = label.cuda()

            tik = tik + 1
            output, adv_out = add_adv(classifier, image, label, adv, default=False)

            # start_time = time.time()
            final_pred = netG(adv_out)
            y_forw = torch.cat((final_pred[:, :3, :, :], gaussian_batch(final_pred[:, 3:, :, :].shape)), dim=1)
            fake_H = netG(x=y_forw, rev=True)
            # end_time = time.time()
            # print('time', end_time-start_time)

            psnr_result = psnr(output.data.cpu().numpy(), fake_H.data.cpu().numpy())
            # psnr_result = psnr2(output.data, fake_H.data)
            print('psnr_res', psnr_result)
            psnr_sum += psnr_result

        #     adv_out_class = classifier(fake_H)
        #     prediction = torch.max(adv_out_class, 1)[1]
        #
        #     correct = correct + torch.eq(prediction, label).float().sum().item()
        #     sum = sum + image.size(0)
        #
        # accuracy = correct / sum
        # print(total)
        # print('Classifier_ACC: ', accuracy)
        print('PSNR: ', psnr_sum / tik)



if __name__ == '__main__':
    # test_advsample_acc()
    test_advsample_acc_one()
    # test_sample_acc()