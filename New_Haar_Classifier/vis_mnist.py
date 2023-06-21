import os,sys
sys.path.append(os.pardir)
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import torch
import torch.utils.data as Data
import torchvision  # 数据库
from matplotlib import pyplot as plt
from HaarWavlet.MNIST.mnist_classifier import Classifier
from torchvision import transforms
from HaarWavlet.adversarial_one import add_adv
from MNIST_1.new_haar_model import InvNet
import numpy as np
import math
from skimage.metrics import peak_signal_noise_ratio as psnr
import time

device = torch.device('cuda')



classifier = Classifier(28, 1)
classifier.load_state_dict(torch.load('/remote-home/cs_igps_yangjin/MyVae3/HaarWavlet/MNIST/classifier_mnist.pt'))
classifier = classifier.cuda()

transform = transforms.Compose([transforms.ToTensor()])
test_data = torchvision.datasets.MNIST(root='/remote-home/cs_igps_yangjin/MyVae3/HaarWavlet/MNIST/MNIST_Test_Data', train=False, transform=transform, download=False)
testloader = torch.utils.data.DataLoader(dataset=test_data, batch_size=128, shuffle=True)

block_num = [3]
netG = InvNet(block_num=block_num)
# netG.load_state_dict(torch.load('/remote-home/cs_igps_yangjin/MyVae3/New_Haar_Classifier/New_haar_cla/params_60.pt')) #40
# netG.load_state_dict(torch.load('/remote-home/cs_igps_yangjin/MyVae3/New_Haar_Classifier/New_haar_cla_noBN_cleanFeature/params_34.pt')) #cla_2 24.pt 98.49 28.pt 98.64
netG.load_state_dict(torch.load('/remote-home/cs_igps_yangjin/MyVae3/New_Haar_Classifier/MNIST_1/New_haar_cla_noBN_Feature_extractor_1_channel/params_56.pt')) #30.pt 98.4 38.pt 98.44 50.pt 98.47 52.pt 98.49
netG = netG.cuda()

def test_sample_acc_One():
    step = 0
    correct = 0
    total = 0
    netG.eval()
    for batch_idx, (data, label) in enumerate(testloader):
        step += 1
        data = data.cuda()
        label = label.cuda()

        final_pred = netG(data)
        y_forw = torch.cat((final_pred[:, :1, :, :], gaussian_batch(final_pred[:, 1:, :, :].shape)), dim=1)
        fake_H = netG(x=y_forw, rev=True)

        logit = classifier(fake_H)
        prediction = torch.max(logit, 1)[1]

        correct = correct + torch.eq(prediction, label).float().sum().item()
        total = total + data.size(0)


    accuracy = correct / total
    print(total)
    print('Classifier_ACC: ', accuracy)

def test_sample_acc():
    step = 0
    correct = 0
    total = 0
    adv_accuracy = {'fgsm': 0}
    for adv in adv_accuracy:
        for batch_idx, (data, label) in enumerate(testloader):
            step += 1
            data = data.cuda()
            label = label.cuda()

            output, adv_out = add_adv(classifier, data, label, adv, default=False)

            logit = classifier(output)
            prediction = torch.max(logit, 1)[1]

            correct = correct + torch.eq(prediction, label).float().sum().item()
            total = total + data.size(0)

            print(correct/total)
        accuracy = correct / total
        print(total)
        print('Classifier_ACC: ', accuracy)


def show_images(x, x_recon):
    fig, axes = plt.subplots(2, 5, figsize=(10, 6))
    for i in range(5):
        axes[0, i].axis("off"), axes[1,i].axis("off")
        axes[0, i].imshow(x[i].cpu().numpy().transpose((1,2,0)))
        axes[0, i].set_title("Clean")

        axes[1, i].imshow(x_recon[i].cpu().numpy().transpose((1,2,0)))
        axes[1, i].set_title("Recon")
    plt.axis("off")
    plt.savefig('./wav_mni_adv.png')
    print('picture already saved!')

def show_images_one(x, x_adv,x_recon):
    fig, axes = plt.subplots(3, 5, figsize=(10, 6))
    for i in range(5):
        axes[0, i].axis("off"), axes[1,i].axis("off"), axes[2,i].axis("off")
        axes[0, i].imshow(x[i].cpu().numpy().transpose((1,2,0)), cmap=plt.get_cmap('gray'))
        axes[0, i].set_title("Clean")

        axes[1, i].imshow(x_adv[i].cpu().numpy().transpose((1,2,0)),cmap=plt.get_cmap('gray'))
        axes[1, i].set_title("Adv")

        axes[2, i].imshow(x_recon[i].cpu().numpy().transpose((1,2,0)),cmap=plt.get_cmap('gray'))
        axes[2, i].set_title("Recon")
    plt.axis("off")
    plt.savefig('./new1.png')
    print('picture already saved!')


def show_images_two(x_adv,x_recon):
    fig, axes = plt.subplots(2, 8, figsize=(10, 6))
    for i in range(8):
        axes[0, i].axis("off"), axes[1,i].axis("off")
        axes[0, i].imshow(x_adv[i].cpu().numpy().transpose((1,2,0)), cmap=plt.get_cmap('gray'))
        axes[0, i].set_title("Adv")

        axes[1, i].imshow(x_recon[i].cpu().numpy().transpose((1,2,0)),cmap=plt.get_cmap('gray'))
        axes[1, i].set_title("Recon")

    plt.axis("off")
    plt.savefig('./new4.png')
    print('picture already saved!')

def show_images_three(clean, x_adv, noise, x_recon, repair):
    fig, axes = plt.subplots(5, 6, figsize=(10, 6))
    for i in range(6):
        axes[0, i].axis("off"), axes[1,i].axis("off"), axes[2,i].axis("off"), axes[3,i].axis("off"), axes[4,i].axis("off")
        axes[0, i].imshow(clean[i].cpu().numpy().transpose((1,2,0)), cmap=plt.get_cmap('gray'))
        # axes[0, i].set_title("Adv")

        axes[1, i].imshow(x_adv[i].cpu().numpy().transpose((1,2,0)),cmap=plt.get_cmap('gray'))
        # axes[1, i].set_title("Recon")

        axes[2, i].imshow(noise[i].cpu().numpy().transpose((1, 2, 0)), cmap=plt.get_cmap('gray'))
        # axes[1, i].set_title("Recon")

        axes[3, i].imshow(x_recon[i].cpu().numpy().transpose((1, 2, 0)), cmap=plt.get_cmap('gray'))
        # axes[1, i].set_title("Recon")

        axes[4, i].imshow(repair[i].cpu().numpy().transpose((1, 2, 0)), cmap=plt.get_cmap('gray'))
        # axes[1, i].set_title("Recon")

    plt.axis("off")
    plt.savefig('./new3.png')
    print('picture already saved!')

def gaussian_batch(dims):
    return torch.randn(tuple(dims)).to(device)

def psnr1(img1, img2):
    # compute mse
    # mse = np.mean((img1-img2)**2)
    img1 = img1.cpu().numpy()
    img2 = img2.cpu().numpy()
    mse = np.mean((img1 / 1.0 - img2 / 1.0) ** 2)
    # compute psnr
    if mse < 1e-10:
        return 100
    psnr1 = 20 * math.log10(255 / math.sqrt(mse))
    return psnr1


def torchPSNR(prd_img, tar_img):
    if not isinstance(prd_img, torch.Tensor):
        prd_img = torch.from_numpy(prd_img)
        tar_img = torch.from_numpy(tar_img)

    imdff = torch.clamp(prd_img, 0, 1) - torch.clamp(tar_img, 0, 1)
    rmse = (imdff**2).mean().sqrt()
    ps = 20 * torch.log10(1/rmse)
    return ps



def test_advsample_acc():
    # adv_accuracy = {'pgd_n': 0, 'pgd_t': 0}
    # adv_accuracy = {'pgd_n': 0}
    # adv_accuracy = {'pgd_t': 0}
    # adv_accuracy = {'ddn_n': 0, 'jsma_t': 0, 'cw_n': 0}
    # adv_accuracy = {'jsma_t': 0}
    #adv_accuracy = {'aa_n': 0}
    # adv_accuracy = {'fgsm': 0}
    adv_accuracy = {'cw_n': 0}
    # adv_accuracy = {'ddn_n': 0}
    netG.eval()
    psnr_sum = 0
    tik = 0
    for adv in adv_accuracy:
        true = 0
        sum = 0
        correct = 0
        print('adv is ', adv)
        total = len(test_data)
        print('total ', total)
        for image, label in testloader:
            image = image.cuda()
            label = label.cuda()
            tik = tik + 1
            # get model output
            output, adv_out = add_adv(classifier, image, label, adv, default=False)
            # output_class = classifier(output)

            # start_time = time.time()
            final_pred = netG(adv_out)
            y_forw = torch.cat((final_pred[:, :1, :, :], gaussian_batch(final_pred[:, 1:, :, :].shape)), dim=1)
            # fake_H = netG(x=y_forw, rev=True)[:, :1, :, :]
            fake_H = netG(x=y_forw, rev=True)
            # end_time = time.time()
            # print("time", end_time-start_time)

            psnr_result = psnr(image.data.cpu().numpy(), fake_H.data.cpu().numpy())
            # psnr_result = torchPSNR(fake_H, image)
            print('psnr_result', psnr_result)
            psnr_sum += psnr_result

            # x_l, x_h = pinglvyu(fake_H)
        #     adv_out_class = classifier(fake_H)
        #
        #     #additional
        #     prediction = torch.max(adv_out_class, 1)[1]
        #     # adversarial_class = torch.argmax(adv_out_class, 1)
        #     correct = correct + torch.eq(prediction, label).float().sum().item()
        #     sum = sum + image.size(0)
        #     # print('sum', sum)
        # accuracy = correct / sum
        # print('Classifier_ACC: ', accuracy)
        print('psnr_avg: ', psnr_sum / tik)

    #         # show_images_one(image.data, adv_out.data, fake_H.data)
    #         # show_images_two(adv_out.data, fake_H.data)
    #         # noise = adv_out - image
    #         # repair = fake_H - image
    #         # show_images_three(image.data, adv_out.data, noise.data, fake_H.data, repair.data)
    #
    #         # get model predicted class
    #         true_class = torch.argmax(output_class, 1)
    #         adversarial_class = torch.argmax(adv_out_class, 1)
    #
    #         # calculate number of correct classification
    #         true += torch.sum(torch.eq(true_class, adversarial_class))
    #
    #         print(int(true) / total)
    #     adv_accuracy[adv] = int(true) / total
    #     print('total: ', total)
    #     print('int(true): ', int(true))
    #     print(int(true) / total)
    #     print('=================================')
    # print()
    #
    # with open(f'./accuracy.txt', 'w') as f:
    #     json.dump(adv_accuracy, f)

if __name__ == '__main__':
    # test_sample_acc_One()
    test_advsample_acc()