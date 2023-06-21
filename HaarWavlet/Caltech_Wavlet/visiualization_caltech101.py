import os, sys
sys.path.append(os.pardir)
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import torch,json
import torch.nn as nn
import torch.utils.data as Data
import torchvision  # 数据库
import torch.nn.functional as F
from matplotlib import pyplot as plt
import numpy as np
from torchvision.models import alexnet
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from cal_adversarial import add_adv
from new import test_dataset
from skimage.metrics import peak_signal_noise_ratio as psnr
# from cal_model import InvNet
# from New_Haar_Classifier.Caltech101_1.caltech101_model import InvNet
from caltech101_model import InvNet

# from purifier_network import Dehaze

device = torch.device('cuda')

# torch.set_num_threads(3)


classifier = alexnet()
classifier.classifier[6] = nn.Linear(4096, 101)
classifier.load_state_dict(torch.load('/remote-home/cs_igps_yangjin/MyMAD_VAE/Project/Caltech101/caltec_model/params_29.pt'))
classifier = classifier.cuda()
classifier.eval()




test_dataloader = DataLoader(dataset=test_dataset, batch_size=8, shuffle=True, num_workers=4, drop_last=True)


block_num = [4, 4]
# block_num = [6]
# block_num = [4]
model = InvNet(block_num=block_num)
# model.load_state_dict(torch.load('/remote-home/cs_igps_yangjin/MyVae3/HaarWavlet/Caltech_Wavlet/Cal_Wav_two/params_12.pt'))
# model.load_state_dict(torch.load('/remote-home/cs_igps_yangjin/MyVae3/HaarWavlet/Caltech_Wavlet/Cal_Wav_four/params_34.pt')) #不错 7block
# model.load_state_dict(torch.load('/remote-home/cs_igps_yangjin/MyVae3/HaarWavlet/Caltech_Wavlet/Cal_Wav_one/params_100.pt')) #不错 7block

model.load_state_dict(torch.load('/remote-home/cs_igps_yangjin/MyVae3/New_Haar_Classifier/Caltech101_1/new_haar_classifier_noBn_FeatureExtract_1/params_62.pt')) #不错 7block
model = model.cuda()
model.eval()


def test_sample_acc():
    step = 0
    correct = 0
    total = 0
    for batch_idx, (data, label) in enumerate(test_dataloader):
        step += 1
        data = data.cuda()
        label = label.cuda()

        final_pred = model(data)
        y_forw = torch.cat((final_pred[:, :3, :, :], gaussian_batch(final_pred[:, 3:, :, :].shape)), dim=1)
        fake_H = model(x=y_forw, rev=True)

        logit = classifier(fake_H)
        prediction = torch.max(logit, 1)[1]

        correct = correct + torch.eq(prediction, label).float().sum().item()
        total = total + data.size(0)


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
    plt.savefig('./cal_wav_two.png')
    print('picture already saved!')


def imgshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.cpu().numpy()
    # plt.imshow(np.transpose(npimg, (1, 2, 0)).astype(np.uint8))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    plt.savefig('./cal_11.png')
    print('image already saved!')
    return


def imgshow1(img):
    img = img / 2 + 0.5     # unnormalize
    # img = img / 255
    npimg = img.cpu().numpy()
    # plt.imshow(np.transpose(npimg, (1, 2, 0)).astype(np.uint8))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    plt.savefig('./cal_11_org.png')
    print('image already saved!')
    return

def img_shows(image, adv_image):
    img = image / 2 +0.5
    adv = adv_image / 2 +0.5
    # rec = recon_image / 2 +0.5
    fig, axes = plt.subplots(2, 4, figsize=(20, 16))
    for i in range(4):
        axes[0, i].axis("off"), axes[1, i].axis("off")
        axes[0, i].imshow(img[i].cpu().numpy().transpose((1, 2, 0)))
        axes[0, i].set_title("Clean")

        axes[1, i].imshow(adv[i].cpu().numpy().transpose((1, 2, 0)))
        axes[1, i].set_title("Recon")

        # axes[2, i].imshow(rec[i].cpu().numpy().transpose((1, 2, 0)))
        # axes[2, i].set_title("Recon")
    plt.axis("off")
    plt.savefig('./new3.png')
    print('picture already saved!')

def show_images_two(x, x_adv,x_recon):
    x = x / 2 + 0.5
    # x_adv = x_adv / 2 + 0.5
    x_recon = x_recon / 2 + 0.5
    fig, axes = plt.subplots(3, 6, constrained_layout=True)
    for i in range(6):
        axes[0, i].axis("off"), axes[1,i].axis("off"), axes[2,i].axis("off")
        axes[0, i].imshow(x[i].cpu().numpy().transpose((1,2,0)))
        axes[0, i].set_title("Clean")

        axes[1, i].imshow(x_adv[i].cpu().numpy().transpose((1,2,0)))
        axes[1, i].set_title("Adv")

        axes[2, i].imshow(x_recon[i].cpu().numpy().transpose((1,2,0)))
        axes[2, i].set_title("Recon")
    plt.axis("off")
    plt.savefig('./new10.png')
    print('picture already saved!')


def gaussian_batch(dims):
    return torch.randn(tuple(dims)).to(device)

def data_normal(orign_data):
    d_min = orign_data.min()
    if d_min < 0:
        orign_data += torch.abs(d_min)
        d_min = orign_data.min()
    d_max = orign_data.max()
    dst = d_max - d_min
    norm_data = (orign_data - d_min).true_divide(dst)
    return norm_data



def test_advsample_acc():
    adv_accuracy = {'pgd_n': 0, 'pgd_t': 0}
    # adv_accuracy = {'pgd_t': 0}
    # adv_accuracy = {'ddn_n': 0, 'jsma_t': 0, 'cw_n': 0}
    # adv_accuracy = {'jsma_t': 0}
    # adv_accuracy = {'cw_n': 0}
    # adv_accuracy = {'aa_n': 0}
    # adv_accuracy = {'fgsm': 0}
    # adv_accuracy = {'l-bfgs': 0}

    for adv in adv_accuracy:
        true = 0
        total = len(test_dataloader)
        #for i in test_dataloader:
        #    print('i', len(i))
        print('total ', total)
        correct = 0
        num = 0
        for image, label in test_dataloader:
            image = image.cuda()
            label = label.cuda()

            # get model output
            output, adv_out = add_adv(classifier, image, label, adv, default=False)
            output_class = classifier(output)

            final_pred = model(adv_out)
            y_forw = torch.cat((final_pred[:, :3, :, :], gaussian_batch(final_pred[:, 3:, :, :].shape)), dim=1)
            fake_H = model(x=y_forw, rev=True)[:, :3, :, :]
            adv_out_class = classifier(fake_H)
            # print('image', image)
            # print('image', fake_H)

            # psnr_result = psnr(data_normal(image.data).cpu().numpy(), data_normal(fake_H.data).cpu().numpy())
            # print('psnr_result', psnr_result)

            # out = torchvision.utils.make_grid(image)
            # out1 = torchvision.utils.make_grid(final_out)
            # imgshow1(out)
            # imgshow(out1)
            # final_out = model(adv_out)
            # img_shows(image.data, fake_H.data)
            show_images_two(image.data, adv_out.data, fake_H.data)

            # show_images(image.data, fake_H.data)
        #     prediction = torch.max(adv_out_class, 1)[1]
        #     correct = correct + torch.eq(prediction, label).float().sum().item()
        #     num = num + adv_out.size(0)
        # accuracy = correct / num
        # print(total)
        # print('Classifier_ACC: ', accuracy)


            # get model predicted class
            true_class = torch.argmax(output_class, 1)
            adversarial_class = torch.argmax(adv_out_class, 1)

            # print(f'attack method {adv}')
            # print(f'actual class {true_class}')
            # print(f'adversarial class {adversarial_class}')

            # calculate number of correct classification
            true += torch.sum(torch.eq(true_class, adversarial_class))

            print(int(true) / total)
        adv_accuracy[adv] = int(true) / total
        print('total: ', total)
        print('int(true): ', int(true))
        print(int(true) / total)
        print('=================================')
    print()

    # with open(f'./accuracy.txt', 'w') as f:
    #     json.dump(adv_accuracy, f)

if __name__ == '__main__':
    # test_sample_acc()
    # eval()
    test_advsample_acc()