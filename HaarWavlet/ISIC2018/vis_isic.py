import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import torch.nn as nn
import torch.utils.data as Data
# import torchvision  # 数据库
from matplotlib import pyplot as plt
from torchvision.models import resnet50
# from torchvision import transforms
from adversarial_three import add_adv
# from isic_model import Final_FFC_Model_Three
from ISICmodel import InvNet
from train_clasifier_resnet1 import valset

device = torch.device('cuda')



classifier = resnet50()
classifier.fc = nn.Linear(2048, 7)
classifier.load_state_dict(torch.load('/remote-home/cs_igps_yangjin/MyVae2/FFC_Model/ISIC2018/isic_resnet/params_186.pt'))
classifier = classifier.cuda()


test_loader = torch.utils.data.DataLoader(valset, batch_size=20, shuffle=False, num_workers=4)


block_num = [4, 4]
netG = InvNet(block_num=block_num)
netG.load_state_dict(torch.load('/remote-home/cs_igps_yangjin/MyVae3/HaarWavlet/ISIC2018/ISIC_twoskip_trans_2e-3_0.08_colour/params_88.pt'))#28 88
netG = netG.cuda()



# def test_sample_acc():
#     step = 0
#     correct = 0
#     total = 0
#     for batch_idx, (data, label) in enumerate(testloader):
#         step += 1
#         data = data.cuda()
#         label = label.cuda()
#
#         logit = classifier(data)
#         prediction = torch.max(logit, 1)[1]
#
#         correct = correct + torch.eq(prediction, label).float().sum().item()
#         total = total + data.size(0)
#
#
#     accuracy = correct / total
#     print(total)
#     print('Classifier_ACC: ', accuracy)


def show_images(x, x_adv,x_recon):
    # fig, axes = plt.subplots(3, 8, figsize=(22, 12))
    fig, axes = plt.subplots(3, 4, constrained_layout=True)
    for i in range(4):
        axes[0, i].axis("off"), axes[1,i].axis("off"),axes[2,i].axis("off")
        axes[0, i].imshow(x[i].cpu().numpy().transpose((1,2,0)))
        axes[0, i].set_title("Clean")

        axes[1, i].imshow(x_adv[i].cpu().numpy().transpose((1, 2, 0)))
        axes[1, i].set_title("Adv")

        axes[2, i].imshow(x_recon[i].cpu().numpy().transpose((1,2,0)))
        axes[2, i].set_title("Recon")
    plt.axis("off")
    plt.savefig('./new11.png')
    print('picture already saved!')

def gaussian_batch(dims):
    return torch.randn(tuple(dims)).to(device)


def test_advsample_acc():
    # adv_accuracy = {'pgd_n': 0, 'pgd_t': 0}
    adv_accuracy = {'pgd_n': 0}
    # adv_accuracy = {'pgd_t': 0}
    # adv_accuracy = {'ddn_n': 0, 'jsma_t': 0, 'cw_n': 0}
    # adv_accuracy = {'jsma_t': 0}
    # adv_accuracy = {'aa_n': 0}
    # adv_accuracy = {'fgsm': 0}
    # adv_accuracy = {'bim': 0}
    # adv_accuracy = {'cw_n': 0}
    # adv_accuracy = {'ddn_n': 0}
    netG.eval()
    for adv in adv_accuracy:
        true = 0
        correct = 0
        total = len(test_loader)
        print('total ', total)
        # start = time.time()
        for image, label in test_loader:
            image = image.cuda()
            label = label.cuda()

            # get model output
            output, adv_out = add_adv(classifier, image, label, adv, default=False)
            # output_class = classifier(output)
            # adv_out_class = classifier(adv_out)

            final_pred = netG(adv_out)
            y_forw = torch.cat((final_pred[:, :3, :, :], gaussian_batch(final_pred[:, 3:, :, :].shape)), dim=1)
            fake_H = netG(x=y_forw, rev=True)
            adv_out_class = classifier(fake_H)

            show_images(image.data, adv_out.data, fake_H.data)
            # show_images_two(adv_out.data, fake_H.data)

            # get model predicted class
    #         true_class = torch.argmax(output_class, 1)
    #         adversarial_class = torch.argmax(adv_out_class, 1)
    #
    #         # calculate number of correct classification
    #         true += torch.sum(torch.eq(true_class, adversarial_class))
    #
    #         print(int(true) / total)
    #     # end = time.time()
    #     # print('Time:', end-start)
    #     adv_accuracy[adv] = int(true) / total
    #     print('total: ', total)
    #     print('int(true): ', int(true))
    #     print(int(true) / total)
    #     print('=================================')
    # print()
            prediction = torch.max(adv_out_class, 1)[1]

            correct = correct + torch.eq(prediction, label).float().sum().item()
            true = true + image.size(0)

            print(correct / true)
        accuracy = correct / true
        print(total)
        print('Classifier_ACC: ', accuracy)


if __name__ == '__main__':
    test_advsample_acc()