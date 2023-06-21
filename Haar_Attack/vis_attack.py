import os,sys
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
sys.path.append(os.pardir)
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
from HaarWavlet.adversarial_one import add_adv
from attack_model import InvNet
# from Fu_final_model import Dehaze
# from purifier_network import Dehaze

device = torch.device('cuda')

# torch.set_num_threads(3)
from torchvision.models import resnet18

# classifier = resnet18(pretrained=True)
# classifier.fc = torch.nn.Linear(512,10)
# classifier = classifier.cuda()
# classifier.eval()
classifier = ResNet18()
classifier.load_state_dict(torch.load('/remote-home/cs_igps_yangjin/DC-VAE-main/models/Result/ResNet18/params_finished.pt'))
classifier = classifier.cuda()



transform = transforms.Compose([transforms.ToTensor()])
test_data = torchvision.datasets.CIFAR10(root='../cifar10/test_data', train=False, transform=transform, download=False)
testloader = torch.utils.data.DataLoader(dataset=test_data, batch_size=32, shuffle=True)


block_num = [4]
netG = InvNet(block_num=block_num)
# netG.load_state_dict(torch.load('/remote-home/cs_igps_yangjin/MyVae3/Haar_Attack/attack_result_one/params_pause.pt'))
netG.load_state_dict(torch.load('/remote-home/cs_igps_yangjin/MyVae3/Haar_Attack/attack_semantic_sep2/params_8.pt'))
# netG.load_state_dict(torch.load('/remote-home/cs_igps_yangjin/MyVae3/Haar_Attack/attack_result/params_pause.pt'))
# netG.load_state_dict(torch.load('/remote-home/cs_igps_yangjin/MyVae3/Haar_Attack/attack_result_two/params_pause.pt'))
netG = netG.cuda()



def test_sample_acc():
    step = 0
    correct = 0
    total = 0
    for batch_idx, (data, label) in enumerate(testloader):
        step += 1
        data = data.cuda()
        label = label.cuda()

        logit = classifier(data)
        prediction = torch.max(logit, 1)[1]

        correct = correct + torch.eq(prediction, label).float().sum().item()
        total = total + data.size(0)


    accuracy = correct / total
    print(total)
    print('Classifier_ACC: ', accuracy)


def show_images(x, x_recon, noise):
    fig, axes = plt.subplots(3, 5, figsize=(10, 6))
    for i in range(5):
        axes[0, i].axis("off"), axes[1,i].axis("off"), axes[2,i].axis("off")
        axes[0, i].imshow(x[i].cpu().numpy().transpose((1,2,0)))
        axes[0, i].set_title("Clean")

        axes[1, i].imshow(x_recon[i].cpu().numpy().transpose((1,2,0)))
        axes[1, i].set_title("Adv")

        axes[2, i].imshow(noise[i].cpu().numpy().transpose((1, 2, 0)))
        axes[2, i].set_title("Noise")
    plt.axis("off")

    plt.savefig('./figs/attack_5.png')
    print('picture already saved!')

def gaussian_batch(dims):
    return torch.randn(tuple(dims)).to(device)

def test_advsample_acc():
    netG.eval()

    correct = 0
    num = 0
    total = len(test_data)
    print('total ', total)
    for image, label in testloader:
        image = image.cuda()
        label = label.cuda()

        final_pred = netG(image)
        y_forw = torch.cat((final_pred[:, 2:, :, :], gaussian_batch(final_pred[:, :2, :, :].shape)), dim=1)
        fake_H = netG(x=y_forw, rev=True)
        adv_out_class, _ = classifier(fake_H)
        noise = image - fake_H

        show_images(image.data, fake_H.data, noise.data)

        prediction = torch.max(adv_out_class, 1)[1]

        correct = correct + torch.eq(prediction, label).float().sum().item()
        num = num + image.size(0)

        # calculate number of correct classification

        print(int(correct) / num)

    accuracy = correct / num
    print(num)
    print('Classifier_ACC: ', accuracy)

    # with open(f'./accuracy.txt', 'w') as f:
    #     json.dump(adv_accuracy, f)

if __name__ == '__main__':
    test_advsample_acc()
    # test_sample_acc()