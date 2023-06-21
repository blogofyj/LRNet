import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
sys.path.append(os.pardir)
import torch,json
import torch.nn as nn
import torch.utils.data as Data
import torchvision  # 数据库
import torch.nn.functional as F
from matplotlib import pyplot as plt
import numpy as np
from classifier_M import Classifier
# from black_box_model import model_e
import umap
# from torchvision import datasets
from torchvision import transforms
# from HaarWavlet.adversarial_one import add_adv
from new_haar_model import InvNet


device = torch.device('cuda')



classifier = Classifier(28, 1)
classifier.load_state_dict(torch.load('/remote-home/cs_igps_yangjin/MyVae3/HaarWavlet/MNIST/classifier_mnist.pt'))
classifier = classifier.cuda()

transform = transforms.Compose([transforms.ToTensor()])
test_data = torchvision.datasets.MNIST(root='/remote-home/cs_igps_yangjin/MyVae3/HaarWavlet/MNIST/MNIST_Test_Data', train=False, transform=transform, download=False)
testloader = torch.utils.data.DataLoader(dataset=test_data, batch_size=128, shuffle=True)

block_num = [3]
model = InvNet(block_num=block_num)
model.load_state_dict(torch.load('/remote-home/cs_igps_yangjin/MyVae3/New_Haar_Classifier/MNIST_1/New_haar_cla_noBN_Feature_extractor_1_channel/params_56.pt')) #30.pt 98.4 38.pt 98.44 50.pt 98.47 52.pt 98.49
model = model.cuda()

def show_images(x, x_adv,x_recon):
    fig, axes = plt.subplots(3, 5, figsize=(10, 6))
    for i in range(5):
        axes[0, i].axis("off"), axes[1,i].axis("off")
        axes[0, i].imshow(x[i].cpu().numpy().transpose((1,2,0)))
        axes[0, i].set_title("Clean")

        axes[1, i].imshow(x_adv[i].cpu().numpy().transpose((1,2,0)))
        axes[1, i].set_title("Adv")

        axes[2, i].imshow(x_recon[i].cpu().numpy().transpose((1,2,0)))
        axes[2, i].set_title("Recon")
    plt.axis("off")
    plt.savefig('./new1.png')
    print('picture already saved!')

def gaussian_batch(dims):
    return torch.randn(tuple(dims)).to(device)

def test_sample_acc():
    # adv_accuracy = {'pgd_n': 0}
    # adv_accuracy = {'i-fgsm': 0}
    adv_accuracy = {'fgsm': 0}

    for adv in adv_accuracy:
        true = 0
        total = len(test_data)
        print('total ', total)
        correct = 0
        num = 0
        print('adv', adv)
        for image, label in testloader:
            image = image.cuda()
            label = label.cuda()

            # get model output
            # output, adv_out = add_adv(classifier, image, label, adv, default=False)
            #
            _, final_out = model(image)
            adv_out_class = classifier(final_out)

            prediction = torch.max(adv_out_class, 1)[1]
            correct = correct + torch.eq(prediction, label).float().sum().item()
            num = num + image.size(0)

        accuracy = correct / total
        print(total)
        print('Classifier_ACC: ', accuracy)

def test_advsample_acc():
    # adv_accuracy = {'pgd_n': 0, 'pgd_t': 0}
    # adv_accuracy = {'pgd_n': 0}
    # adv_accuracy = {'ddn_n': 0}
    # adv_accuracy = {'cw_n': 0}
    # adv_accuracy = {'pgd_t': 0}
    # adv_accuracy = {'ddn_n': 0, 'jsma_t': 0, 'cw_n': 0}
    # adv_accuracy = {'jsma_t': 0}
    # adv_accuracy = {'aa_n': 0}
    adv_accuracy = {'fgsm': 0}
    # adv_accuracy = {'i-fgsm': 0}

    for adv in adv_accuracy:
        adv_examples = np.load('/remote-home/cs_igps_yangjin/MyVae3/cifar10/mnistadvdata_cw/advs_mnist.npy')
        adv_examples = adv_examples.reshape(adv_examples.shape[0], 28 * 28)
        sample = np.random.randint(adv_examples.shape[0], size=3000)
        adv_examples = adv_examples[sample, :]

        adv_images = torch.from_numpy(adv_examples.reshape(adv_examples.shape[0], 1, 28, 28))
        adv_images = adv_images.cuda()

        #with defense
        # final_pred = model(adv_images)
        # y_forw = torch.cat((final_pred[:, :1, :, :], gaussian_batch(final_pred[:, 1:, :, :].shape)), dim=1)
        # fake_H = model(x=y_forw, rev=True)
        # labels = classifier(fake_H)

        labels = classifier(adv_images)
        labels = torch.argmax(labels, 1).detach().cpu().numpy()

        #with defense
        # fake_H = fake_H.detach().cpu().numpy()
        # fake_H = fake_H.reshape(fake_H.shape[0], 28 * 28)

        fit = umap.UMAP(n_components=2, random_state=42)
        u = fit.fit_transform(adv_examples)

        # u = fit.fit_transform(fake_H.reshape(fake_H.shape[0], 28 * 28))

        plt.scatter(u[:, 0], u[:, 1], c=labels, cmap='Spectral', s=14)
        plt.gca().set_aspect('equal', 'datalim')
        clb = plt.colorbar(boundaries=np.arange(11) - 0.5)
        clb.set_ticks(np.arange(10))
        clb.ax.tick_params(labelsize=18)
        plt.xticks([])
        plt.yticks([])
        # plt.title(f'MNIST clustering under {adv.upper()}', fontsize=24);
        # plt.title(f'MNIST(With Defense)', fontsize=16);
        plt.title(f'MNIST(Under CW Attack)', fontsize=16);
        if not os.path.exists('./img'):
            os.makedirs('./img')
        plt.savefig(f'img/MNIST_undefense_cw.png', dpi=300, pad_inches=0)
        plt.clf()


def vis_defense():
    model.eval()
    adv_examples = np.load('/remote-home/cs_igps_yangjin/MyVae3/cifar10/mnistadvdata_cw/advs_mnist.npy')
    adv_examples = adv_examples.reshape(adv_examples.shape[0], 28 * 28)
    sample = np.random.randint(adv_examples.shape[0], size=3000)
    adv_examples = adv_examples[sample, :]

    adv_images = torch.from_numpy(adv_examples.reshape(adv_examples.shape[0], 1, 28, 28))
    adv_images = adv_images.cuda()

    final_pred = model(adv_images)
    y_forw = torch.cat((final_pred[:, :1, :, :], gaussian_batch(final_pred[:, 1:, :, :].shape)), dim=1)
    fake_H = model(x=y_forw, rev=True)

    labels = classifier(fake_H)
    labels = torch.argmax(labels, 1).detach().cpu().numpy()

    def_out = fake_H.detach().cpu().numpy()
    def_out = def_out.reshape(def_out.shape[0], 28 * 28)

    fit = umap.UMAP(n_components=2, random_state=42)
    u = fit.fit_transform(def_out.reshape(def_out.shape[0], 28 * 28))

    plt.scatter(u[:, 0], u[:, 1], c=labels, cmap='Spectral', s=14)
    plt.gca().set_aspect('equal', 'datalim')
    clb = plt.colorbar(boundaries=np.arange(11) - 0.5)
    clb.set_ticks(np.arange(10))
    clb.ax.tick_params(labelsize=18)
    plt.xticks([])
    plt.yticks([])
    plt.title(f'MNIST(CW With Defense)', fontsize=16);
    if not os.path.exists('./img'):
        os.makedirs('./img')
    plt.savefig(f'img/MNIST_underdefense_cw.png', dpi=300, pad_inches=0)
    plt.clf()



def vis_clean_cluster():
    data = np.load('/remote-home/cs_igps_yangjin/adversarial dataset/xs_mnist_v2.npy')
    data = data.reshape(data.shape[0], 28 * 28)
    sample = np.random.randint(data.shape[0], size=3000)
    data = data[sample, :]

    y_s = np.load('/remote-home/cs_igps_yangjin/adversarial dataset/ys_mnist_v2.npy')
    y_s = y_s[sample]

    fit = umap.UMAP(random_state=42, n_components=2)
    u = fit.fit_transform(data)

    plt.scatter(u[:, 0], u[:, 1], c=y_s, cmap='Spectral', s=14)
    plt.gca().set_aspect('equal', 'datalim')
    clb = plt.colorbar(boundaries=np.arange(11) - 0.5)
    clb.set_ticks(np.arange(10))
    clb.ax.tick_params(labelsize=18)
    plt.xticks([])
    plt.yticks([])
    plt.title(f'MNIST(Clean)', fontsize=16);
    if not os.path.exists('./img'):
        os.makedirs('./img')
    plt.savefig(f'img/MNIST_clean2.png', dpi=300)
    plt.clf()

if __name__ == '__main__':
    # test_sample_acc()
    # eval()
    # vis_clean_cluster()
    # test_advsample_acc()
    vis_defense()