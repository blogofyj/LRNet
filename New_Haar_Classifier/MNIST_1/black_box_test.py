import os,sys
sys.path.append(os.pardir)
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import torch
import torch.utils.data as Data
import torchvision  # 数据库
from matplotlib import pyplot as plt
# from black_box_model import *
from train_lenet import LeNet5
from train_other_model_black_box import Net
from torchvision import transforms
from adversarial_black_test import add_adv
from new_haar_model import InvNet

device = torch.device('cuda')



classifier = LeNet5(10)
classifier.load_state_dict(torch.load('/remote-home/cs_igps_yangjin/MyVae3/New_Haar_Classifier/MNIST_1/train_lenet5_params_new/params_100.pt'))
classifier = classifier.cuda()

transform = transforms.Compose([transforms.ToTensor()])
# transform = transforms.Compose([transforms.Resize((32, 32)),transforms.ToTensor()])
test_data = torchvision.datasets.MNIST(root='/remote-home/cs_igps_yangjin/MyVae3/HaarWavlet/MNIST/MNIST_Test_Data', train=False, transform=transform, download=False)
testloader = torch.utils.data.DataLoader(dataset=test_data, batch_size=128, shuffle=True)

block_num = [3]
netG = InvNet(block_num=block_num)
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

            # output, adv_out = add_adv(classifier, data, label, adv, default=False)

            logit = classifier(data)
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

            # get model output
            output, adv_out = add_adv(classifier, image, label, adv, default=False)
            output_class = classifier(output)

            final_pred = netG(adv_out)
            y_forw = torch.cat((final_pred[:, :1, :, :], gaussian_batch(final_pred[:, 1:, :, :].shape)), dim=1)
            # fake_H = netG(x=y_forw, rev=True)[:, :1, :, :]
            fake_H = netG(x=y_forw, rev=True)

            # x_l, x_h = pinglvyu(fake_H)
            adv_out_class = classifier(fake_H)

            #additional
        #     prediction = torch.max(adv_out_class, 1)[1]
        #     adversarial_class = torch.argmax(adv_out_class, 1)
        #     correct = correct + torch.eq(adversarial_class, label).float().sum().item()
        #     sum = sum + image.size(0)
        # accuracy = correct / sum
        # print('Classifier_ACC: ', accuracy)

            # show_images_one(image.data, adv_out.data, fake_H.data)
            # show_images_two(adv_out.data, fake_H.data)
            # noise = adv_out - image
            # repair = fake_H - image
            # show_images_three(image.data, adv_out.data, noise.data, fake_H.data, repair.data)

            # get model predicted class
            true_class = torch.argmax(output_class, 1)
            adversarial_class = torch.argmax(adv_out_class, 1)

            # calculate number of correct classification
            true += torch.sum(torch.eq(true_class, adversarial_class))

            print(int(true) / total)
        adv_accuracy[adv] = int(true) / total
        print('total: ', total)
        print('int(true): ', int(true))
        print(int(true) / total)
        print('=================================')
    print()
    #
    # with open(f'./accuracy.txt', 'w') as f:
    #     json.dump(adv_accuracy, f)

def no_defense():
    adv_accuracy = {'ddn_n': 0}
    netG.eval()
    for adv in adv_accuracy:
        correct = 0
        sum = 0
        print('adv is ', adv)
        total = len(test_data)
        print('total ', total)
        for image, label in testloader:
            image = image.cuda()
            label = label.cuda()

            # get model output
            output, adv_out = add_adv(classifier, image, label, adv, default=False)

            final_pred = netG(adv_out)
            y_forw = torch.cat((final_pred[:, :1, :, :], gaussian_batch(final_pred[:, 1:, :, :].shape)), dim=1)
            fake_H = netG(x=y_forw, rev=True)

            logit = classifier(fake_H)
            prediction = torch.max(logit, 1)[1]

            correct = correct + torch.eq(prediction, label).float().sum().item()
            sum = sum + image.size(0)

            print(correct / sum)
        accuracy = correct / sum
        print(sum)
        print('Classifier_ACC: ', accuracy)
        print('=================================')
    print()

if __name__ == '__main__':
    # test_sample_acc_One()
    test_sample_acc()
    # test_advsample_acc()
    # no_defense()