import os,sys
sys.path.append(os.pardir)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import torch.utils.data as Data
import torchvision  # 数据库
from matplotlib import pyplot as plt
from mnist_classifier import Classifier
from torchvision import transforms
import torch.fft as fft
from adversarial_one import add_adv

device = torch.device('cuda')


classifier = Classifier(28, 1)
classifier.load_state_dict(torch.load('./classifier_mnist.pt'))
classifier = classifier.cuda()

transform = transforms.Compose([transforms.ToTensor()])
test_data = torchvision.datasets.MNIST(root='./MNIST_Test_Data', train=False, transform=transform, download=False)
testloader = torch.utils.data.DataLoader(dataset=test_data, batch_size=128, shuffle=True)


def frequency_domain(image_size, data):
    h, w = image_size, image_size
    lpf = torch.zeros((h, w))
    R = (h+w)//8  #或其他
    for x in range(w):
        for y in range(h):
            if ((x-(w-1)/2)**2 + (y-(h-1)/2)**2) < (R**2):
                lpf[y,x] = 1
    hpf = 1-lpf
    hpf, lpf = hpf.cuda(), lpf.cuda()

    f = fft.fftn(data, dim=(2, 3)).cuda()
    f = torch.roll(f, (h // 2, w // 2), dims=(2, 3))  # 移频操作,把低频放到中央
    f_l = f * lpf
    f_h = f * hpf
    X_l = torch.abs(fft.ifftn(f_l, dim=(2, 3)))
    X_h = torch.abs(fft.ifftn(f_h, dim=(2, 3)))
    return X_l, X_h


def test_sample_acc():
    step = 0
    correct = 0
    correct1 = 0
    total = 0
    for batch_idx, (data, label) in enumerate(testloader):
        step += 1
        data = data.cuda()
        label = label.cuda()

        x_l, x_h = frequency_domain(28, data)

        logit = classifier(x_l)
        logit1 = classifier(x_h)
        prediction = torch.max(logit, 1)[1]
        prediction1 = torch.max(logit1, 1)[1]

        correct = correct + torch.eq(prediction, label).float().sum().item()
        correct1 = correct1 + torch.eq(prediction1, label).float().sum().item()
        total = total + data.size(0)

        print(correct/total)
        print(correct1/total)
    accuracy = correct / total
    accuracy1 = correct1 / total
    print(total)
    print('Classifier_ACC_low: ', accuracy)
    print('Classifier_ACC_high: ', accuracy1)


def show_images(x, x_recon):
    fig, axes = plt.subplots(2, 5, figsize=(10, 6))
    for i in range(5):
        axes[0, i].axis("off"), axes[1,i].axis("off")
        axes[0, i].imshow(x[i].cpu().numpy().transpose((1,2,0)), cmap='gray')
        axes[0, i].set_title("Noise_L")

        axes[1, i].imshow(x_recon[i].cpu().numpy().transpose((1,2,0)), cmap='gray')
        axes[1, i].set_title("Noise_H")
    plt.axis("off")
    plt.savefig('./frequency_difference1.png')
    print('picture already saved!')

def show_images_one(x, x_adv,x_recon):
    # fig, axes = plt.subplots(3, 8, figsize=(10, 4), constrained_layout=True)
    fig, axes = plt.subplots(3, 8, constrained_layout=True)
    # plt.tight_layout()
    for i in range(8):
        axes[0, i].axis("off"), axes[1,i].axis("off"), axes[2,i].axis("off")
        axes[0, i].imshow(x[i].cpu().numpy().transpose((1,2,0)), cmap=plt.get_cmap('gray'))
        axes[0, i].set_title("Clean")

        axes[1, i].imshow(x_adv[i].cpu().numpy().transpose((1,2,0)),cmap=plt.get_cmap('gray'))
        axes[1, i].set_title("Adv")

        axes[2, i].imshow(x_recon[i].cpu().numpy().transpose((1,2,0)),cmap=plt.get_cmap('gray'))
        axes[2, i].set_title("Recon")
    plt.axis("off")
    plt.savefig('./new9.png')
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

def test_advsample_acc():
    # adv_accuracy = {'pgd_n': 0}
    # adv_accuracy = {'pgd_t': 0}
    # adv_accuracy = {'jsma_t': 0}
    # adv_accuracy = {'aa_n': 0}
    adv_accuracy = {'fgsm': 0}
    # adv_accuracy = {'cw_n': 0}
    # adv_accuracy = {'ddn_n': 0}
    correct = 0
    correct1 = 0
    sum = 0
    for adv in adv_accuracy:
        total = len(testloader)
        for image, label in testloader:
            image = image.cuda()
            label = label.cuda()

            # get model output 
            clean_out, adv_out = add_adv(classifier, image, label, adv, default=False)

            x_l, x_h = frequency_domain(28, adv_out)
            y_l, y_h = frequency_domain(28, clean_out)
            img1 = x_l - y_l
            img2 = x_h - y_h

            show_images(img1.data, img2.data)
            break
            adv_out_class = classifier(x_l)
            adv_out_class1 = classifier(x_h)

            prediction = torch.max(adv_out_class, 1)[1]
            prediction1 = torch.max(adv_out_class1, 1)[1]

            correct = correct + torch.eq(prediction, label).float().sum().item()
            correct1 = correct1 + torch.eq(prediction1, label).float().sum().item()
            sum = sum + image.size(0)

        accuracy = correct / sum
        accuracy1 = correct1 / sum
        print(total)
        print('Classifier_ACC_low: ', accuracy)
        print('Classifier_ACC_hign: ', accuracy1)




if __name__ == '__main__':
    # test_sample_acc()
    test_advsample_acc()