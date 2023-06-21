import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import torch
from torchsummary import summary
from new_haar_model import InvNet

if __name__ == "__main__":
    block_num = [3]
    netG = InvNet(block_num=block_num).cuda()

    summary(netG, input_size=(1,28,28))