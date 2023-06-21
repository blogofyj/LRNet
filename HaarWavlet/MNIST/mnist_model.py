import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class HaarDownsampling(nn.Module):
    def __init__(self, channel_in):
        super(HaarDownsampling, self).__init__()
        self.channel_in = channel_in

        self.haar_weights = torch.ones(4, 1, 2, 2)

        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1

        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1

        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = torch.cat([self.haar_weights] * self.channel_in, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = False


    def forward(self, x, rev=False):
        if not rev:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(1/16.)

            out = F.conv2d(x, self.haar_weights, bias=None, stride=2, groups=self.channel_in) / 4.0
            out = out.reshape([x.shape[0], self.channel_in, 4, x.shape[2] // 2, x.shape[3] // 2])
            out = torch.transpose(out, 1, 2)

            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2] // 2, x.shape[3] // 2])
            # out = out.reshape([x.shape[0], self.channel_in * 2, x.shape[2] // 2, x.shape[3] // 2])
            return out
        else:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(16.)

            out = x.reshape([x.shape[0], 4, self.channel_in, x.shape[2], x.shape[3]])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2], x.shape[3]])
            # out = out.reshape([x.shape[0], self.channel_in * 2, x.shape[2], x.shape[3]])


            return F.conv_transpose2d(out, self.haar_weights, bias=None, stride=2, groups = self.channel_in)

    def jacobian(self, x, rev=False):
        return self.last_jac



class InvNet(nn.Module):
    def __init__(self, channel_in=1, channel_out=3, block_num=[], down_num=1):
        super(InvNet, self).__init__()

        operations = []

        current_channel = channel_in
        for i in range(down_num):
            b = HaarDownsampling(current_channel)
            # print('b:', b)
            operations.append(b)
            current_channel *= 4
            for j in range(block_num[i]):
                b = ResBlock_One(current_channel, current_channel)
                # b = ResBlock(current_channel, current_channel)
                # print('b', b)
                operations.append(b)

        # self.i = i
        self.operations = nn.ModuleList(operations)
        self.conv_low = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(True)
        self.sig = nn.Sigmoid()

    def forward(self, x, rev=False, cal_jacobian=False):
        out = x
        jacobian = 0

        # print('i :', self.i)
        # print('model ', self.operations)

        if not rev:
            for op in self.operations:
                out = op.forward(out, rev)

                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)
        else:
            for op in reversed(self.operations):
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)

        if cal_jacobian:
            return out, jacobian
        else:
            return out

    def low_conv(self, x):
        x = self.relu(self.conv_low(x))
        return x


# class ResBlock(nn.Module):
#     def __init__(self, channel_in, channel_out):
#         super(ResBlock, self).__init__()
#         feature = 64
#         self.conv1 = nn.Conv2d(channel_in, feature, kernel_size=3, padding=1)
#         self.relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
#         self.conv2 = nn.Conv2d(feature, feature, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d((feature+channel_in), channel_out, kernel_size=3, padding=1)
#
#     def forward(self, x, rev):
#         residual = self.relu1(self.conv1(x))
#         residual = self.relu1(self.conv2(residual))
#         input = torch.cat((x, residual), dim=1)
#         out = self.conv3(input)
#         return out


class ResBlock_One(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(ResBlock_One, self).__init__()
        feature = 64
        self.conv1 = nn.Conv2d(channel_in, feature, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(feature)
        self.relu1 = nn.ReLU(True)
        self.conv2 = nn.Conv2d(feature, feature, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d((feature+channel_in), channel_out, kernel_size=3, padding=1)

    def forward(self, x, rev):
        residual = self.bn1(self.relu1(self.conv1(x)))
        residual = self.bn1(self.relu1(self.conv2(residual)))
        # residual = self.relu1(self.conv1(x))
        # residual = self.relu1(self.conv2(residual))
        input = torch.cat((x, residual), dim=1)
        out = self.conv3(input)
        return out
