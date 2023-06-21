import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
'''
copy from cifar train_model
'''

class InvBlockExp(nn.Module):
    def __init__(self, subnet_constructor, channel_num, channel_split_num, clamp=1.):
        super(InvBlockExp, self).__init__()

        self.split_len1 = channel_split_num
        self.split_len2 = channel_num - channel_split_num

        self.clamp = clamp

        self.F = subnet_constructor(self.split_len2, self.split_len1)
        self.G = subnet_constructor(self.split_len1, self.split_len2)
        self.H = subnet_constructor(self.split_len1, self.split_len2)

    def forward(self, x, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        if not rev:
            y1 = x1 + self.F(x2)
            self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
            y2 = x2.mul(torch.exp(self.s)) + self.G(y1)
        else:
            self.s = self.clamp * (torch.sigmoid(self.H(x1)) * 2 - 1)
            y2 = (x2 - self.G(x1)).div(torch.exp(self.s))
            y1 = x1 - self.F(y2)

        return torch.cat((y1, y2), 1)

    def jacobian(self, x, rev=False):
        if not rev:
            jac = torch.sum(self.s)
        else:
            jac = -torch.sum(self.s)

        return jac / x.shape[0]


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
            return out
        else:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(16.)

            out = x.reshape([x.shape[0], 4, self.channel_in, x.shape[2], x.shape[3]])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2], x.shape[3]])


            return F.conv_transpose2d(out, self.haar_weights, bias=None, stride=2, groups = self.channel_in)

    def jacobian(self, x, rev=False):
        return self.last_jac

# block_num = [8, 8]
# down_num = int(math.log(4, 2))
# def subnet(net_structure):
#     def constructor(channel_in, channel_out):
#         if net_structure == 'Resnet':
#             return ResBlock(channel_in, channel_out)
#         else:
#             return None
#     return constructor


class InvNet(nn.Module):
    def __init__(self, channel_in=1, channel_out=1, block_num=[], down_num=1):
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
                # print('b', b)
                operations.append(b)

        # self.i = i
        self.operations = nn.ModuleList(operations)
        self.conv_low = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)
        self.clean = nn.Conv2d(4, 3, kernel_size=1)
        self.relu = nn.ReLU(True)
        # self.fc1 = nn.Linear(588,100)
        # self.fc1 = nn.Linear(392,100)
        # self.fc2 = nn.Linear(100, 10)
        self.fc1 = nn.Linear(196, 80)
        self.fc2 = nn.Linear(80, 10)
        # self.fc1 = nn.Linear(196, 100)
        # self.fc2 = nn.Linear(100, 10)
        # self.fc = nn.Linear(planes, 10)

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
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def low_fre(self, x):
        x = self.relu(self.conv_low(x))
        return x
    #
    # def low_clean(self, x):
    #     x = self.relu(self.clean(x))
    #     return x


    # def CAS(self, out, label=None):
    #     fc_in = torch.mean(out.view(out.shape[0], out.shape[1], -1), dim=-1)
    #     fc_out = self.fc(fc_in.view(out.shape[0], out.shape[1]))
    #     if self.training:
    #         N, C, H, W = out.shape
    #         mask = self.fc.weight[label, :]
    #         out = out * mask.view(N, C, 1, 1)
    #     else:
    #         N, C, H, W = out.shape
    #         pred_label = torch.max(fc_out, dim=1)[1]
    #         mask = self.fc.weight[pred_label, :]
    #         out = out * mask.view(N, C, 1, 1)
    #     return out, fc_out

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

