import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AODnet(nn.Module):
    def __init__(self):
        super(AODnet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=32, out_channels=4, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=4, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=4, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=4, kernel_size=7, padding=3)
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, padding=1)
        self.b = 1

        # super(AODnet, self).__init__()
        # self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)
        # self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1)
        # self.conv3 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=5, padding=2)
        # self.conv4 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=7, padding=3)
        # self.conv5 = nn.Conv2d(in_channels=12, out_channels=3, kernel_size=3, padding=1)
        # self.b = 1

    def forward(self, x, x0):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x))
        x3 = F.relu(self.conv3(x))
        x4 = F.relu(self.conv4(x))
        cat3 = torch.cat((x1, x2, x3, x4), 1)
        k = F.relu(self.conv5(cat3))

        # x1 = F.relu(self.conv1(x))
        # x2 = F.relu(self.conv2(x1))
        # cat1 = torch.cat((x1, x2), 1)
        # x3 = F.relu(self.conv3(cat1))
        # cat2 = torch.cat((x2, x3), 1)
        # x4 = F.relu(self.conv4(cat2))
        # cat3 = torch.cat((x1, x2, x3, x4), 1)
        # k = F.relu(self.conv5(cat3))

        # 等效于'if k.size() != x.size():',改变写法是为了消除warning
        # if not torch.zeros_like(k).equal(torch.zeros_like(x)):
            # raise Exception("k, haze image are different size!")

        output = k * x0 - k + self.b
        return F.relu(output)
