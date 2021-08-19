import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, kernel_size=3):
        super(ResBlock, self).__init__()
        self.padding = kernel_size // 2; # auto padding
        self.normal = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, 
                      stride=stride, padding=self.padding, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size, 
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel)
        )

        # shortcut block
        if (stride==1 and in_channel==out_channel):
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        out = self.normal(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class myNet(nn.Module):
    def __init__(self, ResBlock):
        super().__init__() # original: 32 * 32 * 3
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1) # in-channel, out-channel, kernal-size 32 * 32 * 64
        self.resnet = ResBlock(64, 128) # 32 * 32 * 256
        self.pool = nn.MaxPool2d(2, 2) # kernel-size, stride 16 * 16 * 128
        self.conv2 = nn.Conv2d(128, 256, 5) # 12 * 12 * 256 -> 6 * 6 * 256 (pool twice)
        self.fc = nn.Linear(256 * 6 * 6, 1) 

    def forward(self, x):
        out = self.conv1(x)
        out = self.resnet(out)
        
        out = self.pool(out)
        out = self.pool(F.relu(self.conv2(out)))
        # print("----------"*5)
        # print(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        out = torch.sigmoid(out)
        return out

net = myNet(ResBlock)