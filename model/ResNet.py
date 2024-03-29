import torch
import torch.nn as nn

class resnet(nn.Module):
    def __init__(self, input_size, output_size):
        if input_size == 0 or output_size == 0:
            raise ValueError("Requiring input_size and output_size.")
        super(resnet, self).__init__()
        #如果padding='same'，则kernel_size需要是奇数，否则会警告

        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size = 7, padding = 'same'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size = 5, padding = 'same'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size = 3, padding = 'same'),
            nn.BatchNorm1d(64),
            )
        self.shortcut1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size = 1, padding = 'same'),
            nn.BatchNorm1d(64),
            )
        self.act1 = nn.ReLU()

        self.layer2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size = 7, padding = 'same'),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size = 5, padding = 'same'),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size = 3, padding = 'same'),
            nn.BatchNorm1d(128),
            )
        self.shortcut2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size = 1, padding = 'same'),
            nn.BatchNorm1d(128),
            )
        self.act2 = nn.ReLU()

        #block3的输出通道数被我修改为1
        self.layer3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size = 7, padding = 'same'),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size = 5, padding = 'same'),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1, kernel_size = 3, padding = 'same'),
            nn.BatchNorm1d(1),
            )
        self.shortcut3 = nn.Sequential(
            nn.Conv1d(128, 1, kernel_size = 1, padding = 'same'),
            nn.BatchNorm1d(1),
            )
        self.act3 = nn.ReLU()

        self.final = nn.Sequential(
            nn.AdaptiveAvgPool1d(input_size),
            nn.Linear(input_size, output_size),
            nn.Softmax(dim=-1)
            )

    def forward(self, input):
        input = input.unsqueeze(dim=1)

        after_conv = self.layer1(input)
        short = self.shortcut1(input)
        input = self.act1(after_conv + short)

        after_conv = self.layer2(input)
        short = self.shortcut2(input)
        input = self.act2(after_conv + short)

        after_conv = self.layer3(input)
        short = self.shortcut3(input)
        input = self.act3(after_conv + short)

        input = input.squeeze(dim=1)#进入linear之前再去除通道维度
        input = self.final(input)
        return input