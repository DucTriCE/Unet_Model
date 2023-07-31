import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding='same')
        self.act1 = nn.ReLU()
        self.conv1_1 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding='same')
        self.act1_1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2))

        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding='same')
        self.act2 = nn.ReLU()
        self.conv2_1 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding='same')
        self.act2_1 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2))

        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding='same')
        self.act3 = nn.ReLU()
        self.conv3_1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding='same')
        self.act3_1 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=(2,2))

        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding='same')
        self.act4 = nn.ReLU()
        self.conv4_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same')
        self.act4_1 = nn.ReLU()
        self.act4_2 = nn.Dropout(0.5)
        self.act4_3 = nn.MaxPool2d(kernel_size=(2,2))

        # self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding='same')
        # self.act5 = nn.ReLU()
        # self.conv5_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding='same')
        # self.act5_1 = nn.ReLU()
        # self.act5_2 = nn.Dropout(0.5)

        self.up6 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up6_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same')
        self.act6 = nn.ReLU()
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding='same')
        self.act6_1 = nn.ReLU()
        self.conv6_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same')
        self.act6_2 = nn.ReLU()

        self.up7 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up7_1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding='same')
        self.act7 = nn.ReLU()
        self.conv7 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding='same')
        self.act7_1 = nn.ReLU()
        self.conv7_1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding='same')
        self.act7_2 = nn.ReLU()

        self.up8 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up8_1 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding='same')
        self.act8 = nn.ReLU()
        self.conv8 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding='same')
        self.act8_1 = nn.ReLU()
        self.conv8_1 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding='same')
        self.act8_2 = nn.ReLU()

        self.up9 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up9_1 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding='same')
        self.act9 = nn.ReLU()
        self.conv9 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding='same')
        self.act9_1 = nn.ReLU()
        self.conv9_1 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding='same')
        self.act9_2 = nn.ReLU()
        self.conv9_2 = nn.Conv2d(in_channels=8, out_channels=2, kernel_size=3, padding='same')
        self.act9_3 = nn.ReLU()
        self.conv10 = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, padding='same')
        self.act10 = nn.Sigmoid()


    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv1_1(x)
        x = self.act1_1(x)
        temp_3 = x
        # print(f"temp_3: {temp_3.shape}")
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv2_1(x)
        x = self.act2_1(x)
        temp_2 = x
        # print(f"temp_2: {temp_2.shape}")
        x = self.pool2(x)
        # print('conv1, 2: ',x.shape)

        x = self.conv3(x)
        x = self.act3(x)
        x = self.conv3_1(x)
        x = self.act3_1(x)
        temp_1 = x
        # print(f"temp_1: {temp_1.shape}")
        x = self.pool3(x)
        # print('conv3: ',x.shape)

        x = self.conv4(x)
        x = self.act4(x)
        x = self.conv4_1(x)
        x = self.act4_1(x)
        x = self.act4_2(x)
        temp = x
        # print(f"temp: {temp.shape}")
        x = self.act4_3(x)
        # print('conv4: ',x.shape)

        # x = self.conv5(x)
        # x = self.act5(x)
        # x = self.conv5_1(x)
        # x = self.act5_1(x)
        # x = self.act5_2(x)
        # print('conv5: ',x.shape)

        x = self.up6(x)
        x = self.up6_1(x)
        x = self.act6(x)
        # print("Concat: ", x.shape)
        x = torch.concat((x, temp), dim=1)
        x = self.conv6(x)
        x = self.act6_1(x)
        x = self.conv6_1(x)
        x = self.act6_2(x)
        # print('conv6: ',x.shape)

        x = self.up7(x)
        x = self.up7_1(x)
        x = self.act7(x)
        x = torch.concat((x, temp_1), dim=1)
        x = self.conv7(x)
        x = self.act7_1(x)
        x = self.conv7_1(x)
        x = self.act7_2(x)
        # print('conv7: ',x.shape)

        x = self.up8(x)
        x = self.up8_1(x)
        x = self.act8(x)
        x = torch.concat((x, temp_2), dim=1)
        x = self.conv8(x)
        x = self.act8_1(x)
        x = self.conv8_1(x)
        x = self.act8_2(x)
        # print('conv8: ',x.shape)

        x = self.up9(x)
        x = self.up9_1(x)
        x = self.act9(x)
        x = torch.concat((x, temp_3), dim=1)
        x = self.conv9(x)
        x = self.act9_1(x)
        x = self.conv9_1(x)
        x = self.act9_2(x)
        x = self.conv9_2(x)
        x = self.act9_3(x)
        x = self.conv10(x)
        x = self.act10(x)
        return x

def netParams(model):
    return np.sum([np.prod(parameter.size()) for parameter in model.parameters()])

if __name__ == '__main__':
    model = CNN()
    print(netParams(model))
    x = torch.rand(8, 3, 160, 80)
    output = model(x)
    print(output.shape)


