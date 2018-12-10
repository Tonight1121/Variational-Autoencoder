import torch.nn as nn
import torch
import math
import torch.utils.model_zoo as model_zoo
# import cv2
import sys
# sys.path.append('<path to package in your syste>')
import seaborn as sns
# import test_networks

import os
import tensorflow as tf
import time
import copy
from torch.autograd import Variable
from torch.nn import functional as F


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        # self.fc1 = nn.Linear(16 * 16 * 32, 256)


        # Decoder
        # self.fc3 = nn.Linear(256, 512)
        # self.fc4 = nn.Linear(512, 224*224)
        self.upspl1 = nn.Upsample(scale_factor=2, mode='linear')
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.deconv4 = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1)
        self.deconv5 = nn.ConvTranspose2d(128, 3, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        # out = self.relu(self.conv1(x))
        # print('relu1 shape {}'.format(out.shape))
        # out = self.relu(self.conv2(out))
        # print('relu2 shape {}'.format(out.shape))
        # out = self.relu(self.conv3(out))
        # print('relu3 shape {}'.format(out.shape))
        # out = self.relu(self.conv4(out))
        # print('encoded shape {}'.format(out.shape))
        x = self.conv1(x)
        # print('conv1 {}'.format(x.shape))
        x = self.conv2(x)
        # print('conv2 {}'.format(x.shape))
        x = self.conv3(x)
        # print('conv3 {}'.format(x.shape))
        x = self.conv4(x)
        # print('conv4 {}'.format(x.shape))
        x = torch.sigmoid(x)
        # time.sleep(30)
        return x


    def decode(self, z):
        # h3 = self.relu(self.fc3(z))
        # out = self.relu(self.fc4(h3))
        # # import pdb; pdb.set_trace()
        # out = out.view(out.size(0), 32, 16, 16)
        # out = self.relu(self.deconv1(out))
        # out = self.relu(self.deconv2(out))
        # out = self.relu(self.deconv3(out))
        # out = self.sigmoid(self.conv5(out))
        out = self.upspl1(z)
        out = self.deconv1(out)
        # print('deconv1 {}'.format(out.shape))
        # out = self.upspl1(out)
        # out = self.deconv2(out)
        # print('deconv2 {}'.format(out.shape))
        # time.sleep(30)
        # out = self.upspl1(out)
        # out = self.deconv3(out)
        # print('deconv3 {}'.format(out.shape))
        # out = self.upspl1(out)
        # out = self.deconv4(out)
        # print('deconv4 {}'.format(out.shape))
        out = self.upspl1(out)
        out = self.upspl1(out)
        out = self.deconv5(out)
        # print('deconv5 {}'.format(out.shape))
        # time.sleep(30)

        return out

    def forward(self, x):
        code = self.encode(x)
        img = self.decode(code)
        return img, code
