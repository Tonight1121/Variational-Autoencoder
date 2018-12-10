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
import numpy as np
import matplotlib as plt
from matplotlib import *
from sklearn import manifold, datasets
from sklearn.cluster import KMeans
from matplotlib import figure

###################################################################
import matplotlib
import matplotlib.colorbar
from matplotlib import style
from matplotlib import _pylab_helpers, interactive
from matplotlib.cbook import dedent, silent_list, is_numlike
from matplotlib.cbook import _string_to_bool
from matplotlib.cbook import deprecated, warn_deprecated
from matplotlib import docstring
from matplotlib.backend_bases import FigureCanvasBase
from matplotlib.figure import Figure, figaspect
from matplotlib.gridspec import GridSpec
from matplotlib.image import imread as _imread
from matplotlib.image import imsave as _imsave
from matplotlib import rcParams, rcParamsDefault, get_backend
from matplotlib import rc_context
from matplotlib.rcsetup import interactive_bk as _interactive_bk
from matplotlib.artist import getp, get, Artist
from matplotlib.artist import setp as _setp
from matplotlib.axes import Axes, Subplot
from matplotlib.projections import PolarAxes
from matplotlib import mlab  # for csv2rec, detrend_none, window_hanning
from matplotlib.scale import get_scale_docs, get_scale_names

from matplotlib import cm
from matplotlib.cm import get_cmap, register_cmap

import numpy as np

# We may not need the following imports here:
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.text import Text, Annotation
from matplotlib.patches import Polygon, Rectangle, Circle, Arrow
from matplotlib.widgets import SubplotTool, Button, Slider, Widget


from matplotlib.backends import pylab_setup
###################################################################



__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

myinputs = []

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}



def featuremap(x, fc_weights_softmax1, fc_weights_softmax2):
    feature_space = x.data.cpu().numpy()
    feature_map1 = feature_space[0, 0, :, :]
    feature_map2 = feature_space[0, 1, :, :]
    normalized_slice1 = test_networks.array_normalize(feature_map1)
    normalized_slice2 = test_networks.array_normalize(feature_map2)
    normalized_0 = normalized_slice1 * fc_weights_softmax1[0] + normalized_slice2 * fc_weights_softmax1[1]
    normalized_1 = normalized_slice1 * fc_weights_softmax2[0] + normalized_slice2 * fc_weights_softmax1[1]
    cv2.imwrite('heatmaps/featuremap1.jpg', normalized_slice1)
    cv2.imwrite('heatmaps/featuremap2.jpg', normalized_slice2)
    cv2.imwrite('heatmaps/normalized_0.jpg', normalized_0)
    cv2.imwrite('heatmaps/normalized_1.jpg', normalized_1)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# class CNN_VAE(nn.Module):
#     def __init__(self):
#         super(CNN_VAE, self).__init__()
#
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2)
#         self.bn1 = nn.BatchNorm2d(16)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=2)
#         self.conv3 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=2)
#         self.conv4_mu = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1)
#         self.conv4_logvar = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1)
#         self.avgpool = nn.AvgPool2d(25, stride=1)
#
#         self.deconv1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3)
#         self.deconv2 = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=16)
#         self.deconv3 = nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=64)
#         self.deconv4 = nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=144)
#
#         # self.deconv1 = nn.UpsamplingBilinear2d(scale_factor=2)
#         # self.deconv2 = nn.UpsamplingBilinear2d(scale_factor=4)
#         # self.deconv3 = nn.UpsamplingBilinear2d(scale_factor=4)
#         # self.deconv4 = nn.UpsamplingBilinear2d(scale_factor=7)
#         # self
#
#         # self.fc1 = nn.Linear(1024, 6400)
#         self.fc2 = nn.Linear(512, 3*224*224)
#         # self.fc5 = nn.Linear(1600, 150528)
#         # self.fc6 = nn.Linear(6400, 150528)
#
#     def encode(self, x):
#         # print('input x {}'.format(x.shape))
#         x = self.conv1(x)
#         # print('conv1 x {}'.format(x.shape))
#         # x = self.bn1(x)
#         # print('bn1 x {}'.format(x.shape))
#         # x = self.relu(x)
#         # print('relu x {}'.format(x.shape))
#         # x = self.maxpool(x)
#         # print('maxpl x {}'.format(x.shape))
#         x = self.conv2(x)
#         # print('conv2 x {}'.format(x.shape))
#         x = self.conv3(x)
#         # print('conv3 x {}'.format(x.shape))
#
#         mu = self.conv4_mu(x)
#         # print('conv4_mu mu {}'.format(mu.shape))
#         mu = self.avgpool(mu)
#         # print('avgpool mu {}'.format(mu.shape))
#         mu = mu.view(mu.size(0), -1)
#
#         logvar = self.conv4_logvar(x)
#         logvar = self.avgpool(logvar)
#         logvar = logvar.view(logvar.size(0), -1)
#
#         # print('mu shape {}'.format(mu.shape))
#         # print('logvar shape {}'.format(logvar.shape))
#         # time.sleep(30)
#         return mu, logvar
#
#     def reparametrize(self, mu, logvar):
#         std = logvar.mul(0.5).exp_()
#         if torch.cuda.is_available():
#             eps = torch.cuda.FloatTensor(std.size()).normal_()
#         else:
#             eps = torch.FloatTensor(std.size()).normal_()
#         eps = Variable(eps)
#         return eps.mul(std).add_(mu)
#
#     def decode(self, z):
#         # z = z.view(z.size(0), 512, 1, 1)
#         # z = self.deconv1(z)
#         # print('deconv1 z {}'.format(z.shape))
#         # z = self.deconv2(z)
#         # print('deconv1 z {}'.format(z.shape))
#         # z = self.deconv3(z)
#         # print('deconv1 z {}'.format(z.shape))
#         # z = self.deconv4(z)
#         # print('deconv1 z {}'.format(z.shape))
#         # time.sleep(30)
#         # z = self.fc1(z)
#         z = self.fc2(z)
#         return torch.sigmoid(z)
#
#     def forward(self, x):
#
#         mu, logvar = self.encode(x)
#         z = self.reparametrize(mu, logvar)
#         z = self.decode(z)
#         z = z.view(z.size(0), 3, 224, 224)
#         # z = z.type(torch.cuda.FloatTensor)
#         # z = torch.cuda.LongTensor(z.size())
#         return z, mu, logvar



class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)   # 512 * 7 * 7

        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.deconv4 = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1)
        self.upspl1 = nn.Upsample(scale_factor=2, mode='nearest')
        # self.upspl2 = nn.Upsample(scale_factor=4, mode='nearest')


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def encode(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # print('maxpool {}'.format(x.shape))
        x = self.layer1(x)
        # print('layer1 {}'.format(x.shape))
        x = self.layer2(x)
        # print('layer2 {}'.format(x.shape))
        x = self.layer3(x)
        # print('layer3 {}'.format(x.shape))

        code = self.layer4(x)
        # print('layer4 {}'.format(code.shape))
        # time.sleep(30)
        # mu = self.avgpool(mu)
        # print('avgpl {}'.format(mu.shape))
        # mu = mu.view(mu.size(0), -1)

        # logvar = self.layer5(x)
        # print('layer5 {}'.format(logvar.shape))
        # logvar = self.avgpool(logvar)
        # print('avgpl {}'.format(logvar.shape))
        # logvar = logvar.view(logvar.size(0), -1)
        return code

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        z = self.upspl1(z)
        z = self.deconv1(z)
        # print('deconv1 {}'.format(z.shape))
        z = self.upspl1(z)
        z = self.deconv2(z)
        # print('deconv2 {}'.format(z.shape))
        z = self.upspl1(z)
        z = self.deconv3(z)
        # print('deconv3 {}'.format(z.shape))
        z = self.upspl1(z)
        z = self.deconv4(z)
        z = self.upspl1(z)
        # print('deconv4 {}'.format(z.shape))
        # time.sleep(30)
        # return torch.sigmoid(z)
        return z

    def forward(self, x):

        # mu, logvar = self.encode(x)
        code = self.encode(x)
        # z = self.reparametrize(mu, logvar)
        # z = code.view(code.size(0), -1)
        z = self.decode(code)
        # z = z.view(z.size(0), 3, 224, 224)
        # z = z.type(torch.cuda.FloatTensor)
        # z = torch.cuda.LongTensor(z.size())
        return z, code


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
