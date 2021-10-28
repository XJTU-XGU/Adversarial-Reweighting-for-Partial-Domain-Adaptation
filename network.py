import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import numpy as np

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

resnet_dict = {"ResNet18":models.resnet18, "ResNet34":models.resnet34, "ResNet50":models.resnet50, "ResNet101":models.resnet101, "ResNet152":models.resnet152}

class ResNetFc(nn.Module):
  def __init__(self, resnet_name, bottleneck_dim=256, class_num=1000, radius=8.5, normalize_classifier=True):
    super(ResNetFc, self).__init__()
    model_resnet = resnet_dict[resnet_name](pretrained=True)
    self.conv1 = model_resnet.conv1
    self.bn1 = model_resnet.bn1
    self.relu = model_resnet.relu
    self.maxpool = model_resnet.maxpool
    self.layer1 = model_resnet.layer1
    self.layer2 = model_resnet.layer2
    self.layer3 = model_resnet.layer3
    self.layer4 = model_resnet.layer4
    self.avgpool = model_resnet.avgpool
    self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                         self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)
    if normalize_classifier:
        self.bottleneck = nn.Linear(model_resnet.fc.in_features, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.fc = SLR_layer(bottleneck_dim, class_num, bias=True)
        self.__in_features = bottleneck_dim
    else:
        self.bottleneck = nn.Linear(model_resnet.fc.in_features, bottleneck_dim)
        self.fc = nn.Linear(bottleneck_dim, class_num)
        self.bottleneck.apply(init_weights)
        self.fc.apply(init_weights)
        self.__in_features = bottleneck_dim
    self.radius = radius

  def forward(self, x):
    x = self.feature_layers(x)
    x = x.view(x.size(0), -1)
    x = self.bottleneck(x)
    x = self.radius*F.normalize(x,dim=1)
    y = self.fc(x)
    return x, y

  def output_num(self):
    return self.__in_features

  def get_parameters(self):
    parameter_list = [{"params":self.feature_layers.parameters(), "lr_mult":1, 'decay_mult':2}, \
                    {"params":self.bottleneck.parameters(), "lr_mult":10, 'decay_mult':2}, \
                    {"params":self.fc.parameters(), "lr_mult":10, 'decay_mult':2}]

    return parameter_list

vgg_dict = {"VGG11":models.vgg11, "VGG13":models.vgg13, "VGG16":models.vgg16, "VGG19":models.vgg19, "VGG11BN":models.vgg11_bn, "VGG13BN":models.vgg13_bn, "VGG16BN":models.vgg16_bn, "VGG19BN":models.vgg19_bn}
class VGGFc(nn.Module):
  def __init__(self, vgg_name, bottleneck_dim=256, class_num=1000,radius=8.5, use_slr=True):
    super(VGGFc, self).__init__()
    model_vgg = vgg_dict[vgg_name](pretrained=True)
    self.features = model_vgg.features
    self.classifier = nn.Sequential()
    for i in range(6):
        self.classifier.add_module("classifier"+str(i), model_vgg.classifier[i])
    self.feature_layers = nn.Sequential(self.features, self.classifier)

    self.bottleneck = nn.Linear(4096, bottleneck_dim)
    self.bottleneck.apply(init_weights)
    if use_slr:
        self.fc = SLR_layer(bottleneck_dim, class_num, bias=True)
    else:
        self.fc = nn.Linear(bottleneck_dim, class_num, bias=True)
        self.fc.apply(init_weights)
    self.__in_features = bottleneck_dim
    self.radius = radius

  def forward(self, x):
      x = self.feature_layers(x)
      x = x.view(x.size(0), -1)
      x = self.bottleneck(x)
      x = self.radius * F.normalize(x, dim=1)
      y = self.fc(x)
      return x, y

  def output_num(self):
    return self.__in_features

  def get_parameters(self):
    parameter_list = [{"params":self.features.parameters(), "lr_mult":1, 'decay_mult':2}, \
                    {"params":self.classifier.parameters(), "lr_mult":1, 'decay_mult':2}, \
                    {"params":self.bottleneck.parameters(), "lr_mult":10, 'decay_mult':2}, \
                    {"params":self.fc.parameters(), "lr_mult":10, 'decay_mult':2}]

    return parameter_list

class SLR_layer(nn.Module):
    def __init__(self, in_features, out_features,bias=True):
        super(SLR_layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias=torch.nn.Parameter(torch.zeros(out_features))
        self.bias_bool = bias
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input):
        r=input.norm(dim=1).detach()[0]
        if self.bias_bool:
            cosine = F.linear(input, F.normalize(self.weight),r*torch.tanh(self.bias))
        else:
            cosine = F.linear(input, F.normalize(self.weight))
        output=cosine
        return output

class WassersteinDiscriminator(nn.Module):
  def __init__(self, in_feature, hidden_size):
    super(WassersteinDiscriminator, self).__init__()
    self.ad_layer1 = nn.Linear(in_feature, hidden_size)
    self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
    self.ad_layer3 = nn.Linear(hidden_size, 1)
    self.relu1 = nn.ReLU()
    self.relu2 = nn.ReLU()
    self.dropout1 = nn.Dropout(0.5)
    self.dropout2 = nn.Dropout(0.5)
    self.sigmoid = nn.Sigmoid()
    self.apply(init_weights)

  def forward(self, x):
    x = self.ad_layer1(x)
    x = self.relu1(x)
    x = self.dropout1(x)
    x = self.ad_layer2(x)
    x = self.relu2(x)
    x = self.dropout2(x)
    y = self.ad_layer3(x)
    return y

  def output_num(self):
    return 1

  def get_parameters(self):
    return [{"params":self.parameters(), "lr_mult":1, 'decay_mult':2}]

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)


class AdversarialNetwork(nn.Module):
  def __init__(self, in_feature, hidden_size, max_iter = 10000):
    super(AdversarialNetwork, self).__init__()
    self.ad_layer1 = nn.Linear(in_feature, hidden_size)
    self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
    self.ad_layer3 = nn.Linear(hidden_size, 1)
    self.apply(init_weights)
    self.iter_num = 0
    self.alpha = 10
    self.low = 0.0
    self.high = 1.0
    self.max_iter = max_iter

  def forward(self, x):
    if self.training:
        self.iter_num += 1
    coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
    x = x * 1.0
    x.register_hook(grl_hook(coeff))
    x = self.ad_layer1(x)
    x = self.relu1(x)
    x = self.dropout1(x)
    x = self.ad_layer2(x)
    x = self.relu2(x)
    x = self.dropout2(x)
    y = self.ad_layer3(x)
    y = self.sigmoid(y)
    return y

  def output_num(self):
    return 1

  def get_parameters(self):
    return [{"params":self.parameters(), "lr_mult":10, 'decay_mult':2}]
