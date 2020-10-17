import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm

from .mixture_density_net import MixtureDensityNet


class ImageFeature(nn.Module):

    def __init__(self, num_dense_feature: int):
        super(ImageFeature, self).__init__()
        self.num_dense_feature = num_dense_feature
        self.conv1 = spectral_norm(nn.Conv2d(1, 64, 3))
        self.conv2 = spectral_norm(nn.Conv2d(64, 64, 3))
        self.maxpool = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.batch = nn.BatchNorm1d(9216)
        self.linear1 = nn.Linear(9216, 128)
        self.linear2 = nn.Linear(128, 64)

    def forward(self, data):
        dense = data[:, :self.num_dense_feature]
        image = data[:, self.num_dense_feature:]
        image = image.reshape((-1, 1, 28, 28))
        image_feature = F.relu(self.conv1(image))
        image_feature = self.maxpool(F.relu(self.conv2(image_feature)))
        image_feature = torch.flatten(image_feature, start_dim=1)
        image_feature = self.dropout1(image_feature)
        image_feature = self.dropout2(F.relu(self.linear1(image_feature)))
        image_feature = self.linear2(image_feature)
        return torch.cat([dense, image_feature], dim=1)



class ResponseModel(nn.Module):

    def __init__(self, dropout_ratio):
        super(ResponseModel, self).__init__()
        self.net = nn.Sequential(ImageFeature(2),
                                 nn.Linear(66, 32),
                                 nn.ReLU(),
                                 nn.Dropout(dropout_ratio),
                                 nn.Linear(32, 1)
                                 )

    def forward(self, treatment, covariate):
        feature = torch.cat([treatment, covariate], dim=1)
        return self.net(feature)


def build_net_for_demand_image(dropout_rate, **args):
    instrumental_net = nn.Sequential(ImageFeature(2),
                                     nn.Linear(66, 32),
                                     nn.ReLU(),
                                     nn.Dropout(dropout_rate),
                                     MixtureDensityNet(32, 1, 10))

    response_net = ResponseModel(dropout_rate)
    return instrumental_net, response_net
