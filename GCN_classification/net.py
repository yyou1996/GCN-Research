'''
    VGG16 in PyTorch.

'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class VGG_GCN_cifar10(nn.Module):

    def __init__(self, origin_model, threshold, labels=10):
        super(VGG_GCN_cifar10, self).__init__()
        self.threshold = threshold
        self.features = origin_model.features
        self.classifier = nn.Linear(512, 10)

    def graph(self, feature):
        feature_trans = torch.transpose(feature, 0, 1)
        feature_norm = torch.norm(feature, p=2, dim=1)
        feature_norm = feature_norm.repeat(feature.size(0),1).pow(2)
        distance = feature_norm + torch.transpose(feature_norm, 0, 1) - 2*torch.mm(feature, feature_trans)
        onetensor = torch.ones_like(distance)
        zerotensor = torch.zeros_like(distance)
        adj = torch.where(distance<=self.threshold, onetensor, zerotensor)  
        A_hat = adj + torch.eye(feature.size(0)).cuda()
        D_hat_inverse = torch.diag(1/torch.sum(A_hat,dim=1))
        matrix = torch.mm(A_hat, D_hat_inverse)
        return matrix

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        mat = self.graph(out)
        out = torch.mm(mat, out)
        out = self.classifier(out)
        return out

class VGG_cifar10(nn.Module):

    def __init__(self, origin_model, labels=10):
        super(VGG_cifar10, self).__init__()
        self.features = origin_model.features
        self.classifier = nn.Linear(512, 10)

    def graph(self, feature):
        feature_trans = torch.transpose(feature, 0, 1)
        feature_norm = torch.norm(feature, p=2, dim=1)
        feature_norm = feature_norm.repeat(feature.size(0),1).pow(2)
        distance = feature_norm + torch.transpose(feature_norm, 0, 1) - 2*torch.mm(feature, feature_trans)
        return torch.mean(distance)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        dis = self.graph(out)
        out = self.classifier(out)
        return out, dis

def VGG16():
    origin_model = models.vgg16_bn(pretrained=False)
    return VGG_cifar10(origin_model, labels=10)

def VGG16_GCN(threshold):
    origin_model = models.vgg16_bn(pretrained=False)
    return VGG_GCN_cifar10(origin_model, threshold, labels=10)


if __name__ == '__main__':
    origin_model = models.vgg16_bn(pretrained=False)
    print(origin_model)