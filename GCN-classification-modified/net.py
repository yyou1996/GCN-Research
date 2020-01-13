'''
    VGG16 in PyTorch.

'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import pdb


class VGG_GCN_graph(nn.Module):

    def __init__(self, origin_model, threshold=0.5, labels=10):
        super(VGG_GCN_graph, self).__init__()
        self.features = origin_model.features
        self.classifier = nn.Linear(512, 10)
        self.linear = nn.Sequential(
                nn.Linear(512,512),
                nn.ReLU(),
                nn.Linear(512,512)
        )
        self.threshold = threshold

        self.sigmoid = nn.Sigmoid()

    def adj_matrix(self, target):

        batch_size = target.size(0)
        # print(target.size())
        # assert False

        target_matrix = target.repeat(batch_size,1)
        adj = (target_matrix == torch.transpose(target_matrix,0,1))
        adj = adj.float()

        # print(target)
        # print(adj)
        
        return adj.detach()

    def graph(self, feature):

        # feature_trans = torch.transpose(feature, 0, 1)

        '''
        feature_trans = torch.transpose(feature, 0, 1)
        norm = torch.norm(feature, p=2, dim=1).pow(2)
        feature_norm = norm.repeat(feature.size(0),1)
        distance = feature_norm + torch.transpose(feature_norm, 0, 1) - 2*torch.mm(feature, feature_trans)
        '''

        inner_product = torch.einsum('ab,cb->ac', feature, feature)
        inner_product = self.sigmoid(inner_product)

        # inner_product = torch.mm(feature, feature_trans)
        # print(inner_product)
        # pdb.set_trace()
        '''        
        distance = torch.sigmoid(distance)
        distance = 2*(1-distance)
        distance = torch.clamp(distance,0,1)
        '''
        # zerotensor = torch.zeros_like(distance)
        # onetensor = torch.ones_like(distance)
        # top10_adj = torch.where(distance > 0.5, onetensor, zerotensor)


        return inner_product

    def adj(self, adj_matrix):
        batch_size = adj_matrix.size(0)

        '''
        D_inv = torch.diag(1/torch.sum(adj_matrix,dim=1))
        A = torch.mm(adj_matrix, D_inv)
        A = A + torch.eye(batch_size).cuda()
        D_inv = torch.diag(1/torch.sum(A, dim=1))
        matrix = torch.mm(A, D_inv)
        '''

        A_hat = adj_matrix + torch.eye(batch_size).cuda()
        D_hat_inverse = torch.diag(1/torch.sum(A_hat,dim=1))
        matrix = torch.mm(A_hat, D_hat_inverse)
        # print(matrix)

        return matrix.detach()

    def gcn(self, distance):

        batch_size = distance.size(0)
        zerotensor = torch.zeros_like(distance)
        onetensor = torch.ones_like(distance)

        # mask = torch.where(distance > self.threshold, onetensor, zerotensor)  

        # # top10_adj = torch.zeros_like(distance)
        # # idx = torch.topk(distance, 10)[1]
        # # for ii in range(batch_size):
        # #     top10_adj[ii,idx[ii]] = 1
        
        # # top10_adj = top10_adj*mask

        top10_adj = torch.where(distance > 0.5, onetensor, zerotensor)

        A_hat = top10_adj + torch.eye(batch_size).cuda()
        D_hat_inverse = torch.diag(1/torch.sum(A_hat,dim=1))
        matrix = torch.mm(A_hat, D_hat_inverse)

        return matrix.detach()

    def forward(self, input_list, graph_ensemble=True):
        x = input_list[0]
        target = input_list[1]

        out = self.features(x)
        out = out.view(out.size(0), -1)

        # graph
        low_feature = self.linear(out)
        # low_feature = out
        graph_pre = self.graph(low_feature)
        # print(graph_pre.size())
        graph_gt = self.adj_matrix(target)

        # mat = self.gcn(graph_pre)

        graph_pre_1 = torch.zeros(graph_pre.size()).cuda()
        _, topk_idx = torch.topk(graph_pre, 5, dim=1) # 5
        batch_size = graph_pre.size()[0]
        for bs in range(batch_size):
            graph_pre_1[bs, topk_idx[bs]] = graph_pre[bs, topk_idx[bs]]
            ##
            graph_pre_1[topk_idx[bs], bs] = graph_pre[topk_idx[bs], bs]
        # print(graph_pre_1)
        # print(graph_pre_1[0, topk_idx[0]])
        thresh = 0.9
        graph_pre_1[graph_pre_1>thresh] = 1
        graph_pre_1[graph_pre_1<=thresh] = 0
        # graph_pre_1[graph_pre_1!=0] = 1
        gp = graph_pre_1
        mat = self.adj(graph_pre_1)
        # print(mat)
        # out = torch.mm(mat, out)

        out = self.classifier(out)
        out = F.softmax(out, dim=1)
        if graph_ensemble:
            out = torch.mm(mat, out)
        out = torch.log(out)

        # print(out)
        # assert False

        return out, graph_pre, graph_gt, gp

class VGG_GCN_cifar10(nn.Module):

    def __init__(self, origin_model, threshold, labels=10):
        super(VGG_GCN_cifar10, self).__init__()
        self.threshold = threshold
        self.features = origin_model.features
        self.classifier = nn.Linear(512, 10)

    def graph(self, feature):

        feature_trans = torch.transpose(feature, 0, 1)
        norm = torch.norm(feature, p=2, dim=1).pow(2)
        feature_norm = norm.repeat(feature.size(0),1)
        distance = feature_norm + torch.transpose(feature_norm, 0, 1) - 2*torch.mm(feature, feature_trans)
        onetensor = torch.ones_like(distance)
        zerotensor = torch.zeros_like(distance)
        adj = torch.where(distance<=self.threshold, onetensor, zerotensor)  
        A_hat = adj + torch.eye(feature.size(0)).cuda()
        D_hat_inverse = torch.diag(1/torch.sum(A_hat,dim=1))
        matrix = torch.mm(A_hat, D_hat_inverse)
        return matrix.detach()

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

def VGG16_GCN_graph():
    origin_model = models.vgg16_bn(pretrained=False)
    return VGG_GCN_graph(origin_model, threshold=0.5, labels=10)

if __name__ == '__main__':
    origin_model = models.vgg16_bn(pretrained=False)
    print(origin_model)
