import torch
import torch.nn as nn
import numpy as np

import net as net
from utils import load_data
from sklearn.metrics import f1_score


def run():

    setup_seed(11) # 11, 0, 0
    adj, features, labels, idx_train, idx_val, idx_test = load_data('cora') # 'cora', 'citeseer', 'pubmed'
    adj = adj.cuda()
    features = features.cuda()
    labels = labels.cuda()

    net_gcn = net.net_gcn(embedding_dim=[1433,16,7]) # [1433,16,7], [3703,16,6], [500,16,3]
    net_gcn = net_gcn.cuda()
    optimizer = torch.optim.Adam(net_gcn.parameters(), lr=0.01, weight_decay=5e-4)
    loss_func = nn.CrossEntropyLoss()
    loss_val = []
    early_stopping = 10

    for epoch in range(1000):

        optimizer.zero_grad()
        output = net_gcn(features, adj)
        loss_train = loss_func(output[idx_train], labels[idx_train])
        print('epoch', epoch, 'loss', loss_train.data)
        loss_train.backward()
        optimizer.step()

        # validation
        with torch.no_grad():
            output = net_gcn(features, adj, val_test=True)
            loss_val.append(loss_func(output[idx_val], labels[idx_val]).cpu().numpy())
            print('val acc', f1_score(labels[idx_val].cpu(), output[idx_val].cpu().numpy().argmax(axis=1), average='micro'))

        # early stopping
        if epoch > early_stopping and loss_val[-1] > np.mean(loss_val[-(early_stopping+1):-1]):
            break

    # test
    with torch.no_grad():
        output = net_gcn(features, adj, val_test=True)
        print('')
        print('test acc', f1_score(labels[idx_test].cpu(), output[idx_test].cpu().numpy().argmax(axis=1), average='micro'))


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    run()

