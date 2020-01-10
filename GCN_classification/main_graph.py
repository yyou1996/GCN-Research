import argparse
import os
import torch
import random
import numpy as np 
import torch.optim
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import net 
import pdb 


parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')
parser.add_argument('--data', type=str, default='/data4/zzy/data/', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--epochs', default=300, type=int, help='number of total epochs to run')
parser.add_argument('--print_freq', default=50, type=int, help='print frequency')


parser.add_argument('--save_dir', help='The directory used to save the trained models', default='/data4/zzy/model/vgg_thres=300', type=str)
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--decreasing_lr', default='150,250', help='decreasing strategy')

best_prec1 = 0

def main():
    global args, best_prec1
    args = parser.parse_args()
    print(args)

    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
    torch.cuda.set_device(int(args.gpu))

    setup_seed(20)

    model = net.VGG16_GCN_graph()

    model_path = torch.load('/data4/zzy/model/baseline_vgg16bn/best_model.pt', map_location=torch.device('cuda:'+str(args.gpu)))['state_dict']
    model_dict = model.state_dict()
    model_dict.update(model_path)
    model.load_state_dict(model_dict)
    print('model_loaded')


    for parm in model.features.parameters():
        parm.requires_grad = False

    for parm in model.classifier.parameters():
        parm.requires_grad = False 
    
    print('layer fixed')

    model.cuda()

    cudnn.benchmark = True

    train_trans = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    val_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=args.data, train=True, transform=train_trans, download=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=args.data, train=False, transform=val_trans),
        batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True)


    criterion = nn.CrossEntropyLoss()
    criterion_graph = nn.BCELoss()
    criterion = criterion.cuda()
    criterion_graph = criterion_graph.cuda()

    optimizer = torch.optim.SGD(model.linear.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

    
    decreasing_lr = list(map(int, args.decreasing_lr.split(',')))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)

    # if not os.path.exists(args.save_dir):
    #     os.mkdir(args.save_dir)

    prec1 = validate(val_loader, model, criterion)
    
    for epoch in range(args.epochs):
        print("The learning rate is {}".format(optimizer.param_groups[0]['lr']))
        # train for one epoch
        train(train_loader, model, criterion, criterion_graph, optimizer, epoch)
        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        scheduler.step()

        # # remember best prec@1 and save checkpoint
        # is_best = prec1 > best_prec1
        # best_prec1 = max(prec1, best_prec1)

        # if is_best:
        #     save_checkpoint({
        #         'epoch': epoch + 1,
        #         'state_dict': model.state_dict(),
        #         'best_prec1': best_prec1,
        #         'optimizer': optimizer,
        #     }, filename=os.path.join(args.save_dir, 'best_model.pt'))

        # save_checkpoint({
        #     'epoch': epoch + 1,
        #     'state_dict': model.state_dict(),
        #     'best_prec1': best_prec1,
        #     'optimizer': optimizer,
        # }, filename=os.path.join(args.save_dir, 'checkpoint.pt'))


def train(train_loader, model, criterion, criterion_graph, optimizer, epoch):
    
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()
    distance = []

    # if epoch%20 > 10:

    #     flag = 0
    #     for parm in model.features.parameters():
    #         parm.requires_grad = False

    #     for parm in model.classifier.parameters():
    #         parm.requires_grad = False 

    # else:
    #     flag = 1
    #     for parm in model.parameters():
    #         parm.requires_grad = True

    # print('flag = ', flag)

    for i, (input, target) in enumerate(train_loader):

        input = input.cuda()
        target = target.cuda()

        optimizer.zero_grad()
        output, graph_pre, graph_gt = model([input,target])
        acc = new_acc(graph_pre, graph_gt)
# 
        # print(acc)
        # pdb.set_trace()

        # if flag==0:
        loss = criterion_graph(graph_pre, graph_gt)

        # else:
        #     loss = criterion(output, target)
            

        # loss = criterion(output, target)+criterion_graph(graph_pre, graph_gt)
            
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), loss=losses, top1=top1))

    print('train_accuracy {top1.avg:.3f}'.format(top1=top1))

def validate(val_loader, model, criterion):

    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()

        # compute output
        with torch.no_grad():
            output,g1,g2 = model([input,target])
            loss = criterion(output, target)

        

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), loss=losses, top1=top1))
    
    aacc = new_acc(g1,g2)
    print(aacc)
    
    print('valid_accuracy {top1.avg:.3f}'.format(top1=top1))

    return top1.avg

def save_checkpoint(state, filename='weight.pt'):
    """
    Save the training model
    """
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def new_acc(output, target):
    batch_size=target.size(0)**2

    ones = torch.ones_like(output)
    zeros = torch.zeros_like(output)
    adj = torch.where(output>0.5, ones, zeros)
    
    adj = output
    target = target
    acc = torch.sum(adj == target).float()
    accuracy = (acc/batch_size)*100
    return accuracy.item()

def setup_seed(seed): 
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed) 
    random.seed(seed) 
    torch.backends.cudnn.deterministic = True 


if __name__ == '__main__':
    main()