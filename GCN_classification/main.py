import argparse
import os
import torch
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


parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--lr', default=0.005, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=3e-4, type=float, help='weight decay')
parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
parser.add_argument('--print_freq', default=50, type=int, help='print frequency')


parser.add_argument('--save_dir', help='The directory used to save the trained models', default='model', type=str)
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--threshold', type=float, default=500, help='gpu device id')


best_prec1 = 0

def main():
    global args, best_prec1
    args = parser.parse_args()
    print(args)

    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
    torch.cuda.set_device(int(args.gpu))

    # model = net.VGG16()
    model = net.VGG16_GCN(args.threshold)
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
    criterion = criterion.cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

    # if not os.path.exists(args.save_dir):
    #     os.mkdir(args.save_dir)
    
    for epoch in range(args.epochs):
        print("The learning rate is {}".format(optimizer.param_groups[0]['lr']))
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)
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


def train(train_loader, model, criterion, optimizer, epoch):
    
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()
    distance = []

    if epoch > 15:
        model.threshold = args.threshold
    else:
        model.threshold = 0


    for i, (input, target) in enumerate(train_loader):

        input = input.cuda()
        target = target.cuda()

        optimizer.zero_grad()
        output = model(input)

        loss = criterion(output, target)
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
            output = model(input)
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

if __name__ == '__main__':
    main()