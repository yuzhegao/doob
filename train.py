import os
import time
import torch
import argparse
from torch.autograd import Variable
from net import DoobNet
from dataloader import POID_dataset
from loss import Focal_L1_Loss

use_GPU = torch.cuda.is_available()


parser = argparse.ArgumentParser()
parser.add_argument('--numEpochs', default=50, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', default=3e-5, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--workers', default=1, type=int, metavar='N', help='number of data loading workers ')
parser.add_argument('--weight-decay', default=0.0002, type=float, metavar='W', help='weight decay ')
parser.add_argument('--momentum', default=0.9, type=float, metavar='W', help='momentum ')
parser.add_argument('--batch-size', default=2, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument('--save', default='./model', type=str, help='directory for saving')
parser.add_argument('--log', default='./model/log.txt', type=str, help='logfile')
parser.add_argument('--resnet', default='', type=str, help='resnet model file')
parser.add_argument('--dataset', default='/home/yuzhe/Downloads/PIOD/PIOD', type=str, help='path to dataset')
args = parser.parse_args()
print args

train_data = POID_dataset(root_path=args.dataset)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                           num_workers=args.workers, pin_memory=True, drop_last=True)
num_iter = 0

model = DoobNet()
if args.resnet:
    model.load_resnet(args.resnet)
if use_GPU:
    print 'use GPU'
    model.cuda()
    torch.cuda.set_device(0)

criterion = Focal_L1_Loss()                 # define criterion

optimizer = torch.optim.SGD(model.parameters(),
                            lr=args.lr,
                            weight_decay=args.weight_decay,
                            momentum=args.momentum) # define optimizer

def save_checkpoint(state, epoch):
    filename = os.path.join(args.save, '{}_checkpoint.pth.tar'.format(epoch + 1))
    torch.save(state, filename)

def adjust_learning_rate(optimizer):
    global num_iter
    if num_iter%20000 ==0 and num_iter != 0:
        print "lr decay in iter {}".format(num_iter)
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
        with open(args.log,'a') as f:
            f.write("lr decay in iter {}\n".format(num_iter))


def train(epoch):
    global num_iter
    model.train()

    for idx, (imgs, labels) in enumerate(train_loader):
        if use_GPU:
            imgs = Variable(imgs).cuda()
            labels = Variable(labels).cuda()
        else:
            imgs = Variable(imgs)
            labels = Variable(labels)

        output_b, output_o = model(imgs)
        #print output_b.size()

        loss = criterion(output_b, output_o, labels)
        print 'in Epoch{}/iter{} loss={}'.format(epoch,idx,loss.item())
        if num_iter%100 == 0:
            with open(args.log, 'a') as f:
                f.write('in Epoch{}/iter{} loss={}\n'.format(epoch,idx,loss.item()))

        num_iter +=1
        if num_iter%20000 ==0 and num_iter != 0:
            print "lr decay in iter {}".format(num_iter)
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.1
            with open(args.log,'a') as f:
                f.write("lr decay in iter {}\n".format(num_iter))

        optimizer.zero_grad()
        loss.backward()
        """
        print model.conv10_b[-1][0].weight.grad
        print model.conv10_o[-1][0].weight.grad
        print model.conv1.weight.grad
        """
        optimizer.step()
        #exit()

    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, epoch)
    with open(args.log, 'a') as f:
        f.write('\nFinish this epoch in {}\n\n'.format(time.strftime('%Y.%m.%d %H:%M', time.localtime(time.time()))))


with open(args.log, 'a') as f:
    f.write('start training in {}\n\n'.format(time.strftime('%Y.%m.%d %H:%M',time.localtime(time.time()))))

for epoch in range(args.numEpochs):
    train(epoch)




