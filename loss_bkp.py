import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PI = 3.141592654

## orient loss

def ori_smooth_l1_loss(output, target, norm, weight):
    sigma = 3.0

    and1 = np.logical_and((output) > PI, (target) > 0)
    and2 = np.logical_and((output) < -PI, (target) < 0)

    idx1, idx2 = np.logical_or(and1, and2).cuda(), \
                 np.logical_not(np.logical_or(and1, and2)).cuda()

    loss1 = torch.abs(idx1.float() * (target + output)) ## add
    loss2 = torch.abs(idx2.float() * (target - output)) ## sub
    diff = loss1 + loss2

    loss = torch.where(diff < 1.0/sigma, 0.5 * (diff*diff*sigma), diff - 0.5/sigma)
    loss = loss * weight

    #loss_l1 = torch.sum(loss)/norm *1.0
    #print loss.size()
    loss_l1 = torch.mean(loss)

    return loss_l1

## boundary loss

def bce_loss(output,target,norm,alpha):
    loss = -alpha*target*torch.log(output + 1e-8) - (1-alpha)*(1-target)*torch.log(1.0-output + 1e-8)
    #loss = torch.sum(loss)/norm*1.0
    loss = torch.mean(loss)

    return loss

def focal_loss(output,target,norm,alpha):
    loss = -alpha * target * torch.pow(1.0 - output,2) * torch.log(output + 1e-8) - \
            (1.0 - alpha) * (1.0 - target) * torch.pow(output,2) * torch.log(1.0 - output + 1e-8)
    loss_focal = torch.sum(loss)/norm*1.0

    return loss_focal

def attentional_focal_loss(output,target,norm,alpha):
    loss = -alpha * target * 4**((1.0 - output)**0.5) * torch.log(output + 1e-8) - \
           (1.0 - alpha) * (1.0 - target) * 4**(output**0.5) * torch.log(1.0 - output + 1e-8)
    #loss_focal = torch.sum(loss)/norm*1.0
    loss_focal = torch.mean(loss)

    return loss_focal





class Focal_L1_Loss(nn.Module):
    def __init__(self,gamma=2,lamda=0.5):
        super(Focal_L1_Loss, self).__init__()
        self.gamma = gamma
        self.lamda = lamda

    def forward(self,output_b,output_o,label):
        """
        output_b: [N,1,H,W]
        output_o: [N,1,H,W]
        label: [N,2,H,W]
        """
        batch_size, channel, height, width = output_b.size()
        label_b, label_o = label[:, 0], label[:, 1]
        output_b,output_o,label_b,label_o = output_b.contiguous().view(batch_size*height*width,),\
                                            output_o.contiguous().view(batch_size*height*width,),\
                                            label_b.contiguous().view(batch_size*height*width,),\
                                            label_o.contiguous().view(batch_size*height*width,)  ## [N*H*W,]

        num_pos,num_neg = torch.sum(label_b==1).float(),torch.sum(label_b==0).float()
        alpha = num_neg/(num_pos+num_neg)*1.0
        #print alpha
        #alpha = 0.9
        #print num_pos.item(),num_neg.item()

        norm = float(batch_size*channel*height*width)
        loss_focal = attentional_focal_loss(output_b.float(),label_b.float(),norm,alpha)
        loss_l1 = ori_smooth_l1_loss(output_o.float(),label_o.float(),norm,label_b.float())

        print loss_focal.item(),loss_l1.item()

        return  loss_focal + self.lamda * loss_l1



if __name__ == '__main__':
    from torch.autograd import Variable

    b = np.array([2, 4,-4, 1, -4, -2, -2, 1])
    a = np.array([1, 2, -3, 4, -1, -3, -3,-4])

    a, b = torch.from_numpy(a).float(), torch.from_numpy(b).float()
    a, b = Variable(a, requires_grad=True), Variable(b)


