import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def ori_smooth_l1_loss(ouptut,traget,bs):
    ## smooth l1 loss
    and1 = np.logical_and((ouptut) > 3.14, (traget) > 0)
    and2 = np.logical_and((ouptut) < -3.14, (traget) < 0)
    idx1, idx2 = np.where(np.logical_or(and1, and2))[0], \
                 np.where(np.logical_not(np.logical_or(and1, and2)))[0]
    # print idx1.shape,idx2.shape
    loss1 = traget[idx2] + ouptut[idx2]
    loss1 = torch.abs(loss1)
    loss1 = torch.sum(torch.where(loss1 < 1, 0.5 * loss1 ** 2, loss1 - 0.5))

    loss2 = traget[idx2] - ouptut[idx2]
    loss2 = torch.abs(loss2)
    loss2 = torch.sum(torch.where(loss2 < 1, 0.5 * loss2 ** 2, loss2 - 0.5))

    loss_l1 = torch.div(loss1 + loss2, bs)

    return loss_l1

def focal_loss(output,target,bs,alpha,gamma):
    ## focal loss
    eps = 1e-8
    pos_idx = np.where(target == 1)[0]
    neg_idx = np.where(target == 0)[0]
    loss_pos = -alpha * torch.sum(
        torch.pow(1.0 - output[pos_idx], gamma) * torch.log(output[pos_idx] + eps))
    loss_neg = -(1 - alpha) * torch.sum(
        torch.pow(output[neg_idx], gamma) * torch.log(1.0 - output[neg_idx] + eps))
    loss_focal = torch.div(loss_neg + loss_pos, bs)

    return loss_focal

def bce_loss(output,target,bs,alpha,gamma):
    #weight = torch.from_numpy(np.array([1.0 - alpha,alpha])).float().cuda()
    #neg_output = 1.0 - output
    #output = torch.stack([neg_output,output],1)

    weight = target*alpha + (1.0-target)*(1-alpha)
    loss = F.binary_cross_entropy(output,target,weight)

    #loss = -alpha*target*torch.log(output) - (1-alpha)*(1-target)*torch.log(1.0-output)
    #loss = torch.div(torch.sum(loss),bs)

    return loss

def l1_loss(output,target,bs):

    return F.smooth_l1_loss(output,target)



class Focal_L1_Loss(nn.Module):
    def __init__(self,alpha=0.1,gamma=2,lamda=0.5):
        super(Focal_L1_Loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.lamda = lamda

    def forward(self,output_b,output_o,label):
        """
        output_b: [N,1,H,W]
        output_o: [N,1,H,W]
        label: [N,2,H,W]
        """
        batch_size, _, height, width = label.size()
        label_b, label_o = label[:, 0], label[:, 1]
        output_b,output_o,label_b,label_o = output_b.contiguous().view(batch_size*height*width,),\
                                            output_o.contiguous().view(batch_size*height*width,),\
                                            label_b.contiguous().view(batch_size*height*width,),\
                                            label_o.contiguous().view(batch_size*height*width,)  ## [N*H*W,]
        # print label_b.size()
        # print torch.sum(label_b==1),torch.sum(label_b==0)
        # torch.nn.CrossEntropyLoss

        # loss_focal = focal_loss(output_b,label_b,batch_size,self.alpha,self.gamma)
        loss_focal = bce_loss(output_b,label_b,batch_size,0.9,self.gamma)
        loss_l1 = l1_loss(output_o,label_o.float(),batch_size)

        print loss_focal.item(),loss_l1.item()

        return  loss_focal + self.lamda * loss_l1



if __name__ == '__main__':
    N = 2
    H,W = 200,400
    label_o = torch.rand(N,1,H,W).float()
    label_b = torch.randint(0,2,size=(N,1,H,W)).float()
    label = torch.stack((label_b,label_o),1)

    o_b,o_o = torch.rand(N, 1, H, W),torch.rand(N, 1, H, W)

    orientation = Focal_L1_Loss()
    orientation(o_b,o_o,label)
