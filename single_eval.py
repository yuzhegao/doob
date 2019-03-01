import os
import time
import torch
import argparse
import numpy as np
from PIL import Image
import scipy.io as sio
from torch.autograd import Variable
import torchvision.transforms as transforms

from net import DoobNet
from dataloader import POID_dataset
from loss import Focal_L1_Loss


use_GPU = torch.cuda.is_available()

parser = argparse.ArgumentParser()
parser.add_argument('--resume', default='model/4_checkpoint.pth.tar', type=str, help='model file')
parser.add_argument('--data',
                    default='/home/gyz/document3/data/PIOD_data/Augmentation/Aug_JPEGImages/2008_000002.jpg', type=str, help='path to img')
args = parser.parse_args()
print args

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
trans = transforms.Compose([transforms.ToTensor(),
                            normalize])
def read_img(img_filename):
    with open(img_filename, 'rb') as f:
        img = Image.open(f)
        img = img.convert('RGB')
    #img_tensor = trans(img)
    img_tensor = torch.from_numpy(np.array(img).transpose(2,0,1)).float()
    img_tensor = torch.unsqueeze(img_tensor,0)

    return img_tensor

def vis_hist(output):
    import matplotlib.pyplot as plt
    plt.hist(output,bins=400, normed=0, facecolor="blue", edgecolor="black")
    plt.show()

def vis_edge(output_b):
    #bb = np.squeeze(output_b.detach().numpy())
    #vis_hist(bb.flatten())
    b = (np.squeeze(output_b.cpu().detach().numpy()))
    print np.max(b),np.min(b)
    b = np.stack([b, b, b], axis=2)
    print 'num of edge pixel ',np.sum(b)
    b = Image.fromarray((b * 255).astype(np.uint8))
    b.save('tmp/result.jpg')

def eval_ori(output_o):
    o = np.squeeze(output_o.detach().numpy())
    sio.savemat('result.mat', {})


model = DoobNet()
if use_GPU:
    model.cuda()
    checkpoint = torch.load(args.resume)
    #model.load_state_dict(checkpoint['state_dict'])
    state_dict = checkpoint['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
else:
    checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
    #model.load_state_dict(checkpoint['state_dict'])
    state_dict = checkpoint['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] ## remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

# print model.conv1_b[1][0].weight
# exit()

def eval():
    img_tensor = read_img(args.data)
    if use_GPU:
        imgs = Variable(img_tensor).cuda()
    else:
        imgs = Variable(img_tensor)

    output_b, output_o = model(imgs)
    print output_b.size(),output_o.size() ## (1,1,H,W)

    torch.set_printoptions(threshold='nan')
    # print output_b

    vis_edge(output_b)

    # print output_o
    print torch.sum(output_o > 1.57)


if __name__ == '__main__':
    eval()








