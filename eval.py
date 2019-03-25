import os
import cv2
import time
import torch
import argparse
import numpy as np
from PIL import Image
import scipy.io as sio
from torch.autograd import Variable
import torchvision.transforms as transforms
from net import DoobNet

use_GPU = torch.cuda.is_available()
torch.cuda.set_device(1)
parser = argparse.ArgumentParser()
parser.add_argument('--resume', default='/home/yuzhe/1_checkpoint.pth.tar', type=str, help='model file')
parser.add_argument('--data', default='', type=str, help='root path to data')
parser.add_argument('--result', default='', type=str, help='result .mat path')
args = parser.parse_args()
print args

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
trans = transforms.Compose([transforms.ToTensor(),
                            normalize])

t1 = time.time()
model = DoobNet()
if use_GPU:
    model.cuda()
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['state_dict'])
    """
    state_dict = checkpoint['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] ## remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)"""
else:
    checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])


def read_img(img_filename):
    with open(img_filename, 'rb') as f:
        img = Image.open(f)
        img = img.convert('RGB')
    img_tensor = trans(img)
    img_tensor = torch.unsqueeze(img_tensor,0)

    return img_tensor

datapath = args.data
test_lst = []
with open(os.path.join(datapath,'val_doc_2010.txt')) as f:
    lines = f.readlines()
    for line in lines:
        filename = line.rstrip()
        test_lst.append(os.path.join(datapath,'Augmentation','Aug_JPEGImages',filename) + '.jpg')

print "test img num: {}".format(len(test_lst))

for idx,img in enumerate(test_lst):
    print img
    img_tensor = read_img(img)
    if use_GPU:
        imgs = Variable(img_tensor).cuda()
    else:
        imgs = Variable(img_tensor)

    output_b, output_o = model(imgs) ## [1,1,H,W]
    output_b = torch.sigmoid(output_b)

    edgemap,orimap = np.squeeze(output_b.cpu().detach().numpy()),\
                     np.squeeze(output_o.cpu().detach().numpy())

    edge_ori = {}
    edge_ori['edge'] = edgemap
    edge_ori['ori'] = orimap  ##[H,W]
    #print edgemap.shape,orimap.shape

    ## save to .mat
    #img_id = os.path.split()
    #save_file = os.path.join(args.result,)
    cv2.imwrite(args.result + '/' + os.path.split(img)[1].split('.')[0] + '.png', edgemap*255)
    sio.savemat(args.result + '/' + os.path.split(img)[1].split('.')[0] + '.mat',{'edge_ori':edge_ori})
t2 = time.time()
print "total time: {}s".format(t2-t1)





