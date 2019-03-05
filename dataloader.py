import os
import glob
import h5py
import torch
import random
import numpy as np
from PIL import Image,ImageStat
from torch.utils import data
from torchvision import transforms,datasets

def get_subwindow(im, label, center_pos, original_sz, avg_chans):
    """
     img
     pos: center
     original_sz: crop patch size = 320
    """
    if isinstance(center_pos, float):
        center_pos = [center_pos, center_pos]
    sz = original_sz
    im_sz = im.shape ## H,W
    c = (original_sz+1) / 2 # 320/2 = 160

    context_xmin = round(center_pos[0] - c)  # floor(pos(2) - sz(2) / 2);
    context_xmax = context_xmin + sz - 1
    context_ymin = round(center_pos[1] - c)  # floor(pos(1) - sz(1) / 2);
    context_ymax = context_ymin + sz - 1

    left_pad = int(max(0., -context_xmin))
    top_pad = int(max(0., -context_ymin))
    right_pad = int(max(0., context_xmax - im_sz[1] + 1))
    bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

    ## for example, if context_ymin<0, now context_ymin = 0
    context_xmin = context_xmin + left_pad
    context_xmax = context_xmax + left_pad
    context_ymin = context_ymin + top_pad
    context_ymax = context_ymax + top_pad

    # zzp: a more easy speed version
    r, c, k = im.shape
    # avg_chans = np.array(avg_chans).reshape(3,)
    if any([top_pad, bottom_pad, left_pad, right_pad]):
        te_im = np.zeros((r + top_pad + bottom_pad, c + left_pad + right_pad, k), np.uint8)  # 0 is better than 1 initialization
        te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im

        if top_pad:
            te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
        if bottom_pad:
            te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
        if left_pad:
            te_im[:, 0:left_pad, :] = avg_chans
        if right_pad:
            te_im[:, c + left_pad:, :] = avg_chans
        im_patch_original = te_im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]
    else:
        im_patch_original = im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]

    r, c, k = label.shape
    avg_chans = np.array([0, 0]).reshape(2,)
    if any([top_pad, bottom_pad, left_pad, right_pad]):
        te_label = np.zeros((r + top_pad + bottom_pad, c + left_pad + right_pad, k), np.float32)  # 0 is better than 1 initialization
        te_label[top_pad:top_pad + r, left_pad:left_pad + c, :] = label

        if top_pad:
            te_label[0:top_pad, left_pad:left_pad + c, :] = avg_chans
        if bottom_pad:
            te_label[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
        if left_pad:
            te_label[:, 0:left_pad, :] = avg_chans
        if right_pad:
            te_label[:, c + left_pad:, :] = avg_chans
        label_patch_original = te_label[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]
    else:
        label_patch_original = label[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]

    return im_patch_original,label_patch_original

class POID_dataset(data.Dataset):
    def __init__(self,root_path):
        self.img_path = os.path.join(root_path,'Augmentation','Aug_JPEGImages')
        self.label_path = os.path.join(root_path,'Augmentation','Aug_HDF5EdgeOriLabel')
        list_file = os.path.join(root_path,'Augmentation','train_pari_320x320.lst')
        with open(list_file,'r') as f:
            lines = f.readlines()
            for i,line in enumerate(lines):
                lines[i] = os.path.split(line.rstrip())[1]
        self.img_list = sorted(lines)
        ## single sata test
        #self.img_list = [self.img_list[0]]

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.trans = transforms.Compose([
                             transforms.ToTensor(),
                             normalize])  ## pre-process of pre-trained model of pytorch resnet-50
        ## ToTensor() need input (H,W,3)


    def __len__(self):
        return len(self.img_list)


    def __getitem__(self, idx):
        ## img data
        img_filename = os.path.join(self.img_path,self.img_list[idx])
        with open(img_filename, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        w,h = img.size
        img_center = np.array([h/2,w/2]).astype(np.int)
        img = np.array(img) ## (H,W,3) uint8 RGB
        avg_chans = np.mean(img, axis=(0, 1)) ##(3,)

        ## label data
        filename = self.img_list[idx][:-4]
        label_filename = os.path.join(self.label_path, filename + '.h5')
        h = h5py.File(label_filename, 'r')
        label = np.squeeze(h['label'][...])
        label = np.transpose(label,axes=(1,2,0)) ## (H,W,2)   0-edgemap 1-orientmap

        ##random crop 320x320
        offset_x,offset_y = 0,0
        offset = True
        if offset:
            offset_y = int(50*(random.random() - 0.5))
            offset_x = int(50*(random.random() - 0.5))
        img_center = [img_center[0]+offset_y,img_center[1]+offset_x]
        img_crop,label_crop = get_subwindow(img,label,img_center,320,avg_chans)
        check = False
        if check:
            a = Image.fromarray(img_crop)
            a.save('1.jpg')
            b = np.stack([label_crop[:, :, 0], label_crop[:, :, 0], label_crop[:, :, 0]], axis=2)
            print b.shape
            b = Image.fromarray((b * 255).astype(np.uint8))
            b.save('2.jpg')

        label_crop = np.transpose(label_crop,axes=(2,0,1)) ## (2,320,320)

        img_tensor = self.trans(img_crop)
        #img_crop = img_crop[:,:,[2,1,0]] - np.array([104.0,116.6,122.6]) ##(320,320,3)
        #img_crop = np.transpose(img_crop,axes=(2,0,1))
        #img_tensor = torch.from_numpy(img_crop).float()
        label = torch.from_numpy(label_crop).float()

        return img_tensor,label

if __name__ == '__main__':
    d = POID_dataset(root_path='/home/gyz/document3/data/PIOD_data/')
    print len(d)
    # for img in d:
    #     a = img
    #     exit()
    a = d[0]







