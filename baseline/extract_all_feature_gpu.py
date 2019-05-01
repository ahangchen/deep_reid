# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import os
import numpy as np
import scipy.io
from baseline.model import ft_net, ft_net_dense, PCB, PCB_test

######################################################################
# Options
# --------
from utils.file_helper import safe_mkdir, write_line

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir',default='/home/cwh/coding/dataset/sysu',type=str, help='./test_data')
parser.add_argument('--name', default='sysu_train', type=str, help='save model path')
parser.add_argument('--class_cnt', default=696, type=int, help='class count in training set')
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
parser.add_argument('--PCB', action='store_true', help='use PCB' )

opt = parser.parse_args()

str_ids = opt.gpu_ids.split(',')
#which_epoch = opt.which_epoch
name = opt.name
test_dir = opt.test_dir
target = test_dir.split('/')[-2]

gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >=0:
        gpu_ids.append(id)

# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])

######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
#
data_transforms = transforms.Compose([
        transforms.Resize((288,144), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
############### Ten Crop
        #transforms.TenCrop(224),
        #transforms.Lambda(lambda crops: torch.stack(
         #   [transforms.ToTensor()(crop)
          #      for crop in crops]
           # )),
        #transforms.Lambda(lambda crops: torch.stack(
         #   [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop)
          #       for crop in crops]
          # ))
])

if opt.PCB:
    data_transforms = transforms.Compose([
        transforms.Resize((384,192), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


data_dir = test_dir
image_dataset = datasets.ImageFolder( os.path.join(data_dir,'train_all') ,data_transforms)
dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=opt.batchsize,
                                             shuffle=False, num_workers=16)

class_names = image_dataset.classes
use_gpu = torch.cuda.is_available()

######################################################################
# Load model
#---------------------------
def load_network(network):
    save_path = os.path.join('./model',name,'net_%s.pth'%opt.which_epoch)
    network.load_state_dict(torch.load(save_path))
    return network


######################################################################
# Extract feature
# ----------------------
#
# Extract feature from  a trained model.
#
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_feature(model,dataloader):
    features = torch.FloatTensor()
    count = 0
    for data in dataloader:
        img, label = data
        n, c, h, w = img.size()
        count += n
        print(count)
        if opt.use_dense:
            ff = torch.FloatTensor(n,1024).zero_()
        else:
            ff = torch.FloatTensor(n,2048).zero_()
        if opt.PCB:
            ff = torch.FloatTensor(n,2048,6).zero_() # we have four parts
        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            outputs = model(input_img)
            f = outputs.data.cpu()
            ff = ff+f
        # norm feature
        if opt.PCB:
            # feature size (n,2048,4)
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)
        else:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features,ff), 0)
    return features

def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        filename = path.split('/')[-1]
        label = filename[0:4]
        if filename[-3:] == 'jpg':
            camera = int(filename.split('_')[1][1])
        else:
            camera = int(filename.split('_')[1])
        if label[0:2]=='-1' or label == '0000':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(camera)
    return camera_id, labels


def sort_by_score(qf,gf):
    query = qf.view(-1,1)
    # print(query.shape)
    score = torch.mm(gf,query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    # predict index
    index = np.argsort(score)  #from small to large
    index = index[::-1]
    score = score[index]
    return index, score

train_path = image_dataset.imgs


# train_cam, train_label = get_id(train_path)

######################################################################
# Load Collected data Trained model
print('-------test-----------')
if opt.use_dense:
    model_structure = ft_net_dense(opt.class_cnt)
else:
    model_structure = ft_net(opt.class_cnt)

if opt.PCB:
    model_structure = PCB(opt.class_cnt)

model = load_network(model_structure)

# Remove the final fc layer and classifier layer
if not opt.PCB:
    model.model.fc = nn.Sequential()
    model.classifier = nn.Sequential()
else:
    model = PCB_test(model)

# Change to test mode
model = model.eval()
if use_gpu:
    model = model.cuda()

# Extract feature
train_feature = extract_feature(model, dataloader)
print('train_feature shape', train_feature.shape)
transfer_name = opt.name + '_' + target + '-train'
safe_mkdir('eval')
safe_mkdir(os.path.join('eval', transfer_name))
result = {'ft': train_feature.numpy()}
name_log = os.path.join('eval', transfer_name, 'imgs.log')
for img_path in train_path:
    write_line(name_log, img_path[0].split('/')[-1])
scipy.io.savemat(os.path.join('eval', transfer_name, 'train_ft.mat'),result)