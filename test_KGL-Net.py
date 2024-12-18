#!/usr/bin/python3
# coding = gbk
"""
@Author : yuchuang
@Time :
@desc: test
"""
import ast
import argparse
import numpy as np

from utils.utils import *
import torch.optim as optim
import os
import sys
import torch.nn as nn
import torch
from torchinfo import summary
import torch.nn as nn
from torch.nn import init
import time
import cv2
from model.KGLNet import feature_KGL, feature_KGL_metric

eps_fea_norm = 1e-5
eps_l2_norm = 1e-10

class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass




def test_net_out_img(device, net, net_metric, patch, pointID, index, dim_desc=128, sz_batch=256):
    print("测试的batch size:", sz_batch)
    net.eval()
    net_metric.eval()
    nb_patch = pointID.size
    nb_loop = int(np.ceil(nb_patch/sz_batch))
    desc = torch.zeros(nb_patch, dim_desc)
    metr = torch.ones((nb_patch//2,))*-1
    with torch.set_grad_enabled(False):
        for i in range(nb_loop):
            st = i * sz_batch
            en = np.min([(i + 1) * sz_batch, nb_patch])
            batch = (patch[st:en]/255).to(device)
            #out_desc_1, out_desc_2 = net(batch[0::2], batch[1::2], mode='eval')
            out_desc_1, out_com_1, out_dif_1, out_desc_2, out_com_2, out_dif_2 = net(batch[0::2], batch[1::2], mode='eval')
            metric_out = net_metric(out_dif_1, out_dif_2)
            #out_desc = out_desc.to('cpu')
            count_temp_1 = 0
            count_temp_2 = 0
            for j in range(st, en):
                if j%2==0:
                    desc[j] = out_desc_1[count_temp_1]
                    count_temp_1 = count_temp_1+1
                else:
                    desc[j] = out_desc_2[count_temp_2]
                    count_temp_2 = count_temp_2 + 1
            #print(': {} of {}'.format(i, nb_loop), end='\r')
            if i != nb_loop - 1:
                metr[i*sz_batch//2 : (i+1)*sz_batch//2] = metric_out
            else:
                metr[i*sz_batch//2 : nb_patch//2] = metric_out

    contains_minus_one = torch.any(metr == -1).item()
    if contains_minus_one:
        print("错误！！！张量中包含值为 -1")
        sys.exit(1)
    test_true_label = cal_test_true_label(pointID, index)
    fpr95_desc = cal_fpr95(desc, pointID, index)
    #print(test_true_label)
    #print(metr)
    fpr95_metric = cal_fpr95_metric(test_true_label, metr)
    return fpr95_desc, fpr95_metric, test_true_label, metr



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# ----------------获取当前运行文件的文件名------------------#
file_path_in = os.path.abspath(sys.argv[0])
file_name_in = os.path.basename(file_path_in)
file_name_without_ext = os.path.splitext(file_name_in)[0]
# # ------------------------------------------------------#
demo_name = file_name_without_ext


root_path = os.path.abspath('.')
mydata_path = os.path.join(root_path, 'mydata')
parser = argparse.ArgumentParser(description='pyTorch descNet')
parser.add_argument('--data_root', type=str, default=mydata_path)
parser.add_argument('--network_root', type=str, default=root_path)

parser.add_argument('--train_set', type=str, default='os_train')    # choose in ['lwir_train', 'country', 'os_train']
parser.add_argument('--train_split', type=str, default='a')

parser.add_argument('--suffix', type=str, default='')
parser.add_argument('--sz_patch', type=int, default=64)
parser.add_argument('--num_pt_per_batch', type=int, default=256)
parser.add_argument('--dim_desc', type=int, default=128)
parser.add_argument('--nb_pat_per_pt', type=int, default=2)
parser.add_argument('--epoch_max', type=int, default=200)  ###For OS path dataset, epoch_max=200. For VIS-NIR patch dataset and VIS-LWIR patch dataset, epoch_max=100
parser.add_argument('--margin', type=float, default=1.2)
parser.add_argument('--test_batch', type=int, default=512)

parser.add_argument('--flag_dataAug', type=ast.literal_eval, default=True)
parser.add_argument('--is_sosr', type=ast.literal_eval, default=False)
parser.add_argument('--knn_sos', type=int, default=8)

parser.add_argument('--optim_method', type=str, default='Adam')
parser.add_argument('--lr_scheduler', type=str, default='None')  # CosineAnnealing  None

parser.add_argument('--desc_name', type=str, default='HyNet')
parser.add_argument('--alpha', type=float, default=2)

parser.add_argument('--lr', type=float, default=5e-3)
parser.add_argument('--drop_rate', type=float, default=0.3)

parser.add_argument('--train_out_fold_name', type=str, default='None')   ### Change to the folder where the model needs to be tested

args = parser.parse_args()

data_root = args.data_root
train_set = args.train_set
sz_patch = args.sz_patch
epoch_max = args.epoch_max
num_pt_per_batch = args.num_pt_per_batch
nb_pat_per_pt = args.nb_pat_per_pt
num_pt_per_batch = args.num_pt_per_batch
dim_desc = args.dim_desc
margin = args.margin
test_batch = args.test_batch
drop_rate = args.drop_rate
is_sosr = args.is_sosr
knn_sos = args.knn_sos
flag_dataAug = args.flag_dataAug
optim_method = args.optim_method
lr_scheduler = args.lr_scheduler
alpha = args.alpha
desc_name = args.desc_name
train_split = args.train_split
lr = args.lr
train_out_fold_name = args.train_out_fold_name

# get save folder name
folder_name = demo_name + '_' + desc_name + '_' + train_set

if train_set == 'hpatches':
    folder_name += '_split_' + train_split

folder_name += '_epochs_' + str(epoch_max)
folder_name += '_sz_' + str(sz_patch)
folder_name += '_pt_' + str(num_pt_per_batch)
folder_name += '_pat_' + str(nb_pat_per_pt)
folder_name += '_dim_' + str(dim_desc)

if args.is_sosr or args.desc_name == 'SOSNet':
    folder_name += '_SOSR' + '_KNN_' + str(knn_sos)

if args.desc_name == 'HyNet':
    folder_name += '_alpha_' + str(alpha)

folder_name += '_margin_' + str(margin).replace('.', '_')
folder_name += '_drop_' + str(drop_rate).replace('.', '_')
folder_name += '_lr_' + str(lr).replace('.', '_')
folder_name += '_' + optim_method + '_' + lr_scheduler

if flag_dataAug:
    folder_name += '_aug'

if len(args.suffix) > 0:  # for debugging
    folder_name += '-' + args.suffix

net_dir = os.path.join(args.network_root, 'network', folder_name)
print(net_dir)

if not os.path.exists(net_dir):
    os.makedirs(net_dir)
else:
    print('path already exists')

# model, optimizer
net = feature_KGL(dim_desc=dim_desc, drop_rate=drop_rate)
net.to(device)
net_metric = feature_KGL_metric(drop_rate=drop_rate)
net_metric.to(device)


# data preparation
if train_set == 'country':
    patch_train, pointID_train, index_train = load_visnir_for_train(data_root, train_set,
                                                                    sz_patch,
                                                                    num_pt_per_batch, nb_pat_per_pt,
                                                                    epoch_max)
    test_set = ['field', 'forest', 'indoor', 'mountain', 'oldbuilding', 'street', 'urban', 'water']


elif train_set == 'lwir_train':
    patch_train, pointID_train, index_train = load_visnir_for_train_new(data_root, train_set,
                                                                    sz_patch,
                                                                    num_pt_per_batch, nb_pat_per_pt,
                                                                    epoch_max)
    test_set = ['lwir_test']


elif train_set == 'os_train':
    patch_train, pointID_train, index_train = load_visnir_for_train_new(data_root, train_set,
                                                                    sz_patch,
                                                                    num_pt_per_batch, nb_pat_per_pt,
                                                                    epoch_max)
    test_set = ['os_test']

else:
    print("请输入正确的数据集名！！！！")
    sys.exit(0)

nb_batch_per_epoch = len(index_train[0])  # Each epoch has equal number of batches

patch_test = {}
pointID_test = {}
index_test = {}



for i, val in enumerate(test_set):
    patch_test[val], pointID_test[val], index_test[val] = load_visnir_for_test(args.data_root, val, args.sz_patch)
    patch_test[val] = torch.from_numpy(patch_test[val])
    patch_test[val] = patch_test[val].to(torch.float32)
    index_test[val] = index_test[val]



file_txt_name = demo_name + '.txt'
file_txt_out_path = os.path.join(net_dir, file_txt_name)

time1 = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
log_name = "log" + time1 + '_' + demo_name + ".txt"
log_path = os.path.join(net_dir, log_name)
sys.stdout = Logger(os.path.join(net_dir, log_path))



start_time = time.time()



####测试部分
#train_out_fold_name = 'des3_30_32_HyNet_country_epochs_100_sz_64_pt_256_pat_2_dim_128_alpha_2_margin_1_2_drop_0_3_lr_0_005_Adam_None_aug'   ### Change to the folder where the model needs to be tested
net_model_name = os.path.join(root_path,'network',train_out_fold_name,'net-best-metric-corr-desc.pth')
net_metric_model_name = os.path.join(root_path,'network',train_out_fold_name,'net-best-metric.pth')
# net_model_name = os.path.join(root_path,'net-best-metric-corr-desc.pth')
# net_metric_model_name = os.path.join(root_path,'net-best-metric.pth')
net.load_state_dict(torch.load(net_model_name, map_location=device))
net_metric.load_state_dict(torch.load(net_metric_model_name, map_location=device))

test_out_desc_list = []
test_out_metric_list = []

for i, val in enumerate(test_set):
    print(val)
    test_out_desc, test_out_metric, test_true_label, metr_out = test_net_out_img(device, net, net_metric, patch_test[val], pointID_test[val],
                                                        index_test[val], args.dim_desc, sz_batch=test_batch)
    test_out_desc_list.append(test_out_desc)
    test_out_metric_list.append(test_out_metric)
    print(test_out_desc_list)
    print(test_out_metric_list)



    ###################################
    ###保存图像
    save_img_sub_dir_path = os.path.join(root_path,'network',folder_name,val)
    make_dir(save_img_sub_dir_path)
    metr_out_01 = np.where(metr_out>0.5,1,0)
    deal_patch = patch_test[val].cpu().numpy()
    test_true_label_01 = test_true_label.cpu().numpy()
    num_11 = 0
    num_10 = 0
    num_00 = 0
    num_01 = 0
    save_img_sub_dir_path_11 = os.path.join(save_img_sub_dir_path,'true_1_test_1')
    save_img_sub_dir_path_10 = os.path.join(save_img_sub_dir_path, 'true_1_test_0')
    save_img_sub_dir_path_00 = os.path.join(save_img_sub_dir_path, 'true_0_test_0')
    save_img_sub_dir_path_01 = os.path.join(save_img_sub_dir_path, 'true_0_test_1')
    make_dir(save_img_sub_dir_path_11)
    make_dir(save_img_sub_dir_path_10)
    make_dir(save_img_sub_dir_path_00)
    make_dir(save_img_sub_dir_path_01)

    for j in range(int(len(deal_patch)/2)):

        out_patch_pair = np.zeros((64, 130))
        vis_patch_out = deal_patch[2 * j, 0, :, :]
        nir_patch_out = deal_patch[2 * j + 1, 0, :, :]
        out_patch_pair[:, 0:64] = vis_patch_out
        out_patch_pair[:, 66:130] = nir_patch_out
        out_patch_pair_name = str(j)+'.png'

        if int(test_true_label_01[j])==1 and int(metr_out_01[j])==1:
            num_11 = num_11 + 1
            img_pair_final_out_path = os.path.join(save_img_sub_dir_path_11,out_patch_pair_name)
            cv2.imwrite(img_pair_final_out_path, out_patch_pair)

        elif int(test_true_label_01[j])==1 and int(metr_out_01[j])==0:
            num_10 = num_10 + 1
            img_pair_final_out_path = os.path.join(save_img_sub_dir_path_10, out_patch_pair_name)
            cv2.imwrite(img_pair_final_out_path, out_patch_pair)

        elif int(test_true_label_01[j])==0 and int(metr_out_01[j])==0:
            num_00 = num_00 + 1
            img_pair_final_out_path = os.path.join(save_img_sub_dir_path_00, out_patch_pair_name)
            cv2.imwrite(img_pair_final_out_path, out_patch_pair)

        elif int(test_true_label_01[j])==0 and int(metr_out_01[j])==1:
            num_01 = num_01 + 1
            img_pair_final_out_path = os.path.join(save_img_sub_dir_path_01, out_patch_pair_name)
            cv2.imwrite(img_pair_final_out_path, out_patch_pair)

        else:
            print("标签存在错误，请检查！！！！")
            sys.exit(0)

    print("num_11", num_11)
    print("num_10", num_10)
    print("num_00", num_00)
    print("num_01", num_01)
    #################################


print("描述子网络输出：", test_out_desc_list)
print("描述子所有子集平均结果：", np.mean(test_out_desc_list))
print("度量网络输出：",test_out_metric_list)
print("度量网络所有子集平均结果:", np.mean(test_out_metric_list))

time.sleep(60)
print("Done!!!!!")
print("Done!!!!!")
print("Done!!!!!")



