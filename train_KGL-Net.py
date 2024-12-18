#!/usr/bin/python3
# coding = gbk
"""
@Author : yuchuang
@Time : 2024/3/27 10:08
@desc:
"""
import ast
import argparse
from utils.utils import *
import torch.optim as optim
import os
import sys
import torch
import torch.nn as nn
import time
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


def train_net(desc_name, nb_batch_per_epoch):
    net.train()
    running_loss = 0.0
    running_dist_pos = 0.0
    running_dist_neg = 0.0

    run_desc_loss = 0.0
    run_metr_loss = 0.0
    run_cons_L_loss =  0.0
    run_cons_R_loss = 0.0

    for batch_loop in range(nb_batch_per_epoch):

        index_batch = index_train[epoch_loop][batch_loop]
        batch = patch_train[index_batch]
        batch = batch.to(torch.float32)
        if flag_dataAug:
            batch = data_aug4(batch, num_pt_per_batch)

        batch_1 = (batch[0] / 255).to(device)
        batch_2 = (batch[1] / 255).to(device)
        desc_L, desc_raw_L, out_com_L, out_dif_L, desc_R, desc_raw_R, out_com_R, out_dif_R = net(batch_1, batch_2, mode='train')


        if desc_name == 'HyNet':
            loss_des, dist_pos, dist_neg, diff_R_indice = loss_desc.compute(desc_L, desc_R, desc_raw_L, desc_raw_R)
            out_dif_L_shuffle, out_dif_R_shuffle, true_label_shuffle = make_metric_dataset(device, out_dif_L, out_dif_R, diff_R_indice, num_pt_per_batch)
            metric_s_out = net_metric(out_dif_L_shuffle, out_dif_R_shuffle)
            print(metric_s_out[0:10])
            #print(true_label_shuffle)
            loss_metr = loss_bce(metric_s_out, true_label_shuffle)

            loss_cons_L = consistency_loss_l2(out_com_L, out_dif_L)
            loss_cons_R = consistency_loss_l2(out_com_R, out_dif_R)

            loss = loss_des + loss_metr + loss_cons_L + loss_cons_R


        if desc_name == 'HyNet':
            running_loss = running_loss + loss.item()
            running_dist_pos += dist_pos.item()
            running_dist_neg += dist_neg.item()
            run_desc_loss = run_desc_loss + loss_des.item()
            run_metr_loss = run_metr_loss + loss_metr.item()
            run_cons_L_loss = run_cons_L_loss + loss_cons_L.item()
            run_cons_R_loss = run_cons_R_loss + loss_cons_R.item()


            print('epoch {}: {}/{}: dist_pos: {:.4f}, dist_neg: {:.4f}, loss: {:.4f}, desc_loss: {:.4f}, metr_loss: {:.8f}, cons_L_loss: {:.4f}, cons_R_loss: {:.4f}'.format(
                epoch_loop + 1,
                batch_loop + 1,
                nb_batch_per_epoch,
                running_dist_pos / (batch_loop + 1),
                running_dist_neg / (batch_loop + 1),
                running_loss / (batch_loop + 1),
                run_desc_loss / (batch_loop + 1),
                run_metr_loss / (batch_loop + 1),
                run_cons_L_loss / (batch_loop + 1),
                run_cons_R_loss / (batch_loop + 1),
            )),
        else:
            print("输入的描述子有误！！！！")
            sys.exit(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return



def test_net(device, net, net_metric, patch, pointID, index, dim_desc=128, sz_batch=256):
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
    return fpr95_desc, fpr95_metric


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
#'lwir_train': VIS-LWIR patch dataset, 'country': VIS-NIR patch dataset, 'os_train': OS patch dataset
parser.add_argument('--train_set', type=str, default='os_train')  # choose in ['lwir_train', 'country', 'os_train']
parser.add_argument('--train_split', type=str, default='a')  # full

parser.add_argument('--suffix', type=str, default='')
parser.add_argument('--sz_patch', type=int, default=64)
parser.add_argument('--num_pt_per_batch', type=int, default=256)   #default: 256
parser.add_argument('--dim_desc', type=int, default=128)
parser.add_argument('--nb_pat_per_pt', type=int, default=2)
parser.add_argument('--epoch_max', type=int, default=200)  ###For OS path dataset, epoch_max=200. For VIS-NIR patch dataset and VIS-LWIR patch dataset, epoch_max=100
parser.add_argument('--margin', type=float, default=1.2)
parser.add_argument('--test_batch', type=int, default=512)

parser.add_argument('--flag_dataAug', type=ast.literal_eval, default=True)
parser.add_argument('--is_sosr', type=ast.literal_eval, default=False)
parser.add_argument('--knn_sos', type=int, default=8)

parser.add_argument('--optim_method', type=str, default='Adam')
parser.add_argument('--lr_scheduler', type=str, default='None')  # CosineAnnealing  CosineAnnealingWarmRestarts None

parser.add_argument('--desc_name', type=str, default='HyNet')
parser.add_argument('--alpha', type=float, default=2)

parser.add_argument('--lr', type=float, default=5e-3)
parser.add_argument('--drop_rate', type=float, default=0.3)

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


# summary(net, [(512, 1, 64, 64), (512, 1, 64, 64)])
# summary(net_metric, [(512, 256, 8, 8), (512, 256, 8, 8)])


if optim_method == 'Adam':
    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
    optimizer = optim.Adam([{'params': net.parameters()}, {'params': net_metric.parameters(), 'lr': 5e-5}], lr = lr)

elif optim_method == 'SGD':
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9)

# if lr_scheduler == 'CosineAnnealing':
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
#                                                            T_max=epoch_max,
#                                                            eta_min=1e-6,
#                                                            last_epoch=-1)

if lr_scheduler == 'CosineAnnealing':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                           T_max=int(epoch_max/4),
                                                           eta_min=1e-6,
                                                           last_epoch=-1)

if lr_scheduler == 'CosineAnnealingWarmRestarts':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2, eta_min=lr*0.01, last_epoch=-1, verbose=False)





# data preparation
if train_set == 'country':
    patch_train, pointID_train, index_train = load_visnir_for_train(data_root, train_set,
                                                                    sz_patch,
                                                                    num_pt_per_batch, nb_pat_per_pt,
                                                                    epoch_max)
    test_set = ['field', 'forest', 'indoor', 'mountain', 'oldbuilding', 'street', 'urban', 'water']
    # test_set = ['field']

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




# file names
file_fpr95 = 'fpr95_out'

file_fpr95_best_name = file_fpr95 + '_best.npy'
file_fpr95_name = file_fpr95 + '.npy'
file_fpr95_path = os.path.join(net_dir, file_fpr95_name)
file_fpr95_best_path = os.path.join(net_dir, file_fpr95_best_name)
print(file_fpr95_path)
print(file_fpr95_best_path)

file_fpr95_best_name_metric = file_fpr95 + '_metric_best.npy'
file_fpr95_name_metric = file_fpr95 + '_metric.npy'
file_fpr95_path_metric = os.path.join(net_dir, file_fpr95_name_metric)
file_fpr95_best_path_metric = os.path.join(net_dir, file_fpr95_best_name_metric)


net_best_desc_name_path = os.path.join(net_dir, 'net-best-desc.pth')
#print(net_best_desc_name_path)
net_best_desc_corr_metric_name_path = os.path.join(net_dir, 'net-best-desc-corr-metric.pth')

net_best_metric_name_path = os.path.join(net_dir, 'net-best-metric.pth')
#print(net_best_metric_name_path)
net_best_metric_corr_desc_name_path = os.path.join(net_dir, 'net-best-metric-corr-desc.pth')


# descriptor type

file_txt_name = demo_name + '.txt'
file_txt_out_path = os.path.join(net_dir, file_txt_name)

time1 = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
log_name = "log" + time1 + '_' + demo_name + ".txt"
log_path = os.path.join(net_dir, log_name)
sys.stdout = Logger(os.path.join(net_dir, log_path))


if desc_name == 'HyNet':
    loss_desc = Loss_HyNet_metric(device, num_pt_per_batch, dim_desc, margin, alpha, is_sosr, knn_sos)

loss_bce = nn.BCELoss()

# start training
fpr95_desc = []
fpr95_metric = []
fpr_avg_best = 10000
fpr_avg_best_metric = 10000

start_time = time.time()

for epoch_loop in range(args.epoch_max):
    # train
    train_net(desc_name, nb_batch_per_epoch)

    if lr_scheduler != 'None':
        scheduler.step()

    net_name = os.path.join(net_dir, 'net-epoch-{}.pth'.format(epoch_loop + 1))
    #torch.save(net.state_dict(), net_name)
    # validation
    desc_fpr95_per_epoch = []
    metric_fpr95_per_epoch = []
    for i, val in enumerate(test_set):
        print(val)
        test_out_desc_list, test_out_metric_list = test_net(device, net, net_metric, patch_test[val], pointID_test[val], index_test[val], args.dim_desc, sz_batch=test_batch)
        desc_fpr95_per_epoch.append(test_out_desc_list)
        metric_fpr95_per_epoch.append(test_out_metric_list)
    if len(desc_fpr95_per_epoch) > 0:
        fpr95_desc.append(desc_fpr95_per_epoch)
        np.save(file_fpr95_path, fpr95_desc)
        fpr_avg = np.mean(np.array(desc_fpr95_per_epoch))
        if fpr_avg_best > fpr_avg:
            fpr_avg_best = fpr_avg
            fpr_best = desc_fpr95_per_epoch.copy()
            fpr_best.append(epoch_loop + 1)
            fpr_best.append(fpr_avg_best)
            np.save(file_fpr95_best_path, fpr_best)
            torch.save(net.state_dict(), net_best_desc_name_path)
            torch.save(net_metric.state_dict(), net_best_desc_corr_metric_name_path)
            print("最好FPR95列表(最后两个值代表最好模型epoch和平均值-从1开始):", fpr_best)
            print("最好FPR95列表的平均值:", fpr_avg_best)
        else:
            if epoch_loop == 0:
                print("最好FPR95列表(最后两个值代表最好模型epoch和平均值-从1开始):", desc_fpr95_per_epoch)
                print("最好FPR95列表的平均值:", fpr_avg)
            else:
                print("最好FPR95列表(最后两个值代表最好模型epoch和平均值-从1开始):", fpr_best)
                print("最好FPR95列表的平均值:", fpr_avg_best)

    if len(metric_fpr95_per_epoch) > 0:
        fpr95_metric.append(metric_fpr95_per_epoch)
        np.save(file_fpr95_path_metric, fpr95_metric)
        fpr_avg_metric = np.mean(np.array(metric_fpr95_per_epoch))
        if fpr_avg_best_metric > fpr_avg_metric:
            fpr_avg_best_metric = fpr_avg_metric
            fpr_best_metric = metric_fpr95_per_epoch.copy()
            fpr_best_metric.append(epoch_loop + 1)
            fpr_best_metric.append(fpr_avg_best_metric)
            np.save(file_fpr95_best_path_metric, fpr_best_metric)
            torch.save(net_metric.state_dict(), net_best_metric_name_path)
            torch.save(net.state_dict(), net_best_metric_corr_desc_name_path)
            print("最好FPR95_metric列表(最后两个值代表最好模型epoch和平均值-从1开始):", fpr_best_metric)
            print("最好FPR95_metric列表的平均值:", fpr_avg_best_metric)
        else:
            if epoch_loop == 0:
                print("最好FPR95_metric列表(最后两个值代表最好模型epoch和平均值-从1开始):", metric_fpr95_per_epoch)
                print("最好FPR95_metric列表的平均值:", fpr_avg_metric)
            else:
                print("最好FPR95_metric列表(最后两个值代表最好模型epoch和平均值-从1开始):", fpr_best_metric)
                print("最好FPR95_metric列表的平均值:", fpr_avg_best_metric)



end_time = time.time()
print("Time(h):", (end_time-start_time)/3600)



file_txt = open(file_txt_out_path, 'w')
for i in range(len(fpr95_desc)):
    file_txt.write(str(i) + ': ' + str(fpr95_desc[i]))
    file_txt.write('\n')
file_txt.write("--------------------------------------")
file_txt.write('\n')
file_txt.write('\n')
file_txt.write('fpr_best:' + str(fpr_best))
file_txt.write('\n')
file_txt.write('\n')
file_txt.write('\n')
file_txt.write('\n')

for i in range(len(fpr95_metric)):
    file_txt.write(str(i) + ': ' + str(fpr95_metric[i]))
    file_txt.write('\n')
file_txt.write("--------------------------------------")
file_txt.write('\n')
file_txt.write('\n')
file_txt.write('fpr_best_metric:' + str(fpr_best_metric))
file_txt.write('\n')
file_txt.write('\n')
file_txt.write('\n')
file_txt.write('\n')
file_txt.close()

print("Done!!!!!")
print("Done!!!!!")
print("Done!!!!!")
