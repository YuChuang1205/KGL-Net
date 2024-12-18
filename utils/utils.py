import os
import numpy as np
import cv2
from tqdm import tqdm
import random
import json
from imgaug import augmenters as iaa
import torch
import torch.nn.functional as F
from sklearn import metrics
from scipy import interpolate

dist_th = 8e-3  # threshold from HardNet, negative descriptor pairs with the distances lower than this threshold are treated as false negatives
eps_sqrt = 1e-6




class Loss_HyNet_metric():

    def __init__(self, device, num_pt_per_batch, dim_desc, margin, alpha, is_sosr, knn_sos=8):
        self.device = device
        self.margin = margin
        self.alpha = alpha
        self.is_sosr = is_sosr
        self.num_pt_per_batch = num_pt_per_batch
        self.dim_desc = dim_desc
        self.knn_sos = knn_sos
        self.index_desc = torch.LongTensor(range(0, num_pt_per_batch))
        self.index_dim = torch.LongTensor(range(0, dim_desc))
        diagnal = torch.eye(num_pt_per_batch)
        self.mask_pos_pair = diagnal.eq(1).float().to(self.device)
        self.mask_neg_pair = diagnal.eq(0).float().to(self.device)

    def sort_distance(self):
        L = self.L.clone().detach()
        L = L + 2 * self.mask_pos_pair
        L = L + 2 * L.le(dist_th).float()

        R = self.R.clone().detach()
        R = R + 2 * self.mask_pos_pair
        R = R + 2 * R.le(dist_th).float()

        LR = self.LR.clone().detach()
        LR = LR + 2 * self.mask_pos_pair
        LR = LR + 2 * LR.le(dist_th).float()

        self.indice_L = torch.argsort(L, dim=1)
        self.indice_R = torch.argsort(R, dim=0)
        self.indice_LR = torch.argsort(LR, dim=1)
        self.indice_RL = torch.argsort(LR, dim=0)
        return

    def triplet_loss_hybrid(self):
        L = self.L
        R = self.R
        LR = self.LR
        indice_L = self.indice_L[:, 0]
        indice_R = self.indice_R[0, :]
        indice_LR = self.indice_LR[:, 0]
        diff_R_indice = indice_LR
        indice_RL = self.indice_RL[0, :]
        index_desc = self.index_desc

        dist_pos = LR[self.mask_pos_pair.bool()]
        dist_neg_LL = L[index_desc, indice_L]
        dist_neg_RR = R[indice_R, index_desc]
        dist_neg_LR = LR[index_desc, indice_LR]
        dist_neg_RL = LR[indice_RL, index_desc]
        dist_neg = torch.cat((dist_neg_LL.unsqueeze(0),
                              dist_neg_RR.unsqueeze(0),
                              dist_neg_LR.unsqueeze(0),
                              dist_neg_RL.unsqueeze(0)), dim=0)
        dist_neg_hard, index_neg_hard = torch.sort(dist_neg, dim=0)
        dist_neg_hard = dist_neg_hard[0, :]
        # scipy.io.savemat('dist.mat', dict(dist_pos=dist_pos.cpu().detach().numpy(), dist_neg=dist_neg_hard.cpu().detach().numpy()))

        loss_triplet = torch.clamp(self.margin + (dist_pos + dist_pos.pow(2) / 2 * self.alpha) - (
                    dist_neg_hard + dist_neg_hard.pow(2) / 2 * self.alpha), min=0.0)

        self.num_triplet_display = loss_triplet.gt(0).sum()

        self.loss = self.loss + loss_triplet.sum()
        self.dist_pos_display = dist_pos.detach().mean()
        self.dist_neg_display = dist_neg_hard.detach().mean()

        return diff_R_indice

    def norm_loss_pos(self):
        diff_norm = self.norm_L - self.norm_R
        self.loss += diff_norm.pow(2).sum().mul(0.1)

    def sos_loss(self):
        L = self.L
        R = self.R
        knn = self.knn_sos
        indice_L = self.indice_L[:, 0:knn]
        indice_R = self.indice_R[0:knn, :]
        indice_LR = self.indice_LR[:, 0:knn]
        indice_RL = self.indice_RL[0:knn, :]
        index_desc = self.index_desc
        num_pt_per_batch = self.num_pt_per_batch
        index_row = index_desc.unsqueeze(1).expand(-1, knn)
        index_col = index_desc.unsqueeze(0).expand(knn, -1)

        A_L = torch.zeros(num_pt_per_batch, num_pt_per_batch).to(self.device)
        A_R = torch.zeros(num_pt_per_batch, num_pt_per_batch).to(self.device)
        A_LR = torch.zeros(num_pt_per_batch, num_pt_per_batch).to(self.device)

        A_L[index_row, indice_L] = 1
        A_R[indice_R, index_col] = 1
        A_LR[index_row, indice_LR] = 1
        A_LR[indice_RL, index_col] = 1

        A_L = A_L + A_L.t()
        A_L = A_L.gt(0).float()
        A_R = A_R + A_R.t()
        A_R = A_R.gt(0).float()
        A_LR = A_LR + A_LR.t()
        A_LR = A_LR.gt(0).float()
        A = A_L + A_R + A_LR
        A = A.gt(0).float() * self.mask_neg_pair

        sturcture_dif = (L - R) * A
        self.loss = self.loss + sturcture_dif.pow(2).sum(dim=1).add(eps_sqrt).sqrt().sum()

        return

    def compute(self, desc_L, desc_R, desc_raw_L, desc_raw_R):
        self.desc_L = desc_L
        self.desc_R = desc_R
        self.desc_raw_L = desc_raw_L
        self.desc_raw_R = desc_raw_R
        self.norm_L = self.desc_raw_L.pow(2).sum(1).add(eps_sqrt).sqrt()
        self.norm_R = self.desc_raw_R.pow(2).sum(1).add(eps_sqrt).sqrt()
        self.L = cal_l2_distance_matrix(desc_L, desc_L)
        self.R = cal_l2_distance_matrix(desc_R, desc_R)
        self.LR = cal_l2_distance_matrix(desc_L, desc_R)

        self.loss = torch.Tensor([0]).to(self.device)

        self.sort_distance()
        diff_R_indice = self.triplet_loss_hybrid()
        self.norm_loss_pos()
        if self.is_sosr:
            self.sos_loss()

        return self.loss, self.dist_pos_display, self.dist_neg_display, diff_R_indice



def make_dir(path):
    if os.path.exists(path) == False:
        os.makedirs(path)


def cal_l2_distance_matrix(x, y, flag_sqrt=True):
    ''''distance matrix of x with respect to y, d_ij is the distance between x_i and y_j'''
    D = torch.abs(2 * (1 - torch.mm(x, y.t())))
    if flag_sqrt:
        D = torch.sqrt(D + eps_sqrt)
    return D


def read_visnir_pointID(train_root, name):
    pointID = []
    input_txt_name = name + '_info.txt'
    with open(os.path.join(train_root, input_txt_name)) as f:
        for line in f:
            id = int(line)
            pointID.append(id)
            print('reading pointID:id{}'.format(id), end='\r')
    print('max ID:{}'.format(id))
    return np.array(pointID)


def cal_index_train_visnir(index_unique_label, num_label_each_batch, num_img_each_label, epoch_max):
    print('calculating index_train...')
    # ensure input is numpy array
    index_train = []

    num_label = len(index_unique_label)
    num_patch = 0
    for i in range(num_label):
        num_patch += index_unique_label[i].size

    index_index = [i for i in range(num_label)]  # for random shuffule

    index_unique_label0 = index_unique_label.copy()

    sz_batch = num_img_each_label * num_label_each_batch
    num_batch_each_epoch = int(num_patch / sz_batch)
    for e_loop in range(epoch_max):
        # loop over each epoch
        each_epoch_index = []
        print('calculating train index:epoch {} of {}'.format(e_loop, epoch_max))
        for b_loop in tqdm(range(num_batch_each_epoch)):  # num_batch_each_epoch
            # loop over each batch in each epoch
            each_batch_index = []
            for i in range(num_label_each_batch):
                # loop over each label in each batch
                if len(index_unique_label[i]) < num_img_each_label:
                    # np.random.shuffle(index_unique_label0[i])
                    index_unique_label[i] = index_unique_label0[i]
                    # refill the variable if less than num_img_each_label
                for j in range(num_img_each_label):
                    each_batch_index.append(index_unique_label[i][0])
                    if b_loop + i + j == 0:
                        unique_label_temp = np.delete(index_unique_label[i], [0])  ##np.delete 先展开后删除
                        index_unique_label = list(index_unique_label)
                        index_unique_label[i] = unique_label_temp
                        index_unique_label = np.array(index_unique_label, dtype=object)
                    else:
                        index_unique_label[i] = np.delete(index_unique_label[i], [0])
                # print(index_unique_label[i])

                #     print(index_unique_label)
                #     print("index_unique_label[i]",index_unique_label[i])
                #     print("np.delete(index_unique_label[i], [0])", np.delete(index_unique_label[i], [0]))
                #
                #     index_unique_label[i] = np.delete(index_unique_label[i], [0])
                # print(index_unique_label[i])

            each_epoch_index.append(each_batch_index)
            index_unique_label = np.roll(index_unique_label, -num_label_each_batch)
            index_unique_label0 = np.roll(index_unique_label0, -num_label_each_batch)

            if (b_loop + 1) % int(np.ceil(num_label / num_label_each_batch)) == 0:
                random.shuffle(index_index)
                index_unique_label = index_unique_label[index_index]
                index_unique_label0 = index_unique_label0[index_index]

        index_train.append(each_epoch_index)

    return np.array(index_train)


def load_visnir_for_train(data_root, train_set, sz_patch=64, nb_pt_each_batch=512, nb_pat_per_pt=2, epoch_max=200,
                          flag_load_index=True):  # all outputs are numpy arrays
    train_root = os.path.join(data_root, train_set)

    file_data_train = os.path.join(train_root, train_set + '_sz' + str(sz_patch) + '.npz')

    file_index_train = os.path.join(train_root, train_set + '_epoch' + str(epoch_max) + '_index_train_ID' + str(
        nb_pt_each_batch) + '_pat' + str(nb_pat_per_pt) + '.npy')
    if os.path.exists(file_data_train):
        print('train data of {} already exists!'.format(train_set))
        data = np.load(file_data_train, allow_pickle=True)
        patch = data['patch']
        pointID = data['pointID']
        index_unique_ID = data['index_unique_ID']
        del data
    else:
        print(train_set)
        country_patch_path = os.path.join(train_root, 'country_sos.npy')
        patch = np.load(country_patch_path)
        # patch = read_UBC_patch_opencv(train_root, sz_patch)
        pointID = read_visnir_pointID(train_root, train_set)
        index_unique_ID = []  # it is a list
        pointID_unique = np.unique(pointID)  
        for id in pointID_unique:
            index_unique_ID.append(
                np.argwhere(pointID == id).squeeze())  
        np.savez(file_data_train, patch=patch, pointID=pointID, index_unique_ID=np.array(index_unique_ID, dtype=object))
    index_train = []
    if flag_load_index:
        if os.path.exists(file_index_train):
            print('index_train of {} already exists!'.format(train_set))
            index_train = np.load(file_index_train, allow_pickle=True)
        else:
            index_train = cal_index_train_visnir(index_unique_ID, nb_pt_each_batch, nb_pat_per_pt, epoch_max)
            np.save(file_index_train, index_train)

    return torch.from_numpy(patch), pointID, index_train


def load_visnir_for_train_new(data_root, train_set, sz_patch=64, nb_pt_each_batch=512, nb_pat_per_pt=2, epoch_max=200,
                          flag_load_index=True):  # all outputs are numpy arrays
    train_root = os.path.join(data_root, train_set)

    file_data_train = os.path.join(train_root, train_set + '_sz' + str(sz_patch) + '.npz')

    file_index_train = os.path.join(train_root, train_set + '_epoch' + str(epoch_max) + '_index_train_ID' + str(
        nb_pt_each_batch) + '_pat' + str(nb_pat_per_pt) + '.npy')
    if os.path.exists(file_data_train):
        print('train data of {} already exists!'.format(train_set))
        data = np.load(file_data_train, allow_pickle=True)
        patch = data['patch']
        pointID = data['pointID']
        index_unique_ID = data['index_unique_ID']
        del data
    else:
        print(train_set)
        input_npy_name = train_set + '_sos.npy'
        country_patch_path = os.path.join(train_root, input_npy_name)
        patch = np.load(country_patch_path)
        # patch = read_UBC_patch_opencv(train_root, sz_patch)
        pointID = read_visnir_pointID(train_root, train_set)
        index_unique_ID = []  # it is a list
        pointID_unique = np.unique(pointID)
        for id in pointID_unique:
            index_unique_ID.append(
                np.argwhere(pointID == id).squeeze())
        np.savez(file_data_train, patch=patch, pointID=pointID, index_unique_ID=np.array(index_unique_ID, dtype=object))
    index_train = []
    if flag_load_index:
        if os.path.exists(file_index_train):
            print('index_train of {} already exists!'.format(train_set))
            index_train = np.load(file_index_train, allow_pickle=True)
        else:
            index_train = cal_index_train_visnir(index_unique_ID, nb_pt_each_batch, nb_pat_per_pt, epoch_max)
            np.save(file_index_train, index_train)

    return torch.from_numpy(patch), pointID, index_train


def extract_visnir_test(patch_test, pointID_test):
    index_test = []
    patches_pair_num = int(len(pointID_test) / 2)
    for i in range(patches_pair_num):
        index_test.append([2 * i + 0, 2 * i + 1])

    return patch_test, pointID_test, np.array(index_test)


def load_visnir_for_test(data_root, test_set, sz_patch=64):  # all outputs are numpy arrays
    test_root = os.path.join(data_root, test_set)
    file_data_test = os.path.join(test_root, test_set + '_sz' + str(sz_patch) + '_test.npz')

    if os.path.exists(file_data_test):
        print('Test data of {} already exists!'.format(test_set))
        data = np.load(file_data_test, allow_pickle=True)
        patch_test = data['patch']
        pointID_test = data['pointID']
        index_test = data['index']  # Only tesy data have attribuate 'index'
    else:
        print(test_set)
        pathch_name = test_set + '_sos.npy'
        pathch_path = os.path.join(test_root, pathch_name)
        patch_train = np.load(pathch_path)
        pointID_train = read_visnir_pointID(test_root, test_set)
        patch_test, pointID_test, index_test = extract_visnir_test(patch_train, pointID_train)
        np.savez(file_data_test, patch=patch_test, pointID=pointID_test, index=index_test)

    return patch_test, pointID_test, index_test



def random_array(device, out_dif_R):
    length = out_dif_R.shape[0]
    diff_R_indice = torch.zeros((length,)).to(device)
    for i in range(length):
        random_num = np.random.randint(0, length)
        while random_num == i:
            random_num = np.random.randint(0, length)
        diff_R_indice[i] = random_num
    #print(diff_R_indice)
    #print(diff_R_indice.dtype)
    diff_R_indice = torch.as_tensor(diff_R_indice, dtype=torch.int32)
    #print(diff_R_indice.dtype)
    return diff_R_indice


def make_metric_dataset_random(device, out_dif_L, out_dif_R, num_pt_per_batch):
    out_dif_L_new = torch.cat([out_dif_L, out_dif_L], dim=0)
    #print(diff_R_indice.shape)

    diff_R_indice = random_array(device, out_dif_R)
    out_dif_R_hard = torch.index_select(out_dif_R, dim=0, index=diff_R_indice)
    out_dif_R_new = torch.cat([out_dif_R, out_dif_R_hard], dim=0)
    ones_label = torch.ones((num_pt_per_batch,))
    zeros_label = torch.zeros((num_pt_per_batch,))
    true_label = torch.cat([ones_label, zeros_label], dim=0).to(device)

    perm_indices = torch.randperm(2 * num_pt_per_batch)
    out_dif_L_shuffle = out_dif_L_new[perm_indices]
    out_dif_R_shuffle = out_dif_R_new[perm_indices]
    true_label_shuffle = true_label[perm_indices]
    return out_dif_L_shuffle, out_dif_R_shuffle, true_label_shuffle





def data_aug(patch, num_ID_per_batch):
    # sz = patch.size()
    patch.squeeze_()
    patch = patch.numpy()
    for i in range(0, num_ID_per_batch):
        if random.random() > 0.5:
            nb_rot = np.random.randint(1, 4)
            patch[2 * i] = np.rot90(patch[2 * i], nb_rot)
            patch[2 * i + 1] = np.rot90(patch[2 * i + 1], nb_rot)

        if random.random() > 0.5:
            patch[2 * i] = np.flipud(patch[2 * i])
            patch[2 * i + 1] = np.flipud(patch[2 * i + 1])

        if random.random() > 0.5:
            patch[2 * i] = np.fliplr(patch[2 * i])
            patch[2 * i + 1] = np.fliplr(patch[2 * i + 1])

    patch = torch.from_numpy(patch)
    patch.unsqueeze_(1)
    return patch



def data_aug4(patch, num_ID_per_batch):
    rgb_batch = np.zeros((num_ID_per_batch, 64, 64))
    nir_batch = np.zeros((num_ID_per_batch, 64, 64))

    for i in range(0, num_ID_per_batch):
        rgb_batch[i] = patch[2 * i, 0, :, :]
        nir_batch[i] = patch[2 * i + 1, 0, :, :]

    rgb_batch = np.expand_dims(rgb_batch, axis=-1)
    nir_batch = np.expand_dims(nir_batch, axis=-1)

    rgb_batch = np.array(rgb_batch, dtype='uint8')
    # print(rgb_batch.shape)
    nir_batch = np.array(nir_batch, dtype='uint8')

    seq1 = iaa.SomeOf((0, 2), [

        iaa.GaussianBlur(0, 1),
        iaa.GammaContrast((0.5, 1.5)),
        iaa.ScaleX((0.95, 1.1), mode="symmetric"),
        iaa.ScaleY((0.95, 1.1), mode="symmetric"),
        iaa.TranslateX(percent=(-0.05, 0.05), mode="symmetric"),
        iaa.TranslateY(percent=(-0.05, 0.05), mode="symmetric"),
        # iaa.PiecewiseAffine(scale=(0, 0.05), mode="symmetric"),
        iaa.AdditiveGaussianNoise(scale=0.02 * 255),
        iaa.CropAndPad(percent=(-0.1, 0.1), pad_mode="symmetric")
    ], random_order=True)  

    seq2 = iaa.SomeOf((0, 2), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.Affine(rotate=90),
        iaa.Affine(rotate=180),
        iaa.Affine(rotate=270)],
                      random_order=True) 

    nir_aug = seq1(images=nir_batch)
    rgb_aug = seq1(images=rgb_batch)

    #     rgb_aug = np.array(rgb_aug,dtype='int32')
    #     #print(rgb_batch.shape)
    #     nir_aug = np.array(nir_aug,dtype='int32')
    #     #print(nir_batch.shape)

    rgb_aug2, nir_aug2 = seq2(images=rgb_aug, segmentation_maps=nir_aug)
    rgb_aug2 = np.squeeze(rgb_aug2, axis=-1)
    nir_aug2 = np.squeeze(nir_aug2, axis=-1)
    # rgb_aug2 = np.array(rgb_aug2, dtype='float')
    # nir_aug2 = np.array(nir_aug2, dtype='float')
    rgb_aug2 = torch.from_numpy(rgb_aug2).float()
    nir_aug2 = torch.from_numpy(nir_aug2).float()
    rgb_aug2.unsqueeze_(1)
    nir_aug2.unsqueeze_(1)

    # print(np.shape(rgb_aug2))

    return rgb_aug2, nir_aug2



def gen(rgb_data, nir_data, label, batch_size):
    leng = len(label)
    rgb_batch = np.zeros((batch_size, 1, 64, 64))
    nir_batch = np.zeros((batch_size, 1, 64, 64))
    label_batch = np.zeros(batch_size)

    temp_list = []
    for i in range(batch_size):
        temp = np.random.randint(0, leng)
        while (temp in temp_list):
            temp = np.random.randint(0, leng)
        temp_list.append(temp)
        rgb_batch[i] = rgb_data[temp]
        nir_batch[i] = nir_data[temp]
        label_batch[i] = label[temp]

    rgb_batch = np.squeeze(rgb_batch, axis=1)
    nir_batch = np.squeeze(nir_batch, axis=1)
    rgb_batch = np.expand_dims(rgb_batch, axis=-1)
    nir_batch = np.expand_dims(nir_batch, axis=-1)

    rgb_batch = np.array(rgb_batch, dtype='uint8')
    #print(rgb_batch.shape)
    nir_batch = np.array(nir_batch, dtype='uint8')


    seq2 = iaa.SomeOf((0, 2), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.Affine(rotate=90),
        iaa.Affine(rotate=180),
        iaa.Affine(rotate=270)],
                      random_order=True) 


    rgb_aug2, nir_aug2 = seq2(images=rgb_batch, segmentation_maps=nir_batch)

    rgb_aug2 = np.squeeze(rgb_aug2, axis=-1)
    nir_aug2 = np.squeeze(nir_aug2, axis=-1)
    rgb_aug2 = np.expand_dims(rgb_aug2, axis=1)
    nir_aug2 = np.expand_dims(nir_aug2, axis=1)
    rgb_aug2 = torch.from_numpy(rgb_aug2).float()
    nir_aug2 = torch.from_numpy(nir_aug2).float()
    label_batch = torch.from_numpy(label_batch).float()
    #print(rgb_aug2.shape)

    return rgb_aug2, nir_aug2, label_batch



def gen4(rgb_data, nir_data, label, batch_size):
    leng = len(label)
    rgb_batch = np.zeros((batch_size, 1, 64, 64))
    nir_batch = np.zeros((batch_size, 1, 64, 64))
    label_batch = np.zeros(batch_size)

    temp_list = []
    for i in range(batch_size):
        temp = np.random.randint(0, leng)
        while (temp in temp_list):
            temp = np.random.randint(0, leng)
        temp_list.append(temp)
        rgb_batch[i] = rgb_data[temp]
        nir_batch[i] = nir_data[temp]
        label_batch[i] = label[temp]

    rgb_batch = np.squeeze(rgb_batch, axis=1)
    nir_batch = np.squeeze(nir_batch, axis=1)
    rgb_batch = np.expand_dims(rgb_batch, axis=-1)
    nir_batch = np.expand_dims(nir_batch, axis=-1)

    rgb_batch = np.array(rgb_batch, dtype='uint8')
    #print(rgb_batch.shape)
    nir_batch = np.array(nir_batch, dtype='uint8')

    seq1 = iaa.SomeOf((0, 2), [

        iaa.GaussianBlur(0, 1),
        iaa.GammaContrast((0.5, 1.5)),
        iaa.ScaleX((0.95, 1.05), mode="symmetric"),
        iaa.ScaleY((0.95, 1.05), mode="symmetric"),
        iaa.TranslateX(percent=(-0.05, 0.05), mode="symmetric"),
        iaa.TranslateY(percent=(-0.05, 0.05), mode="symmetric"),
        # iaa.PiecewiseAffine(scale=(0,0.05),mode="symmetric"),
        iaa.AdditiveGaussianNoise(scale=0.02 * 255),
        iaa.CropAndPad(percent=(-0.1, 0.1), pad_mode="symmetric")
    ], random_order=True)  

    seq2 = iaa.SomeOf((0, 2), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.Affine(rotate=90),
        iaa.Affine(rotate=180),
        iaa.Affine(rotate=270)],
                      random_order=True)  

    nir_aug = seq1(images=nir_batch)
    rgb_aug = seq1(images=rgb_batch)


    rgb_aug2, nir_aug2 = seq2(images=rgb_aug, segmentation_maps=nir_aug)

    rgb_aug2 = np.squeeze(rgb_aug2, axis=-1)
    nir_aug2 = np.squeeze(nir_aug2, axis=-1)
    rgb_aug2 = np.expand_dims(rgb_aug2, axis=1)
    nir_aug2 = np.expand_dims(nir_aug2, axis=1)
    rgb_aug2 = torch.from_numpy(rgb_aug2).float()
    nir_aug2 = torch.from_numpy(nir_aug2).float()
    label_batch = torch.from_numpy(label_batch).float()
    #print(rgb_aug2.shape)

    return rgb_aug2, nir_aug2, label_batch



def cal_fpr95(desc, pointID, pair_index):
    dist = desc[pair_index[:, 0], :] - desc[pair_index[:, 1], :]
    dist.pow_(2)
    dist = torch.sqrt(torch.sum(dist, 1)) 
    pairSim = pointID[pair_index[:, 0]] - pointID[pair_index[:, 1]]
    pairSim = torch.Tensor(pairSim)
    dist_pos = dist[pairSim == 0]
    dist_neg = dist[pairSim != 0]
    # print("dist",dist)
    # print("dist_pos的长度",len(dist_pos))
    # print("dist_neg的长度", len(dist_neg))
    dist_pos, indice = torch.sort(dist_pos)
    loc_thr = int(np.ceil(dist_pos.numel() * 0.95))
    thr = dist_pos[loc_thr]

    # print("dist_pos.numel()",dist_pos.numel())
    # print("thr",thr)
    # print("dist_neg.numel()",dist_neg.numel())
    # print("dist_neg.le(thr).sum()",dist_neg.le(thr).sum())

    fpr95 = float(dist_neg.le(thr).sum()) / dist_neg.numel()
    return fpr95



def cal_test_true_label(pointID, pair_index):
    test_true_label = pointID[pair_index[:, 0]] - pointID[pair_index[:, 1]]
    test_true_label = np.where(test_true_label == 0, 1, 0)
    test_true_label = torch.Tensor(test_true_label)
    return test_true_label


def cal_fpr95_metric(test_true_label, metric_out):
    # metric_out_1 = np.where(metric_out > 0.5, 1, 0)
    # accuracy = metrics.accuracy_score(test_true_label, metric_out_1)
    
    # precision = metrics.precision_score(test_true_label, metric_out_1)
    
    # recall = metrics.recall_score(test_true_label, metric_out_1)
    
    # f1 = metrics.f1_score(test_true_label, metric_out_1)
    
    fpr, tpr, thresholds = metrics.roc_curve(test_true_label, metric_out)
    
    fpr95 = float(interpolate.interp1d(tpr, fpr)(0.95))
    
    # area = metrics.roc_auc_score(test_true_label, metric_out)
    return fpr95





def make_metric_dataset(device, out_dif_L, out_dif_R, diff_R_indice, num_pt_per_batch):
    out_dif_L_new = torch.cat([out_dif_L, out_dif_L], dim=0)
    #print(diff_R_indice.shape)
    out_dif_R_hard = torch.index_select(out_dif_R, dim=0, index=diff_R_indice)
    out_dif_R_new = torch.cat([out_dif_R, out_dif_R_hard], dim=0)
    ones_label = torch.ones((num_pt_per_batch,))
    zeros_label = torch.zeros((num_pt_per_batch,))
    true_label = torch.cat([ones_label, zeros_label], dim=0).to(device)

    perm_indices = torch.randperm(2 * num_pt_per_batch)
    out_dif_L_shuffle = out_dif_L_new[perm_indices]
    out_dif_R_shuffle = out_dif_R_new[perm_indices]
    true_label_shuffle = true_label[perm_indices]
    return out_dif_L_shuffle, out_dif_R_shuffle, true_label_shuffle


def consistency_loss_l1(feature_map1, feature_map2):
    l1_loss = F.l1_loss(feature_map1, feature_map2)
    return l1_loss

def consistency_loss_l2(feature_map1, feature_map2):
    l2_loss = F.mse_loss(feature_map1, feature_map2)
    return l2_loss













