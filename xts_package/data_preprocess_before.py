import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sp
import time as tm
import enum
from collections import Counter
from datetime import datetime
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
from scipy.cluster import hierarchy


def xts_feature_normalized(features, norm_axis=0):

    mu = np.mean(features, axis=norm_axis)
    sigma = np.std(features, axis=norm_axis)
    return (features - mu) / sigma


def xts_one_hot(label, num_class):

    out = torch.zeros(label.size(0), num_class)
    idx = label.long().view(-1, 1)  # idx: (n, 1)维的矩阵
    out.scatter_(dim=1, index=idx, value=1)  # 根据idx对dim上的数据用value进行替换。dim=1：对某一列，将第idx个数据用value进行替换
    return out


def xts_shuffle_data_label(x, y):

    shuffler = np.random.permutation(len(y))
    # 可通过np.random.RandomState(72).permutation(len(y)) 调优随机数种子 len(y)也可换成y.shape[0],只要表示样本数就行
    x_shuffled = x[shuffler]
    y_shuffled = y[shuffler]
    return x_shuffled, y_shuffled


def xts_seq_sliding_window(sequence, window_size, lable_length=1):

    subsequence_feature = []
    subsequence_label = []
    for i in range(sequence.shape[0]):

        len_i = len(sequence[i, :, 0][~np.isnan(sequence[i, :, 0])])

        num_subsequence_i = len_i - window_size + 1
        for j in range(num_subsequence_i):
            idx_start = j
            idx_end = idx_start + window_size
            subsequence_i = sequence[i, idx_start:idx_end, :]

            feature_length = subsequence_i.shape[0] - lable_length
            subsequence_feature.append(subsequence_i[:feature_length, :])
            subsequence_label.append(subsequence_i[feature_length:, :])

    return np.array(subsequence_feature), np.array(subsequence_label)


def xts_split_dataset(features, labels, train_ratio=0.8, test_ratio=0.2):

    data_length = features.shape[0]
    valid_ratio = 1 - train_ratio - test_ratio
    train_data_length, valid_data_length = int(data_length * train_ratio), int(data_length * valid_ratio)
    train_x, train_y = features[:train_data_length], labels[:train_data_length]
    valid_x, valid_y = features[train_data_length: train_data_length + valid_data_length], labels[
                                                                                           train_data_length: train_data_length + valid_data_length]
    test_x, test_y = features[train_data_length + valid_data_length:], labels[train_data_length + valid_data_length:]

    return train_x, train_y, valid_x, valid_y, test_x, test_y


def xts_sort_dict(my_dict, according=1, decline=False):

    sorted_result = sorted(my_dict.items(), key=lambda x: x[according], reverse=decline)
    converted_dict = dict(sorted_result)
    return converted_dict


def xts_time_watch():
    time_epoch_start = tm.time()
    time_epoch_end = tm.time()
    print(f'Cost time: {time_epoch_end - time_epoch_start}s')


def xts_check_time_cost(msg, last_checkpoint_time):

    print(msg, ":", round(tm.time() - last_checkpoint_time, 2), "sec")  # 保留2位小数
    last_checkpoint_time = tm.time()

    return last_checkpoint_time


def xts_numpy2sparse(numpy_data, filepath):

    numpy_sparse = sp.csr_matrix(numpy_data)
    sp.save_npz(filepath, numpy_sparse)
    return numpy_sparse


def xts_sparse2numpy(filepath):

    numpy_sparse = sp.load_npz(filepath)
    numpy_data = numpy_sparse.toarray()
    return numpy_data


def xts_tensor2sparse(tensor_data, filepath):

    tensor_sparse = tensor_data.to_sparse()
    torch.save(tensor_sparse, filepath)
    return tensor_sparse


def xts_sparse2tensor(filepath):

    tensor_sparse = torch.load(filepath)
    tensor_data = tensor_sparse.to_dense()
    return tensor_data


def xts_numpy_join(arrays, join_style, join_axis=0):

    if join_style == "new":

        res = np.stack(arrays, axis=join_axis)
    elif join_style == "old":

        res = np.concatenate(arrays, axis=join_axis)
    return res


def xts_tensor_join(tensors, join_style, join_axis=0):

    if join_style == "new":

        res = torch.stack(tensors, axis=join_axis)
    elif join_style == "old":

        res = torch.cat(tensors, axis=join_axis)
    return res


def xts_compute_class_distribution(tensor_data):

    numpy_data = tensor_data.cpu().numpy()
    cou = Counter(numpy_data.flatten())
    dic = dict(cou)
    dic = xts_sort_dict(dic, decline=True)
    np.save("xxx.npy", dic)
    read = np.load("xxx.npy", allow_pickle=True)
    # convert numpy to dict
    read = read.tolist()


def xts_multiGPU(num_gpu, model):

    if num_gpu > 1:

        print(f"Found and using {num_gpu} GPUs!")
        model = nn.DataParallel(model)

    return model


def xts_get_time_now_str(format="%m_%d_%H:%M"):

    time_string = datetime.now().strftime(format)

    return time_string

def xts_hierarchical_tree(feat, value, criterion_type='numcluster'):

    dis = pdist(feat, metric='euclidean')

    tree_process = hierarchy.linkage(dis, method='ward')

    if criterion_type == "numcluster":
        res_cluster = hierarchy.cut_tree(tree_process, value).squeeze()
    else:
        res_cluster = hierarchy.fcluster(tree_process, value, criterion=criterion_type)


    return res_cluster

def xts_counter2mask(counter):

    d = sum(counter.values())
    mask = np.zeros((len(counter), d))

    values = list(counter.values())
    indices = np.cumsum(values)
    start = 0
    for i, end in enumerate(indices):
        mask[i, start:end] = 1
        start = end

    return mask

def xts_decompose_mask(mask):

    indices = np.where(mask[:, 0] == 1)

    idx = indices[0]
    mask_ones = mask[idx[0]:idx[1]]
    mask_level1 = mask[idx[1]:idx[2]]
    mask_level2 = mask[idx[2]:idx[3]]
    mask_level3 = mask[idx[3]:]

    return mask_ones, mask_level1, mask_level2, mask_level3

def xts_generate_adj_from_two_matirx_2d(a, b):

    c = np.zeros((b.shape[0], a.shape[0]), dtype=int)
    idx = np.where(b == 1)

    c[idx[0], np.where(a[:, idx[1]] == 1)[0]] = 1
    return c

def xts_generate_adj_from_two_matirx(a, b):
    """
    The function receives two 3D tensors a and b with shapes [batch_size, n, d] and [batch_size, m, d] respectively,
    and generates a new tensor c of shape [batch_size, m, n]. The tensor c reflects the correspondence between rows
    of tensors a and b in the following manner: for each element in tensor b that is 1, the function finds the index of
    the corresponding element in tensor a that is 1, and sets the corresponding element in tensor c to 1.

    :param a: Tensor[batch_size, n, d]
    :param b: Tensor[batch_size, m, d]
    :return: c: Tensor[batch_size, m, n]
    """
    batch_size, n, d = a.shape
    _, m, _ = b.shape
    c = torch.zeros((batch_size, m, n), device=a.device)

    for batch in range(batch_size):
        idx = (b[batch] == 1).nonzero(as_tuple=True)
        c[batch, idx[0], (a[batch, :, idx[1]] == 1).nonzero(as_tuple=True)[0]] = 1

    return c

def xts_find_duplicate_indices(arr):

    unique_elements, counts = np.unique(arr, return_counts=True)
    duplicate_elements = unique_elements[counts > 1]
    duplicate_indices = np.where(np.isin(arr, duplicate_elements))[0]
    return duplicate_indices

def xts_gather_features_by_index(index_matrix, feature_matrix):

    if len(index_matrix.shape) == 2 and len(feature_matrix.shape) == 2:
        # Handle the unbatched case
        idx_expanded = index_matrix.unsqueeze(2).expand(-1, -1, feature_matrix.shape[1])
        feature_expanded = feature_matrix.unsqueeze(1).expand(-1, index_matrix.shape[1], -1)
        gathered_features = torch.gather(feature_expanded, 0, idx_expanded)

    elif len(index_matrix.shape) == 3 and len(feature_matrix.shape) == 3:
        # Handle the batched case
        idx_expanded = index_matrix.unsqueeze(3).expand(-1, -1, -1, feature_matrix.shape[2])
        feature_expanded = feature_matrix.unsqueeze(2).expand(-1, -1, index_matrix.shape[2], -1)
        gathered_features = torch.gather(feature_expanded, 1, idx_expanded)

    else:
        raise ValueError("Mismatch in number of dimensions between index_matrix and feature_matrix.")

    return gathered_features

