import enum
import os
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import torch
from torch.utils.data import Dataset
from my_package.new_model import *
from my_package.utils import *
from xts_package.plot import *
from xts_package.data_preprocess_before import *
from xts_package.data_preprocess_after import *
from xts_package.data_load import *
from my_package.utils import xts_preprocess_features


class MedicationDataInputSource(enum.Enum):
    SHORT = 'short'
    FULL = 'full'
    cohort_5_19 = '/cohort_5_19'
    local_cohort = '../dataset/cohort'
    local_aaai = '../dataset/aaai'


def read_medication_csv(filename, _type, input_source=MedicationDataInputSource.SHORT):
    """input_source: Parent directory of CSV data files"""
    parent_dir = input_source.value

    data = []

    assert _type in (int, np.uint8, np.int16, float, np.float16), "_type must be int or float"

    with open(os.path.join(parent_dir, filename)) as f:
        line = f.readline()
        while line:
            data.append(np.array([_type(v) for v in line.split(",")[2:]]))  # Skip MRN, CASE_DATE
            line = f.readline()

    return data


def read_medication_data(input_source=MedicationDataInputSource.cohort_5_19):
    """input_source: Parent directory of CSV data files"""
    parent_dir = input_source.value

    num_pharmacy_class = sum(
        1 for _ in open(
            os.path.join(parent_dir, 'encodings/pharmacy_class.txt')
        )
    )

    med_name = read_medication_csv('med_name.csv', np.int16, input_source=input_source)
    pharmacy_class = read_medication_csv('pharmacy_class.csv', np.int16, input_source=input_source)
    taken_dt = read_medication_csv('taken_dt.csv', np.float16, input_source=input_source)
    time_bin = read_medication_csv('time_bin.csv', np.int16, input_source=input_source)

    stacked = np.stack((med_name, pharmacy_class, taken_dt, time_bin), axis=0)
    return stacked, num_pharmacy_class


def preprocess_seq_sliding_window(med_data, window_size, label_length=1):
    # Last dimension should be features
    # med_data: (R, S, F)
    subsequence_feature = []
    subsequence_label = []

    R = med_data.shape[1]
    feature_len = window_size - label_length

    for r in range(R):
        subseq_len = med_data[0, r].shape[0]
        for j in range(subseq_len - window_size):
            subsequence_feature.append(
                np.array([x[j:j + feature_len] for x in med_data[:, r]]))
            subsequence_label.append(
                np.array([x[j + feature_len:j + window_size] for x in med_data[:, r]]))

    features = np.transpose(
        np.array(subsequence_feature), axes=[0, 2, 1])
    labels = np.transpose(
        np.array(subsequence_label), axes=[0, 2, 1])

    # print("features shape:", features.shape)

    return features, labels



def xts_create_tree(X, threshold):
    final_tree_adj01 = []
    final_tree_adj12 = []
    final_tree_adj23 = []
    scale_mask_list_l0 = []
    scale_mask_list_l1 = []
    scale_mask_list_l2 = []

    for i in tqdm(range(len(X)), desc="generating trees"):
        time = X[i, :, 1]
        scale_mask_l0 = np.zeros((len(time), 3))
        scale_mask_l1 = np.zeros((threshold[2], 2))

        # CREAT TREE METHOD2: create tree by numcluster and without padding
        tree = xts_hierarchical_tree(time.reshape(-1, 1), threshold, "numcluster")
        tree1 = tree[:, 2].squeeze()
        tree2 = tree[:, 1].squeeze()
        tree3 = tree[:, 0].squeeze()

        idx_scale1_l0 = xts_find_duplicate_indices(tree1)
        raw_scale2_l0 = xts_find_duplicate_indices(tree2)
        idx_scale2_l0 = np.setdiff1d(raw_scale2_l0, idx_scale1_l0)
        raw_scale3_l0 = xts_find_duplicate_indices(tree3)
        idx_scale3_l0 = np.setdiff1d(raw_scale3_l0, raw_scale2_l0)

        m1 = xts_counter2mask(Counter(tree1))
        m2 = xts_counter2mask(Counter(tree2))
        m3 = xts_counter2mask(Counter(tree3))

        scale_mask_l0[idx_scale1_l0, 0] = 1
        scale_mask_l0[idx_scale2_l0, 1] = 1
        scale_mask_l0[idx_scale3_l0, 2] = 1
        scale_mask_list_l0.append(scale_mask_l0)

        idx_scale3_l1, cols = np.where(m1[:, idx_scale3_l0] == 1)
        idx_scale3_l1 = np.unique(idx_scale3_l1)
        idx_scale1_2_l1 = np.setdiff1d(np.arange(threshold[2]), idx_scale3_l1)
        scale_mask_l1[idx_scale1_2_l1, 0] = 1
        scale_mask_l1[idx_scale3_l1, 1] = 1
        scale_mask_list_l1.append(scale_mask_l1)

        # sp_adj01 = sp.csr_matrix(m3)
        # final_tree_adj01.append(sp_adj01)
        final_tree_adj01.append(m1)

        adj12 = xts_generate_adj_from_two_matirx_2d(m1, m2)
        # sp_adj12 = sp.csr_matrix(adj12)
        # final_tree_adj12.append(sp_adj12)
        final_tree_adj12.append(adj12)

        adj23 = xts_generate_adj_from_two_matirx_2d(m2, m3)
        # sp_adj23 = sp.csr_matrix(adj23)
        # final_tree_adj23.append(sp_adj23)
        final_tree_adj23.append(adj23)

    return final_tree_adj01, final_tree_adj12, final_tree_adj23, scale_mask_list_l0, scale_mask_list_l1


class MedicationDataset(Dataset):
    def __init__(self,
                 input_source=MedicationDataInputSource.SHORT,
                 total_subseq_len=11,
                 target_subseq_len=1,
                 time_idx=2,
                 load_file_features=None,
                 load_file_labels=None,
                 threshold=None):
        self.time_idx = time_idx

        if load_file_features and load_file_labels:
            self.load(load_file_features, load_file_labels, input_source.value)
        else:
            df = np.load(r"mimic-iv-3.1\emar_sequences.npy", allow_pickle=True)
            self.num_pharmacy_class = 68
            self.features, self.labels = preprocess_seq_sliding_window(df, total_subseq_len, target_subseq_len)
            self.features = self.features[:100]
            self.labels = self.labels[:100]
            print(f"Feature Shape: {self.features.shape}")
            print(f"Labels Shape: {self.labels.shape})")


            self.tree01, self.tree12, self.tree23, self.scale_mask_l0, self.scale_mask_l1 = xts_create_tree(self.features, threshold)
            # self.index = xts_create_index(self.features, threshold)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.from_numpy(self.features[idx]), torch.from_numpy(self.labels[idx]), torch.from_numpy(
            self.tree01[idx]), torch.from_numpy(self.tree12[idx]), torch.from_numpy(self.tree23[idx]), torch.from_numpy(
            self.scale_mask_l0[idx]), torch.from_numpy(self.scale_mask_l1[idx])

    def load(self, load_file_features, load_file_labels, parent_dir):
        self.num_pharmacy_class = 68
        self.features = np.load(load_file_features)
        self.labels = np.load(load_file_labels)

    def save(self, save_file_features, save_file_labels):
        np.save(save_file_features, self.features)
        np.save(save_file_labels, self.labels)

def load_pkl_data(name, dict_name):
    with open(name, 'rb') as f:
        data = pickle.load(f, encoding='latin-1')
        num_types = data['dim_process']
        data = data[dict_name]
        return data, int(num_types)


class rebuttal_MedicationDataset(Dataset):
    def __init__(self,
                 input_source=MedicationDataInputSource.SHORT,
                 total_subseq_len=11,
                 target_subseq_len=1,
                 time_idx=2,
                 load_file_features=None,
                 load_file_labels=None,
                 threshold=None):
        self.time_idx = time_idx

        if load_file_features and load_file_labels:
            self.load(load_file_features, load_file_labels, input_source.value)
        else:
            df = np.load("../dataset/aaai/xtstars_model2_truncated.npy", allow_pickle=True)
            self.num_pharmacy_class = sum(
                1 for _ in open(
                    os.path.join(input_source.value, 'encodings/pharmacy_class.txt')
                )
            )

            train_data, num_types = load_pkl_data('../dataset/financial/train.pkl', 'train')
            self.features, self.labels = preprocess_seq_sliding_window(df, total_subseq_len, target_subseq_len)
            self.features = self.features[:, :, 1:3]
            self.labels = self.labels[:, :, 1:3]

            print(f"Feature Shape: {self.features.shape}")
            print(f"Labels Shape: {self.labels.shape})")


    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.from_numpy(self.features[idx]), torch.from_numpy(self.labels[idx])

    def load(self, load_file_features, load_file_labels, parent_dir):
        self.num_pharmacy_class = sum(
            1 for _ in open(
                os.path.join(parent_dir, 'encodings/pharmacy_class.txt')
            )
        )
        self.features = np.load(load_file_features)
        self.labels = np.load(load_file_labels)

    def save(self, save_file_features, save_file_labels):
        np.save(save_file_features, self.features)
        np.save(save_file_labels, self.labels)



def preprocess_start_0(medication_dataset):
    d = medication_dataset

    d.labels[:, :, d.time_idx] -= d.features[:, -1, d.time_idx, np.newaxis]

    d.features[:, :, d.time_idx] -= d.features[:, 0, d.time_idx, np.newaxis]


def preprocess_one_hot(medication_dataset):
    # (R, S, F)
    d = medication_dataset
    n = d.num_pharmacy_class

    pharmacy_class_feature = d.features[:, :, 1].astype(np.int16)
    pharmacy_class_data = np.eye(n, dtype=np.int16)[pharmacy_class_feature]

    d.features = np.concatenate([d.features[:, :, [0]], pharmacy_class_data, d.features[:, :, 2:]], axis=2)



def preprocess_labels_time_predict(medication_dataset):
    d = medication_dataset
    d.labels = d.labels[:, :, d.time_idx]


def preprocess_labels_medication(medication_dataset):
    d = medication_dataset
    d.labels = d.labels[:, :, 1]


def preprocess_time_predict(medication_dataset):
    preprocess_start_0(medication_dataset)
    preprocess_one_hot(medication_dataset)
    preprocess_labels_time_predict(medication_dataset)


def preprocess_medication(medication_dataset):
    preprocess_start_0(medication_dataset)
    preprocess_one_hot(medication_dataset)
    preprocess_labels_medication(medication_dataset)


def build_pharmacy_class_dictionary(input_source=MedicationDataInputSource.SHORT):
    """input_source: Parent directory of CSV data files"""
    parent_dir = input_source.value
    with open(os.path.join(parent_dir, 'encodings/pharmacy_class.txt')) as f:
        # return [line.strip() for line in f.readlines()]
        return [line.strip() for line in f.readlines()]


def build_med_name_dictionary(input_source=MedicationDataInputSource.SHORT):
    """input_source: Parent directory of CSV data files"""
    parent_dir = input_source.value
    with open(os.path.join(parent_dir, 'encodings/med_name.txt')) as f:
        return [line.strip() for line in f.readlines()]
