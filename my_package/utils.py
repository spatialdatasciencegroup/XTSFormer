import os

import pandas as pd

from xts_package.plot import *
from xts_package.data_preprocess_before import *
from xts_package.data_preprocess_after import *
# from medication_data.dataset import MedicationDataInputSource


# def xts_read_medication_data(input_source=MedicationDataInputSource.SHORT):
#     """input_source: Parent directory of CSV data files"""
#     parent_dir = input_source.value
#
#     num_pharmacy_class = sum(
#         1 for _ in open(
#             os.path.join(parent_dir, 'encodings/pharmacy_class.txt')
#         )
#     )
#
#     dfn_med_name = pd.read_parquet(os.path.join(parent_dir, 'med_name.par'))
#     dfn_pharmacy_class = pd.read_parquet(os.path.join(parent_dir, 'pharmacy_class.par'))
#     dfn_taken_dt = pd.read_parquet(os.path.join(parent_dir, 'taken_dt.par'))
#     dfn_time_bin = pd.read_parquet(os.path.join(parent_dir, 'time_bin.par'))
#
#     med_name = dfn_med_name.iloc[:, 2:].to_numpy(na_value=-1).astype("int")
#     # value: from 0 to 2534
#     pharmacy_class = dfn_pharmacy_class.iloc[:, 2:].to_numpy(na_value=-1).astype("int")
#     # value: from 0 to 277
#     taken_dt = dfn_taken_dt.iloc[:, 2:].to_numpy(na_value=-1).astype("float")
#     time_bin = dfn_time_bin.iloc[:, 2:].to_numpy(na_value=-1).astype("int")
#
#     df = np.stack((med_name, pharmacy_class, taken_dt, time_bin), axis=2)
#     return df, num_pharmacy_class


def xts_subseq_startime_with0(features, label, idx_for_time, len_subseq, len_subseq_for_test):
    # predict relative delta T
    label[:, :, idx_for_time] = label[:, :, idx_for_time] - features[:, :, idx_for_time][:, -1].view(-1, 1)
    # predict absolute T
    # label[:, :, idx_for_time] = label[:, :, idx_for_time] - features[:,:,idx_for_time][:,0].view(-1, 1)
    features[:, :, idx_for_time] = features[:, :, idx_for_time] - features[:, :, idx_for_time][:, 0].view(-1, 1).repeat(
        1, len_subseq - len_subseq_for_test)

    return features, label


def xts_preprocess_features(features, num_provider_type, len_subseq, len_subseq_for_test, device):
    feat_pharmacy_class = features[:, :, 1].view(-1, 1).to("cpu")
    feat_pharmacy_class_one_hot = xts_one_hot(feat_pharmacy_class, num_provider_type).view(features.shape[0], len_subseq - len_subseq_for_test,
                                                                   -1).to(device)
    # feat_pharmacy_class_one_hot = feat_pharmacy_class.to(device)
    # feature_pharmacy_class_one_hot: [634461, 5, 278]
    feat_time = features[:, :, 2]
    feat_others = xts_tensor_join((features[:, :, 0], features[:, :, 3]), join_style="new", join_axis=2)
    feat_others = torch.from_numpy(xts_feature_normalized(feat_others.cpu().numpy(), norm_axis=0)).float().to(device)
    features = xts_tensor_join((feat_time.unsqueeze(2), feat_pharmacy_class_one_hot, feat_others), join_style="old",
                               join_axis=2)

    return features


def xts_preprocess_labels(train_y, test_y, valid_y, idx_for_class, idx_for_time, num_provider_type, device):
    # preprocess label: generate time and one hot for class.
    label_class = xts_one_hot((train_y[:, :, idx_for_class]).long().to("cpu"), num_provider_type).to(device)
    label_time = train_y[:, :, idx_for_time].float().to(device)

    test_label_class = xts_one_hot((test_y[:, :, idx_for_class]).long().to("cpu"), num_provider_type).to(device)
    test_label_time = test_y[:, :, idx_for_time].float().to(device)

    valid_label_class = xts_one_hot((valid_y[:, :, idx_for_class]).long().to("cpu"), num_provider_type).to(device)
    valid_label_time = valid_y[:, :, idx_for_time].float().to(device)

    return label_class, label_time, test_label_class, test_label_time, valid_label_class, valid_label_time


def xts_provider_type_regroup(data, style):
    if style == 1:
        pro = data[:, :, 0]
        pro[np.where(pro == 3)] = 2
        pro[np.where(pro == 4)] = 3
        pro[np.where(pro == 5)] = 3
        pro[np.where(pro == 7)] = 3
        pro[np.where(pro == 6)] = 3
        pro[np.where(pro == 8)] = 3
        pro[np.where(pro > 8)] = 3
        data[:, :, 0] = pro
    elif style == 2:
        pro = data[:, :, 0]
        pro[np.where(pro == 3)] = 2
        pro[np.where(pro == 4)] = 3
        pro[np.where(pro == 5)] = 3
        pro[np.where(pro == 7)] = 3
        pro[np.where(pro == 6)] = 3
        pro[np.where(pro == 8)] = 3
        pro[np.where(pro > 8)] = 3
        data[:, :, 0] = pro
    elif style == 3: # All - Removed Provider Types: Nayan email type: from 1-23 to 1-8 based on my excel regroup
        pro = data[:,:,0]
        pro[np.where(pro == 4)] = 6
        pro[np.where(pro == 5)] = 6
        pro[np.where(pro == 6)] = 6
        pro[np.where(pro == 7)] = 4
        pro[np.where(pro == 8)] = 1
        pro[np.where(pro == 9)] = 4
        pro[np.where(pro == 10)] = 7
        pro[np.where(pro == 11)] = 7
        pro[np.where(pro == 12)] = 7
        pro[np.where(pro == 13)] = 8
        pro[np.where(pro == 14)] = 8
        pro[np.where(pro == 15)] = 8
        pro[np.where(pro == 16)] = 8
        pro[np.where(pro == 17)] = 6
        pro[np.where(pro == 18)] = 6
        pro[np.where(pro == 19)] = 8
        pro[np.where(pro == 20)] = 8
        pro[np.where(pro == 21)] = 3
        pro[np.where(pro == 22)] = 6
        pro[np.where(pro == 23)] = 8
        data[:, :, 0] = pro

    elif style == 4: # All - Removed Provider Types: Nayan email type: from 1-23 to 1-4 based on shou zi mu regroup
        pro = data[:,:,0]
        pro[np.where(pro == 1)] = 1 # CIRCULATOR
        pro[np.where(pro == 8)] = 1

        pro[np.where(pro == 2)] = 2 # ANESTHESI...
        pro[np.where(pro == 4)] = 2
        pro[np.where(pro == 5)] = 2
        pro[np.where(pro == 6)] = 2
        pro[np.where(pro == 16)] = 2
        pro[np.where(pro == 17)] = 2
        pro[np.where(pro == 18)] = 2
        pro[np.where(pro == 21)] = 2
        pro[np.where(pro == 22)] = 2

        pro[np.where(pro == 3)] = 3  # SURGICAL...
        pro[np.where(pro == 7)] = 3
        pro[np.where(pro == 9)] = 3
        pro[np.where(pro == 10)] = 3
        pro[np.where(pro == 11)] = 3
        pro[np.where(pro == 12)] = 3

        pro[np.where(pro == 13)] = 4  # Others
        pro[np.where(pro == 14)] = 4
        pro[np.where(pro == 15)] = 4
        pro[np.where(pro == 19)] = 4
        pro[np.where(pro == 20)] = 4
        pro[np.where(pro == 23)] = 4
        data[:, :, 0] = pro

    elif style == 5:  # All - Removed Provider Types: Nayan email type: from 1-23 to 1-3 based on shou zi mu regroup (ba style=3 zhong de 4 shan diao)
        pro = data[:, :, 0]
        pro[np.where(pro == 1)] = 1  # CIRCULATOR
        pro[np.where(pro == 8)] = 1

        pro[np.where(pro == 2)] = 2  # ANESTHESI...
        pro[np.where(pro == 4)] = 2
        pro[np.where(pro == 5)] = 2
        pro[np.where(pro == 6)] = 2
        pro[np.where(pro == 16)] = 2
        pro[np.where(pro == 17)] = 2
        pro[np.where(pro == 18)] = 2
        pro[np.where(pro == 21)] = 2
        pro[np.where(pro == 22)] = 2

        pro[np.where(pro == 3)] = 3  # SURGICAL...
        pro[np.where(pro == 7)] = 3
        pro[np.where(pro == 9)] = 3
        pro[np.where(pro == 10)] = 3
        pro[np.where(pro == 11)] = 3
        pro[np.where(pro == 12)] = 3

        pro[np.where(pro == 13)] = 2  # Others
        pro[np.where(pro == 14)] = 2
        pro[np.where(pro == 15)] = 2
        pro[np.where(pro == 19)] = 2
        pro[np.where(pro == 20)] = 2
        pro[np.where(pro == 23)] = 2
        data[:, :, 0] = pro

    elif style == 6:  # 678 to 567
        pro = data[:, :, 0]
        pro[np.where(pro == 6)] = 5
        pro[np.where(pro == 7)] = 6
        pro[np.where(pro == 8)] = 7
        data[:, :, 0] = pro

    elif style == 7:
        # medication_name: 0-14
        # pharmacy_class: 0-7
        # {0.0: 800434,
        #  1.0: 472079,
        #  4.0: 364644,
        #  2.0: 343052,
        #  6.0: 314753,
        #  5.0: 294306,
        #  7.0: 292292,
        #  3.0: 290745}
        # medication_name = data[:, :, 0]
        # medication_name[np.where(medication_name == 1994)] = -1
        # medication_name[np.where(medication_name == 1991)] = -2
        # medication_name[np.where(medication_name == 2188)] = -3
        # medication_name[np.where(medication_name == 1048)] = -4
        # medication_name[np.where(medication_name == 1092)] = -5
        # medication_name[np.where(medication_name == 1783)] = -6
        # medication_name[np.where(medication_name == 23)] = -7
        # medication_name[np.where(medication_name == 878)] = -8
        # medication_name[np.where(medication_name == 1869)] = -9
        # medication_name[np.where(medication_name == 2097)] = -10
        # medication_name[np.where(medication_name == 1900)] = -11
        # medication_name[np.where(medication_name == 691)] = -12
        # medication_name[np.where(medication_name == 1332)] = -13
        # medication_name[np.where(medication_name == 1757)] = -14
        # medication_name[np.where(medication_name > -1)] = 0
        # for i in range(14):
        #     medication_name[np.where(medication_name == -(i+1))] = i+1
        # data[:, :, 0] = medication_name

        pharmacy_class = data[:, :, 1]
        pharmacy_class[np.where(pharmacy_class == 136)] = -1
        pharmacy_class[np.where(pharmacy_class == 206)] = -2
        pharmacy_class[np.where(pharmacy_class == 224)] = -3
        pharmacy_class[np.where(pharmacy_class == 139)] = -4
        pharmacy_class[np.where(pharmacy_class == 218)] = -4
        pharmacy_class[np.where(pharmacy_class == 216)] = -4
        pharmacy_class[np.where(pharmacy_class == 15)] = -5
        pharmacy_class[np.where(pharmacy_class == 165)] = -5
        pharmacy_class[np.where(pharmacy_class == 113)] = -5
        pharmacy_class[np.where(pharmacy_class == 270)] = -6
        pharmacy_class[np.where(pharmacy_class == 245)] = -6
        pharmacy_class[np.where(pharmacy_class == 258)] = -6
        pharmacy_class[np.where(pharmacy_class == 189)] = -6
        pharmacy_class[np.where(pharmacy_class == 215)] = -7
        pharmacy_class[np.where(pharmacy_class == 156)] = -7
        pharmacy_class[np.where(pharmacy_class == 193)] = -7
        pharmacy_class[np.where(pharmacy_class == 184)] = -7
        pharmacy_class[np.where(pharmacy_class == 231)] = -7
        pharmacy_class[np.where(pharmacy_class == 154)] = -7
        pharmacy_class[np.where(pharmacy_class > -1)] = 0
        for i in range(7):
            pharmacy_class[np.where(pharmacy_class == -(i+1))] = i+1
        data[:, :, 1] = pharmacy_class

    return data


