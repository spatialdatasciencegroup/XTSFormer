import time as tm
from datetime import datetime

from tabulate import tabulate
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm
# from XTSFormer.constants import *
from XTSFormer.dataset import MedicationDataset, MedicationDataInputSource, preprocess_medication, \
    build_med_name_dictionary, build_pharmacy_class_dictionary
from my_package.models import *
from my_package.utils import *
from xts_package.data_preprocess_before import *
from xts_package.data_preprocess_after import *
from xts_package.data_load import *
from copy import deepcopy
import logging
import time

log_path = 'log/' + str(time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))) + '.log'
logging.basicConfig(filename=log_path, filemode='a', datefmt='%H:%M:%S', level=logging.INFO,
                    format='%(asctime)s: \n%(message)s')

NUM_EPOCHS = 200
LEARNING_RATE = 9e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 1280
# Compute validation loss for every epoch_valid_interval epochs
EPOCH_VALIDATION_INTERVAL = 1
# set patience for early stopping. If the model doesn't improve after 'patience' epochs, stop training.
best_loss = float('inf')
best_valid_loss = float('inf')
patience = 25
patience_counter = 0
# 278
total_subseq_len = 71
target_subseq_len = 1
embedding_all_dim = 256
embedding_time_dim = 128
threshold = [4, 15, 30]
# threshold = [1, 2, 3]
device = torch.device("cuda:0")
INPUT_SOURCE = MedicationDataInputSource.local_cohort
# INPUT_SOURCE = '../dataset/cohort'
last_checkpoint_time = tm.time()

pharmacy_class_dictionary = xts_txt_load(xts_dir_join(INPUT_SOURCE.value, 'encodings/pharmacy_class.txt'))
med_name_dictionary = xts_txt_load(xts_dir_join(INPUT_SOURCE.value, 'encodings/med_name.txt'))
time_bin_dictionary = ["Pre", "Intra", "Post"]

# dataset = MedicationDataset(
#     input_source=INPUT_SOURCE,
#     total_subseq_len=total_subseq_len,
#     load_file_features="features_short.npy",
#     load_file_labels="labels_short.npy")

dataset = MedicationDataset(
    input_source=INPUT_SOURCE,
    total_subseq_len=total_subseq_len,
    target_subseq_len=target_subseq_len,
    threshold=threshold)

logging.info(f'''Hyper-parameter:
NUM_EPOCHS: {NUM_EPOCHS}
LEARNING_RATE: {LEARNING_RATE}
WEIGHT_DECAY: {WEIGHT_DECAY}
BATCH_SIZE: {BATCH_SIZE}
Total_subseq_len: {total_subseq_len}
Target_subseq_len: {target_subseq_len}
Embedding_all_dim: {embedding_all_dim}
Embedding_time_dim: {embedding_time_dim}
Threshold_create_tree: {threshold}
Early_stop_patience: {patience}
Num_pharmacy_class: {dataset.num_pharmacy_class}
Feature_shape: {dataset.features.shape}
Labels_shape: {dataset.labels.shape}
''')

logging.info(f"Dataset Loading & Preprocessing: {round(tm.time() - last_checkpoint_time, 2)} sec")
last_checkpoint_time = xts_check_time_cost("Dataset Loading & Preprocessing", last_checkpoint_time)

# features, label = xts_seq_sliding_window(df, total_subseq_len, target_subseq_len)
# features: (503332, 10, 4)
# label: (503332, 1, 4)

# Regroup Class
# features = xts_provider_type_regroup(features, style=7)
# label = xts_provider_type_regroup(label, style=7)
# preprocess_medication(dataset)

# features, label = xts_subseq_startime_with0(features, label, time_idx, total_subseq_len, target_subseq_len)

# re-construct features with normalized other_feat and one-hot
# features = xts_preprocess_features(features, num_pharmacy_class, total_subseq_len, target_subseq_len, device)
# features: [time, feat_pharmacy_class_one_hot, feat_others]

# split
# train_x, train_y, valid_x, valid_y, test_x, test_y = xts_split_dataset(features, label, 0.8, 0.1)

lengths = [int(len(dataset) * v) for v in [.8, .1]]
lengths.append(len(dataset) - sum(lengths))
train_set, val_set, test_set = random_split(dataset, lengths)

# dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)
logging.info(f"Dataset Splitting: {round(tm.time() - last_checkpoint_time, 2)} sec")
last_checkpoint_time = xts_check_time_cost("Dataset Splitting", last_checkpoint_time)

# # # compute_class_weight_for_imbalance
# dic=np.load("dict_pharmacy_class_short.npy", allow_pickle=True)
# dic=dic.tolist()
# dic[125]=0
# dic = xts_sort_dict(dic, according=0)
# list_value = [value for value in dic.values()]
# numpy_value = np.array(list_value)
# weights = torch.from_numpy(numpy_value).to(device)
# weights = 10 * weights / weights.max()

decoder = task_type_to_decoder("class",
                               embedding_all_dim=embedding_all_dim,
                               feat_class_dim=dataset.num_pharmacy_class)

XTSFormer = XTSFormer(device=device,
                             feat_input_dim=dataset.features.shape[-1],
                             feat_class_dim=dataset.num_pharmacy_class,
                             decoder=decoder,
                             embedding_all_dim=embedding_all_dim,
                             embedding_time_dim=embedding_time_dim,
                             num_layers=1,
                             n_head=2,
                             drop_out=0,
                             gamma=1e-1,
                             total_subseq_len=total_subseq_len)

XTSFormer = xts_multiGPU(1, XTSFormer)

XTSFormer = XTSFormer.to(device)
# optimizer = torch.optim.Adam(XTSFormer.parameters(), lr=LEARNING_RATE)
loss_mse = nn.MSELoss()
loss_l1 = nn.L1Loss()
loss_ce = nn.CrossEntropyLoss()
# loss_wce = nn.CrossEntropyLoss(weights)
lr_optimizer = torch.optim.Adam(XTSFormer.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.StepLR(lr_optimizer, step_size=5, gamma=0.9)

loss_history, valid_loss_history = [], []
file_train_path = "../medication_data/sparse_adj_len40_batch1280/train"
file_val_path = "../medication_data/sparse_adj_len40_batch1280/val"
file_test_path = "../medication_data/sparse_adj_len40_batch1280/test"
XTSFormer.train()
for epoch in tqdm(range(NUM_EPOCHS), desc="Epoch"):
    # train_x: (1528, 5, 3)
    # train_y: (1528, 3)
    curr_epoch_total_loss = 0
    # train_x, label_class_1d = xts_shuffle_data_label(train_x, label_class_1d)
    idx_batch = 0
    # for k in range(num_batch):
    for X, y, tree01, tree12, tree23, scale_mask_l0, scale_mask_l1 in train_dataloader:
        time_batch_start = tm.time()
        # predict: [1280, 86]
        predict = XTSFormer(X.float().to(device), tree01.float().to(device), tree12.float().to(device),
                       tree23.float().to(device), scale_mask_l0.float().to(device),
                       scale_mask_l1.float().to(device))
        # loss = loss_ce(predict, y.squeeze().long().to(device))
        loss = loss_ce(predict, y.squeeze()[:, 1].long().to(device))
        lr_optimizer.zero_grad()
        loss.backward()
        lr_optimizer.step()
        curr_epoch_total_loss += loss
        idx_batch += 1

    curr_epoch_avg_loss = curr_epoch_total_loss.item() / len(train_dataloader)
    scheduler.step()
    loss_history.append(curr_epoch_avg_loss)

    loss_valid = 0
    if epoch % EPOCH_VALIDATION_INTERVAL == 0:
        with torch.no_grad():
            losses = []
            idx_batch = 0
            for X, y, tree01, tree12, tree23, scale_mask_l0, scale_mask_l1 in val_dataloader:
                # predict_valid = XTSFormer(X.float().to(device), idx_batch, file_val_path, "val")
                predict_valid = XTSFormer(X.float().to(device), tree01.float().to(device), tree12.float().to(device),
                                     tree23.float().to(device), scale_mask_l0.float().to(device),
                                     scale_mask_l1.float().to(device))
                loss_valid = loss_ce(predict_valid, y.squeeze()[:, 1].long().to(device))
                # loss_valid = loss_ce(predict_valid, y.squeeze().long().to(device))
                # loss_valid = loss_ce(predict_valid, valid_label_class_1d.squeeze().long())
                # valid_loss_history.append(loss_valid.item())
                losses.append(loss_valid.item())
                time_batch_end = tm.time()
                # print(f'Val Batch {idx_batch}th/{len(val_dataloader)} Cost Time: {(time_batch_end - time_batch_start):.2f}s')
                idx_batch += 1
            valid_loss_history.append(sum(losses) / len(losses))
            loss_valid = sum(losses) / len(losses)

            if loss_valid + 0.0 < best_valid_loss:
                best_valid_loss = loss_valid
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

logging.info("Finish Training and Begin Testing: ")

XTSFormer.eval()
preds_test = []
all_y = []
class_loss_test = 0
Total_correct = 0
with torch.no_grad():
    idx_batch = 0
    start_time = time.time()
    for X, y, tree01, tree12, tree23, scale_mask_l0, scale_mask_l1 in test_dataloader:
        # predict_test = XTSFormer(X.float().to(device), idx_batch, file_test_path, "test")
        predict_test = XTSFormer(X.float().to(device), tree01.float().to(device), tree12.float().to(device),
                            tree23.float().to(device), scale_mask_l0.float().to(device),
                            scale_mask_l1.float().to(device))
        idx_batch += 1
        subseq_mname_tensor = X[0, :, 0]
        subseq_mname = [med_name_dictionary[int(mname.item())] for mname in subseq_mname_tensor]

        # subseq_pclasses_tensor = X[0, :, 1:-2].argmax(dim=-1)
        subseq_pclasses_tensor = X[0, :, 1]
        subseq_pclasses = [pharmacy_class_dictionary[pclass.long().item()] for pclass in subseq_pclasses_tensor]

        subseq_time_dt_tensor = X[0, :, -2]
        subseq_time_dt = [time_dt.item() for time_dt in subseq_time_dt_tensor]

        subseq_time_bins_tensor = X[0, :, -1]
        subseq_time_bins = [time_bin_dictionary[int(time_bin.item())] for time_bin in subseq_time_bins_tensor]

        print("Case Study of Medication:")
        logging.info("Case Study of Medication:")
        # subseq_data = zip(subseq_mname, subseq_pclasses, subseq_time_dt, subseq_time_bins)
        case_study_table = tabulate({
            "Medication Name": subseq_mname,
            "Pharmacy Class": subseq_pclasses,
            "Time Delta (hours)": subseq_time_dt,
            "Time Bin": subseq_time_bins
        }, tablefmt="orgtbl", showindex=True, headers="keys")
        print(case_study_table)
        logging.info(case_study_table)
        preds = torch.nn.functional.softmax(predict_test[0, :])
        # gt = pharmacy_class_dictionary[int(y[0].item())]
        gt = pharmacy_class_dictionary[int(y[:, :, 1][0].item())]

        topk = preds.topk(5)
        pharmacy_classes = [pharmacy_class_dictionary[i] for i in topk.indices]
        topk_with_probs = list(zip(pharmacy_classes, [value.item() for value in topk.values]))
        predictions = [f"  {i + 1}) {pair[0]}: {round(pair[1] * 100, 2)}%" for i, pair in enumerate(topk_with_probs)]
        predictions_string = "\n".join(predictions)

        print(f"Top 5 predictions:\n{predictions_string}\nGround Truth: {gt}\n---")
        logging.info(f"Top 5 predictions:\n{predictions_string}\nGround Truth: {gt}\n---")

        preds_test.append(predict_test.cpu())
        all_y.append(y[:, :, 1].cpu())
        # predict_train = XTSFormer(train_x.float().to(device))
        class_loss_test += loss_ce(predict_test, y[:, :, 1].squeeze().long().to(device)).item()

        b_preds = torch.nn.functional.softmax(predict_test)
        b_topk = b_preds.topk(5)
        b_label = y[:, :, 1].int()
        Res = (b_label == b_topk.indices.cpu()).any(dim=1, keepdim=True)
        Correct = Res.sum()
        Total_correct = Total_correct + Correct

    end_time = time.time()

    execution_time = end_time - start_time


full_predict_test = torch.cat(preds_test)
full_y = torch.cat(all_y)

topk_pred = full_predict_test.topk(10).indices

xts_classify_1(full_predict_test, full_y)
print(f"CE Loss on test dataset (average over batches): {class_loss_test / len(test_dataloader)}")
print(f"Top 5 accuracy on test dataset (average over subject): {Total_correct / lengths[2]}")

# Show Top 5 to 10 Result
for k in range(5, 11):
    res = xts_classify_topk_class_string(full_y.int().squeeze().tolist(), topk_pred.tolist(), k=k,
                                         class_dic=pharmacy_class_dictionary)

    print(f"*****************BELOW K={k}***************************")
    print(res)
    logging.info(f"*****************BELOW K={k}***************************")
    logging.info(f"{res}")

dt_string = datetime.now().strftime("%m-%d-%H-%M")
file_name = f"plot/class_plot_{dt_string}"

lines_data = [
		{'x': range(len(loss_history)), 'y': loss_history, 'label': 'train loss', 'style': 'o-r', 'marker_size': 5, 'y_axis': 'left'},
		{'x': range(len(valid_loss_history)), 'y': valid_loss_history, 'label': 'valid loss', 'style': 'x--b', 'marker_size': 6, 'y_axis': 'left'},
		# {'x': range(len(acc_history)), 'y': acc_history, 'label': 'train acc', 'style': 's:g', 'marker_size': 4, 'y_axis': 'right'},
		# {'x': range(len(valid_acc_history)), 'y': valid_acc_history, 'label': 'valid acc', 'style': 'D-.m', 'marker_size': 3, 'y_axis': 'right'}
	]


xts_plot_lines(line_1_x=range(len(loss_history)),
               line_1_y=loss_history,
               legend1='Train Dataset',
               line_2_x=range(len(valid_loss_history)),
               line_2_y=valid_loss_history,
               legend2='Valid Dataset',
               xlabel='Epoch',
               ylabel='Loss',
               title='Predict Class',
               label_size=14,
               tick_size=10,
               title_size=16,
               legend_size=15,
               legend_loc='upper right',
               save_string=file_name)
print(f"Saved test/train loss curve to {file_name}")

is_save_weights = input("Save model weights? [y/n]")[0].lower() == "y"
if is_save_weights:
    save_string = input("Please input the string: ")
    torch.save(XTSFormer, save_string + ".pth")

print("Finished, exiting.")
