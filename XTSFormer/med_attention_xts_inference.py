import logging
import time

from torch.utils.data import DataLoader, random_split
import random
# from XTSFormer.constants import *
from XTSFormer.dataset import MedicationDataset, MedicationDataInputSource,MedicationDataset_inference
from my_package.models import *
from xts_package.data_load import *
from xts_package.data_preprocess_after import *
from xts_package.data_preprocess_before import *

log_path = 'log/' + str(time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))) + '.log'
logging.basicConfig(filename=log_path, filemode='a', datefmt='%H:%M:%S', level=logging.INFO,
                    format='%(asctime)s: \n%(message)s')

NUM_EPOCHS = 200
LEARNING_RATE = 9e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 1280
EPOCH_VALIDATION_INTERVAL = 1
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

last_checkpoint_time = tm.time()

pharmacy_class_dictionary = xts_txt_load(xts_dir_join(INPUT_SOURCE.value, 'encodings/pharmacy_class.txt'))
med_name_dictionary = xts_txt_load(xts_dir_join(INPUT_SOURCE.value, 'encodings/med_name.txt'))
time_bin_dictionary = ["Pre", "Intra", "Post"]

dataset = MedicationDataset_inference(
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

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed(42)

logging.info(f"Dataset Loading & Preprocessing: {round(tm.time() - last_checkpoint_time, 2)} sec")
last_checkpoint_time = xts_check_time_cost("Dataset Loading & Preprocessing", last_checkpoint_time)

lengths = [int(len(dataset) * v) for v in [.8, .1]]
lengths.append(len(dataset) - sum(lengths))
train_set, val_set, test_set = random_split(dataset, lengths)


test_dataloader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)
logging.info(f"Dataset Splitting: {round(tm.time() - last_checkpoint_time, 2)} sec")
last_checkpoint_time = xts_check_time_cost("Dataset Splitting", last_checkpoint_time)

decoder = task_type_to_decoder("class",
                               embedding_all_dim=embedding_all_dim,
                               feat_class_dim=dataset.num_pharmacy_class)
loss_ce = nn.CrossEntropyLoss()

XTSFormer = torch.load("XTSFormer.pth")

XTSFormer.eval()

logging.info("Finish Training and Begin Testing: ")

preds_test = []
all_y = []
class_loss_test = 0
Total_correct = 0
with XTSFormer.no_grad():
    idx_batch = 0
    start_time = time.time()
    for X, y, tree01, tree12, tree23, scale_mask_l0, scale_mask_l1 in test_dataloader:
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
        class_loss_test += loss_ce(predict_test, y[:, :, 1].squeeze().long().to(device)).item()

        b_preds = torch.nn.functional.softmax(predict_test)
        b_topk = b_preds.topk(5)
        b_label = y[:, :, 1].int()
        Res = (b_label == b_topk.indices.cpu()).any(dim=1, keepdim=True)
        Correct = Res.sum()
        Total_correct = Total_correct + Correct

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"execution_time cost: {execution_time} s")

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

print("Finished, exiting.")
