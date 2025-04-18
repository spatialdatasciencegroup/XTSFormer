import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, \
    f1_score, roc_auc_score
from tabulate import tabulate
import logging
import time
from matplotlib import pyplot as plt

def xts_classify_3(predict, label):

    pred_index = predict.argmax(dim=1)

    label_index = label.argmax(dim=1)

    num_correct = pred_index.eq(label_index).sum().float().item()

    accuracy = num_correct / predict.shape[0]

    return accuracy


def xts_classify_2(predict, label, binary_class=False):


    cm = confusion_matrix(label, predict)
    print(f"*****************BELOW K=1***************************")
    print(f"Confusion Matrix:\n {cm}")
    print(f"Classification Report:\n {classification_report(label, predict, digits=2)}")
    print(f"Micro Precision: {precision_score(label, predict, average='micro')}")
    print(f"Micro Recall: {recall_score(label, predict, average='micro')}")
    print(f"Micro f1-score: {f1_score(label, predict, average='micro')}")

    if binary_class:
        roc = roc_auc_score(label, predict)
        print(f"ROC_AUC_Score: {roc}")

def xts_classify_2_logging(predict, label, binary_class=False):

    cm = confusion_matrix(label, predict)
    print("*****************BELOW K=1***************************")
    logging.info("*****************BELOW K=1***************************")
    print(f"Confusion Matrix:\n {cm}")
    logging.info(f"Confusion Matrix:\n {cm}")
    print(f"Classification Report:\n {classification_report(label, predict, digits=2)}")
    logging.info(f"Classification Report:\n {classification_report(label, predict, digits=2)}")
    print(f"Micro Precision: {precision_score(label, predict, average='micro')}")
    logging.info(f"Micro Precision: {precision_score(label, predict, average='micro')}")
    print(f"Micro Recall: {recall_score(label, predict, average='micro')}")
    logging.info(f"Micro Recall: {recall_score(label, predict, average='micro')}")
    print(f"Micro f1-score: {f1_score(label, predict, average='micro')}")
    logging.info(f"Micro f1-score: {f1_score(label, predict, average='micro')}")

    if binary_class:
        roc = roc_auc_score(label, predict)
        print(f"ROC_AUC_Score: {roc}")
        logging.info(f"ROC_AUC_Score: {roc}")

def xts_classify_1(predict, label):

    pred_index = predict.argmax(dim=1)

    label_index = label#.argmax(dim=1)

    xts_classify_2_logging(pred_index.cpu().numpy(), label_index.cpu().numpy())


def xts_exp_pdf(lamda, x):

    return lamda * torch.exp(-lamda * x)

def xts_exp_loss(lamda, x):
    # loss = xts_exp_pdf(lamda, x)
    # lamda is not delta_T, but is the "shijianlv", the ratio of event in unit time.
    # 1/lamda is the delta_T

    pdf = torch.log(lamda + 1e-6) - lamda.mul(x)
    # pdf: probability dense function
    return -pdf.sum()/len(pdf)

def xts_weibull_log(x, lamda, k):
    # x,lamda,k must be >= 0, so they must be proposed by relu.
    x = x + 1e-3
    lamda = lamda + 1e-2
    k = k
    # res = (k / lamda) * (x / lamda)**(k - 1) * torch.exp(-(x / lamda)**k)
    res = torch.log(k) - torch.log(lamda) + (k-1)*(torch.log(x)-torch.log(lamda)) - (x / lamda)**k

    return res

def xts_weibull_loss(x, lamda, k):

    pdf = xts_weibull_log(x, lamda, k)

    return -pdf.sum()/len(pdf)

def xts_weibull_median(lamda, k):

    return lamda * torch.pow(0.693, 1/k)

def xts_classify_topk(y_true, y_pred, k=3):


    y_pred = [each[0:k] for each in y_pred]

    unique_label = xts_count_unique_label(y_true, y_pred)


    res = {}
    for curr_class in unique_label:
        cur_res = []
        tp_fn = y_true.count(curr_class)  # TP+FN
        # TP+FP
        tp_fp = 0
        for i in y_pred:
            if curr_class in i:
                tp_fp += 1
        # TP
        tp = 0
        for i in range(len(y_true)):
            if y_true[i] == curr_class and curr_class in y_pred[i]:
                tp += 1
        support = tp_fn
        try:
            precision = round(tp / tp_fp, 2)
            recall = round(tp / tp_fn, 2)
            f1_score = round(2 / ((1 / precision) + (1 / recall)), 2)
        except ZeroDivisionError:
            precision = 0
            recall = 0
            f1_score = 0
        cur_res.append(precision)
        cur_res.append(recall)
        cur_res.append(f1_score)
        cur_res.append(support)
        res[str(curr_class)] = cur_res
    result_table = [['Index', 'Precision@' + str(k), 'Recall@' + str(k), 'F1_Score@' + str(k), 'Support']]
    for k, v in res.items():
        v.insert(0, k)
        result_table.append(v)
    num_of_instance = len(y_true)
    weight_info = [(v[1] * v[4], v[2] * v[4], v[3] * v[4]) for k, v in sorted(res.items())]

    weight_precision = 0
    weight_recall = 0
    weight_f1_score = 0
    for each_class in weight_info:
        weight_precision += each_class[0]
        weight_recall += each_class[1]
        weight_f1_score += each_class[2]
    weight_precision /= num_of_instance
    weight_recall /= num_of_instance
    weight_f1_score /= num_of_instance
    last_line = ['Avg_total', str(round(weight_precision, 2)), str(round(weight_recall, 2)), str(
        round(weight_f1_score, 2)), num_of_instance]
    result_table.append(last_line)

    return tabulate(result_table)

def xts_classify_topk_class_string(y_true, y_pred, k, class_dic):

    y_pred = [each[0:k] for each in y_pred]

    unique_label = xts_count_unique_label(y_true, y_pred)

    res = {}
    for curr_class in unique_label:
        cur_res = []
        tp_fn = y_true.count(curr_class)  # TP+FN
        # TP+FP
        tp_fp = 0
        for i in y_pred:
            if curr_class in i:
                tp_fp += 1
        # TP
        tp = 0
        for i in range(len(y_true)):
            if y_true[i] == curr_class and curr_class in y_pred[i]:
                tp += 1
        support = tp_fn
        try:
            precision = round(tp / tp_fp, 2)
            recall = round(tp / tp_fn, 2)
            f1_score = round(2 / ((1 / precision) + (1 / recall)), 2)
        except ZeroDivisionError:
            precision = 0
            recall = 0
            f1_score = 0
        cur_res.append(precision)
        cur_res.append(recall)
        cur_res.append(f1_score)
        cur_res.append(support)
        # res[str(curr_class)] = cur_res
        res[class_dic[curr_class]] = cur_res
    result_table = [['Index', 'Precision@' + str(k), 'Recall@' + str(k), 'F1_Score@' + str(k), 'Support']]
    for k, v in res.items():
        v.insert(0, k)
        result_table.append(v)
    num_of_instance = len(y_true)
    weight_info = [(v[1] * v[4], v[2] * v[4], v[3] * v[4]) for k, v in sorted(res.items())]

    weight_precision = 0
    weight_recall = 0
    weight_f1_score = 0
    for each_class in weight_info:
        weight_precision += each_class[0]
        weight_recall += each_class[1]
        weight_f1_score += each_class[2]
    weight_precision /= num_of_instance
    weight_recall /= num_of_instance
    weight_f1_score /= num_of_instance
    last_line = ['Avg_total', str(round(weight_precision, 2)), str(round(weight_recall, 2)), str(
        round(weight_f1_score, 2)), num_of_instance]
    result_table.append(last_line)

    return tabulate(result_table)

def xts_count_unique_label(y_true, y_pred):

    unique_label = []
    for each in y_true:
        if each not in unique_label:
            unique_label.append(each)
    for i in y_pred:
        for j in i:
            if j not in unique_label:
                unique_label.append(j)
    unique_label = list(set(unique_label))
    return unique_label


def xts_plot_lines(lines_data,
                    xlabel="",
                    ylabel_left="",
                    ylabel_right=None,
                    title="",
                    figsize=(8, 6),
                    label_size=14,
                    tick_size=12,
                    title_size=16,
                    legend_size=15,
                    dpi=500,
                    save_string="xts_plot_lines.jpg",
                    show_plot=True):
    """
        Plots line charts with optional dual y-axes, allowing customization for each line's style, label, and marker size.

        Parameters:
        - lines_data: A list of dictionaries, each representing a series of data points to be plotted. Each dictionary should
                      include 'x' (x-axis data points), 'y' (y-axis data points), 'style' (optional, matplotlib style string
                      such as 'o-g' for green circles), 'label' (optional, label for the line in legend),
                      'y_axis' (optional, 'left' or 'right' to specify which y-axis to use), and
                      'marker_size' (optional, size of the markers).
e.g.,for single y axis:
 lines_data = [
    {'x': range(1, 4), 'y': [5, 9, 4], 'label': 'Data Set 1', 'style': 'o-r', 'marker_size': 5},
    {'x': range(1, 6), 'y': [3, 5, 8, 3, 5], 'label': 'Data Set 2', 'style': 'x--b', 'marker_size': 6},
    {'x': range(1, 6), 'y': [2, 3, 2, 4, 3], 'label': 'Data Set 3', 'style': 's:g', 'marker_size': 4},
]
xts_plot_lines(lines_data, xlabel="X Axis", ylabel_left="Y Axis", title="Complex Line Chart", save_string="complex_line_chart.jpg")

for dual y axis: (make sure the y_axis is specified, and the value can only be 'left' or 'right', and in the xts_plot_lines function make sure ylabel_right is not None!)
lines_data = [
    {'x': range(1, 4), 'y': [5, 9, 4], 'label': 'Data Set 1', 'style': 'o-r', 'marker_size': 5, 'y_axis': 'left'},
    {'x': range(1, 6), 'y': [3, 5, 8, 3, 5], 'label': 'Data Set 2', 'style': 'x--b', 'marker_size': 6, 'y_axis': 'right'},
    {'x': range(1, 6), 'y': [2, 3, 2, 4, 3], 'label': 'Data Set 3', 'style': 's:g', 'marker_size': 4, 'y_axis': 'left'},
]
xts_plot_lines(lines_data, xlabel="X Axis", ylabel_left="Y Axis", ylabel_right="Y test", title="Complex Line Chart", save_string="complex_line_chart.jpg")

        - xlabel: Label for the x-axis.
        - ylabel_left: Label for the primary (left) y-axis.
        - ylabel_right: Label for the secondary (right) y-axis. If provided, enables dual y-axis feature.
        - title: Title of the plot.
        - figsize: Tuple specifying the width and height of the figure in inches.
        - label_size: Font size for the axis labels.
        - tick_size: Font size for the axis tick labels.
        - title_size: Font size for the plot title.
        - legend_size: Font size for the legend.
        - save_string: File path or name to save the plot image.
        - show_plot: If True, displays the plot window.

        Returns:
        None. The function saves the plot to the specified path and optionally displays it.
        """
    # Create the figure and primary y-axis
    fig, ax_left = plt.subplots(figsize=figsize)
    # Create secondary y-axis if specified
    if ylabel_right:
        ax_right = ax_left.twinx()
        ax_right.set_ylabel(ylabel_right, size=label_size)
    else:
        ax_right = None

    # Plot lines and collect handles and labels for the legend
    handles, labels = [], []
    for line in lines_data:
        # Determine the appropriate y-axis for the current line
        ax = ax_left if line.get('y_axis', 'left') == 'left' else ax_right
        # Plot the line using the specified or default style and marker size
        ln, = ax.plot(line['x'], line['y'], line.get('style', '*-r'), label=line.get('label', ''),
                      markersize=int(line.get('marker_size', 5)))
        # line.get('style', '*-r') is equivalent to line['style'] if 'style' in line else '*-r'
        # Collect the handle and label for the legend
        handles.append(ln)
        labels.append(line.get('label', ''))

    # Set axis labels and tick label sizes
    ax_left.set_xlabel(xlabel, size=label_size)
    ax_left.set_ylabel(ylabel_left, size=label_size)
    ax_left.tick_params(axis='x', labelsize=tick_size)
    ax_left.tick_params(axis='y', labelsize=tick_size)
    if ax_right:
        ax_right.tick_params(axis='y', labelsize=tick_size)

    # Using ax_left for the legend. Matplotlib will automatically place it in the best location
    ax_left.legend(handles, labels, fontsize=legend_size)

    # Set the plot title and save the figure to file
    plt.title(title, size=title_size)
    plt.savefig(save_string, dpi=dpi, bbox_inches="tight")

    # Show the plot if requested
    if show_plot:
        plt.show()
