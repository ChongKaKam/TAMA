import yaml
import matplotlib.pyplot as plt
import numpy as np
import os
import re

log_path = '/home/zhuangjiaxin/workspace/TensorTSL/TimeLLM/log/UCR_log.yaml'
data_path = '/home/zhuangjiaxin/workspace/TensorTSL/TimeLLM/output/UCR'
log_image_output_path = '/home/zhuangjiaxin/workspace/TensorTSL/TimeLLM/log/UCR_image'
log = yaml.safe_load(open(log_path, 'r'))

data_length = {
    '135': 6301,
    '136': 5900,
    '137': 5200,
    '138': 4500,
}

tp = 0
fp = 0
tn = 0
fn = 0
# point-wise
window_length = 600
stride = 188
error_range = 0

def str2list(info:str):
    if info == '[]':
        return []
    else:
        return list(map(int, info.strip('[]').split(',')))

def str2range(info:str):
    pattern1 = r'\(\d+,\s\d+\)'
    pattern2 = r'\(\d+\)'
    # pattern = r'\((\d+)(,\s*\d+)?\)'
    if info == '[]':
        return [[]]
    else:
        matches = re.findall(pattern1, info)
        range_list = []
        for match in matches:
            match = match.strip('()')
            start, end = map(int, match.split(','))
            range_list.append((start, end))
        matches = re.findall(pattern2, info)
        for match in matches:
            match = match.strip('()')
            range_list.append((int(match),))
        return range_list

def get_abnormal_detection_set(pred:list, offset=0):
    abnormal_detection_set = set()
    for start_end in pred:
        if len(start_end) == 0:
            continue
        elif len(start_end) == 1:
            # single outlier
            abnormal_detection_set.add(start_end[0]+offset)
        else:
            # outlier range
            start, end = start_end
            for i in range(start, end+1):
                abnormal_detection_set.add(i+offset)
    return abnormal_detection_set
def adjust_anomaly_detection_results(results, labels):
    """
    Adjust anomaly detection results based on the ground-truth labels.
    
    Args:
        results (np.ndarray): The anomaly detection results (0 or 1).
        labels (np.ndarray): The ground-truth labels (0 or 1).
    
    Returns:
        np.ndarray: The adjusted anomaly detection results.
    """
    adjusted_results = results.copy()
    in_anomaly = False
    start_idx = 0
    
    for i in range(len(labels)):
        if labels[i] == 1 and not in_anomaly:
            in_anomaly = True
            start_idx = i
        elif labels[i] == 0 and in_anomaly:
            in_anomaly = False
            if np.any(results[start_idx:i] == 1):
                adjusted_results[start_idx:i] = 1
    
    # Handle the case where the last segment is an anomaly
    if in_anomaly and np.any(results[start_idx:] == 1):
        adjusted_results[start_idx:] = 1
    return adjusted_results

# initialize pred_array
pred_array = {name: np.zeros(data_length[name]) for name in data_length}
label_array = {name: np.zeros(data_length[name]) for name in data_length}
def get_id_from_image_path(img_path:str):
    pattern = r'/UCR/(\d+)/'
    match = re.search(pattern, img_path)
    return match.group(1)
# version -- 2
for idx in log:
    image_path = log[idx]['image']
    labels = str2list(log[idx]['labels'])
    pred = str2range(log[idx]['abnormal_index'])
    abnormal_type = log[idx]['abnormal_type']
    abnormal_description = log[idx]['abnormal_description']

    # data_path/{data_id}/test/image/{data_index}.png
    data_id = get_id_from_image_path(image_path)
    data_index = int(os.path.basename(image_path).split('.')[0])
    
    offset = int(data_index * stride)
    abnormal_detection_set = get_abnormal_detection_set(pred, offset)
    labels_offset = set([label + offset for label in labels])
    
    # mark label
    for labels_offset_idx in labels_offset:
        label_array[data_id][labels_offset_idx] = 1
    # mark pred
    for abnormal_detection_set_idx in abnormal_detection_set:
        pred_array[data_id][abnormal_detection_set_idx] += 1
# vote: if pred_array[data_id][data_index] >= 2, then it is an abnormal point
for data_id in pred_array:
    for data_index in range(data_length[data_id]):
        if pred_array[data_id][data_index] >= 3:
            pred_array[data_id][data_index] = 1
        else:
            pred_array[data_id][data_index] = 0
# adjust anomaly detection results
for data_id in pred_array:
    pred_array[data_id] = adjust_anomaly_detection_results(pred_array[data_id], label_array[data_id])
# calculate tp, fp, tn, fn
for data_id in pred_array:
    for data_index in range(data_length[data_id]):
        if pred_array[data_id][data_index] == 1 and label_array[data_id][data_index] == 1:
            tp += 1
        elif pred_array[data_id][data_index] == 1 and label_array[data_id][data_index] == 0:
            fp += 1
        elif pred_array[data_id][data_index] == 0 and label_array[data_id][data_index] == 0:
            tn += 1
        else:
            fn += 1
# visualization
for data_id in data_length:
    data = np.load(os.path.join(data_path, data_id, 'test', 'data.npy'))
    label = np.load(os.path.join(data_path, data_id, 'test', 'labels.npy'))
    for idx in range(data.shape[0]):
        fig, ax = plt.subplots(figsize=(8, 3.2), dpi=100)
        index_data = data[idx]
        ax.plot(index_data, label='RawData')

        offset = int(idx * stride)
        pred_data = pred_array[data_id][offset:int(offset+window_length)]
        pred_data = pred_data * np.max(index_data)
        pred_data[pred_data == 0] = np.min(index_data)
        ax.plot(pred_data, 'green', label='Prediction')
        label_data = label_array[data_id][offset:int(offset+window_length)]
        label_index = np.where(label_data == 1)[0]
        for label_idx in label_index:
            ax.plot(label_idx, index_data[label_idx], 'o', color='red', label='Anomaly' if label_idx == label_index[0] else '', markersize=2)
        # for label_idx in range(data_length[data_id]):
        #     if label_array[data_id][label_idx] == 1:
        #         ax.plot(label_idx, index_data[label_idx], 'o', color='red', label='Anomaly' if label_idx == 0 else '', markersize=2)
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('Time Series Data')
        plt.legend()
        plt.savefig(os.path.join(log_image_output_path, f"{data_id}_{idx}.png"))
        plt.close()
# version -- 1
# for idx in log:
#     item = log[idx]
#     labels = str2list(item['labels'])
#     pred = str2range(item['abnormal_index'])
#     # print(f"origin: {item['abnormal_index']}")
#     # print(f"2range: {pred}")
#     # continue
#     abnormal_detection_set = get_abnormal_detection_set(pred)
#     if pred != labels:
#         print(f'Index: {idx}\nimage: {item["image"]}\nlabels: {labels}\npred: {pred}\n')
#     for i in range(window_length):
#         if i in labels and i in abnormal_detection_set:
#             tp += 1
#         elif i in labels and i not in abnormal_detection_set:
#             fn += 1
#         elif i not in labels and i in abnormal_detection_set:
#             fp += 1
#         else:
#             tn += 1
#     # plot image
#     img_path = item['image']
#     data_path = os.path.join(os.path.dirname(os.path.dirname(img_path)), 'data.npy')
#     index = int(os.path.basename(img_path).split('.')[0])
#     data = np.load(data_path)
#     fig, ax = plt.subplots(figsize=(8, 3.2), dpi=100)
#     index_data = data[index]
#     ax.plot(index_data, label='Time Series Data')
#     pred_seq = np.ones_like(index_data) * np.min(index_data)
#     for pred_idx in abnormal_detection_set:
#         if pred_idx < len(pred_seq):
#             pred_seq[pred_idx] = np.max(index_data)
#     ax.plot(pred_seq, 'green', label='Prediction')
#     for label_idx in labels:
#         ax.plot(label_idx, index_data[label_idx], 'o', color='red', label='Anomaly' if label_idx == labels[0] else '', markersize=2)
#     plt.xlabel('Time')
#     plt.ylabel('Value')
#     plt.title('Time Series Data')
#     plt.legend()
#     plt.savefig(os.path.join(log_image_output_path, f"{idx}.png"))
accuracy = (tp+tn) / (tp+tn+fp+fn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * precision * recall / (precision + recall)
print(f'TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}')
print(f'Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}')
 