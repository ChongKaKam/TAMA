import yaml
import matplotlib.pyplot as plt
import numpy as np
import os

log_path = '/home/zhuangjiaxin/workspace/TensorTSL/TimeLLM/log/UCR_log.yaml'
data_path = '/home/zhuangjiaxin/workspace/TensorTSL/TimeLLM/data/UCR'
log_image_output_path = '/home/zhuangjiaxin/workspace/TensorTSL/TimeLLM/log/UCR_image'
log = yaml.safe_load(open(log_path, 'r'))

tp = 0
fp = 0
tn = 0
fn = 0
# point-wise
window_length = 320
error_range = 0

def str2list(info:str):
    if info == '[]':
        return []
    else:
        return list(map(int, info.strip('[]').split(',')))

def get_abnormal_detection_set(pred:str):
    pair_num = len(pred) // 2
    single_num = len(pred) % 2
    abnormal_detection_set = set()
    for i in range(pair_num):
        start = pred[i*2]
        end = pred[i*2+1]
        for j in range(start, end+1):
            abnormal_detection_set.add(j)
    for i in range(single_num):
        single_index = pred[-1]
        start = single_index - error_range
        end = single_index + error_range
        for j in range(start, end+1):
            abnormal_detection_set.add(j)
    # print(pred)
    # print(abnormal_detection_set)
    return abnormal_detection_set

for idx in log:
    item = log[idx]
    labels = str2list(item['labels'])
    pred = str2list(item['abnormal_index'])
    abnormal_detection_set = get_abnormal_detection_set(pred)
    if pred != labels:
        print(f'Index: {idx}\nimage: {item["image"]}\nlabels: {labels}\npred: {pred}\n')
    for i in range(window_length):
        if i in labels and i in abnormal_detection_set:
            tp += 1
        elif i in labels and i not in abnormal_detection_set:
            fn += 1
        elif i not in labels and i in abnormal_detection_set:
            fp += 1
        else:
            tn += 1
    # plot image
    img_path = item['image']
    # /home/zhuangjiaxin/workspace/TensorTSL/TimeLLM/output/UCR/137/test/data.npy
    data_path = os.path.join(os.path.dirname(os.path.dirname(img_path)), 'data.npy')
    index = int(os.path.basename(img_path).split('.')[0])
    data = np.load(data_path)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    index_data = data[index]
    ax.plot(index_data, label='Time Series Data')
    pred_seq = np.ones_like(index_data) * np.min(index_data)
    for pred_idx in abnormal_detection_set:
        if pred_idx < len(pred_seq):
            pred_seq[pred_idx] = np.max(index_data)
    ax.plot(pred_seq, 'green', label='Prediction')
    # labels_seq = np.zeros_like(index_data)
    # for label_idx in labels:
    #     if label_idx < len(labels_seq):
    #         labels_seq[label_idx] = index_data[label_idx]
    # for label in labels:
    # ax.plot(labels_seq, 'yellow', label='Anomaly')
    for label_idx in labels:
        ax.plot(label_idx, index_data[label_idx], 'o', color='red', label='Anomaly' if label_idx == labels[0] else '', markersize=2)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Time Series Data')
    plt.legend()
    plt.savefig(os.path.join(log_image_output_path, f"{idx}.png"))
    

    
        

precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * precision * recall / (precision + recall)
print(f'TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}')
print(f'Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}')
    
    
    

        