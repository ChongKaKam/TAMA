import os
from Datasets.Dataset import EvalDataLoader, Evaluator
import make_dataset
import tabulate
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# configuration
# dataset_name = 'UCR'  
# vote_thres = 2
# point_adjust_enable = True
# plot_enable = True
# channel_shared = False
# processed_data_root = f'./output/'
processed_data_root = f'./output/'
# log_root = f'./log/'
# log_root = f'./log/log_nas/anormaly_detection/UCR'
image_config= {
    'width': 2000,
    'height': 480,
    'x_ticks': 50,
    'dpi': 100,
}
dataset_name = 'NASA-MSLTo sign in, use a web browser to open the page https://microsoft.com/devicelogin and enter the code S6GZU5GPQ to authenticate.'
key_name = 'DCheck' + '_adjust'
dataset_info = make_dataset.dataset_config[dataset_name]
window_size = dataset_info['window']
stride = dataset_info['stride']
config = {
    "UCR": {
        'log_root': './log/log_nas/anormaly_detection/UCR',
        # 'log_root': '/home/zhuangjiaxin/workspace/TensorTSL/TimeLLM/log/text_modality',
        'data_id_list': ['135','136','137','138'],
    },
    "NASA-MSL": {
        'log_root': './log/log_nas/anormaly_detection/NASA-MSL-All-1',
        'data_id_list': ['3', '9', '10', '11', '15', '23', '24'],
    },
    "NormA": {
        'log_root': './log/log_nas/anormaly_detection/NormA-1_4_7_13',
        'data_id_list': ['1', '4', '7', '13'],
    },
    "NASA-SMAP": {
        'log_root': './log/log_nas/anormaly_detection/NASA-SMAP-2_24_27_37_45',
        'data_id_list': ['2', '24', '27', '37', '45'],
    },
}
log_path =  config[dataset_name]['log_root'] #.replace('log_nas', 'Final-Results').replace('anormaly_detection/', '')
evaluator = Evaluator(dataset_name, stride, processed_data_root, log_root=log_path)
# evaluator.calculate_TP_FP_TN_FN(9, 0, data_id_list=config[dataset_name]['data_id_list'], show_results=True)
# res = evaluator.calculate_TP_FP_TN_FN(confidence_thres=9, data_id_list=config[dataset_name]['data_id_list'],show_results=True)
# pre_list = []
# rec_list = []
# f1_list = []
# auc_pr_list = []
# auc_roc_list = []
# for data_id in config[dataset_name]['data_id_list']:
#     print(data_id)
#     metrics = evaluator.calculate_f1_aucpr_aucroc(9, 0, data_id_list=[data_id])
#     tabel = [
#         ['Name', 'P', 'R', 'F1', 'AUCPR', 'AUCROC']
#     ]
#     for name in metrics:
#         pre = metrics[name]['Pre']
#         rec = metrics[name]['Rec']
#         f1 = metrics[name]['F1']
#         aucpr = metrics[name]['AUC_PR']
#         aucroc = metrics[name]['AUC_ROC']
#         tabel.append([name, f"{pre:.3f}", f"{rec:.3f}", f"{f1:.3f}", f"{aucpr:.3f}", f"{aucroc:.3f}"])
#     pre_list.append(metrics[key_name]['Pre'])
#     rec_list.append(metrics[key_name]['Rec'])
#     f1_list.append(metrics[key_name]['F1'])
#     auc_pr_list.append(metrics[key_name]['AUC_PR'])
#     auc_roc_list.append(metrics[key_name]['AUC_ROC'])

#     print(tabulate.tabulate(tabel, headers='firstrow', tablefmt='fancy_grid'))

# print(f"Method: {key_name}")
# print(f'Precision: {max(pre_list):.3f} / {sum(pre_list)/len(pre_list):.3f}')
# print(f'Recall: {max(rec_list):.3f} /  {sum(rec_list)/len(rec_list):.3f}')
# print(f'F1: {max(f1_list):.3f} / {sum(f1_list)/len(f1_list):.3f}')
# print(f'AUC_PR: {max(auc_pr_list):.3f} / {sum(auc_pr_list)/len(auc_pr_list):.3f}')
# print(f'AUC_ROC: {max(auc_roc_list):.3f} / {sum(auc_roc_list)/len(auc_roc_list):.3f}')

evaluator.calculate_roc_pr_auc(config[dataset_name]['data_id_list'])
# evaluator.calculate_adjust_PR_curve_auc(config[dataset_name]['data_id_list'])