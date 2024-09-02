import os
from Datasets.Dataset import Evaluator
import make_dataset
import tabulate
import matplotlib.pyplot as plt

dataset_info_map = make_dataset.dataset_config

# nr_3_map = {
#     'Pred': {
#         'NASA-MSL': {'Pre': 0.739,'Rec': 0.193,'F1': 0.306,'AUC_PR': 0.299,'AUC_ROC': 0.704,},
#         'NASA-SMAP': {'Pre': 0.602,'Rec': 0.388,'F1': 0.472,'AUC_PR': 0.475,'AUC_ROC': 0.781,},
#         'UCR': {'Pre': 0.827,'Rec': 0.774,'F1': 0.8,'AUC_PR': 0.769,'AUC_ROC': 0.961,},
#     },
#     'Pred_adjust': {
#         'NASA-MSL': {'Pre': 0.936,'Rec': 1,'F1': 0.967,'AUC_PR': 0.299,'AUC_ROC': 0.704,},
#         'NASA-SMAP': {'Pre': 0.602,'Rec': 0.388,'F1': 0.472,'AUC_PR': 0.475,'AUC_ROC': 0.781,},
#         'UCR': {'Pre': 0.827,'Rec': 0.774,'F1': 0.8,'AUC_PR': 0.769,'AUC_ROC': 0.961,},
#     },
#     'DCheck': {
#         'NASA-MSL': {'Pre': 0.739,'Rec': 0.193,'F1': 0.306,'AUC_PR': 0.299,'AUC_ROC': 0.704,},
#         'NASA-SMAP': {'Pre': 0.602,'Rec': 0.388,'F1': 0.472,'AUC_PR': 0.475,'AUC_ROC': 0.781,},
#         'UCR': {'Pre': 0.827,'Rec': 0.774,'F1': 0.8,'AUC_PR': 0.769,'AUC_ROC': 0.961,},
#     },
#     'DCheck_adjust': {
#         'NASA-MSL': {'Pre': 0.739,'Rec': 0.193,'F1': 0.306,'AUC_PR': 0.299,'AUC_ROC': 0.704,},
#         'NASA-SMAP': {'Pre': 0.602,'Rec': 0.388,'F1': 0.472,'AUC_PR': 0.475,'AUC_ROC': 0.781,},
#         'UCR': {'Pre': 0.827,'Rec': 0.774,'F1': 0.8,'AUC_PR': 0.769,'AUC_ROC': 0.961,},
#     },
# }

nr_3_map = {
    'NASA-MSL': {
        'Pred': {'Pre': 0.739, 'Rec': 0.193, 'F1': 0.306, 'AUC_PR': 0.299, 'AUC_ROC': 0.704},
        'Pred_adjust': {'Pre': 0.936, 'Rec': 1, 'F1': 0.967, 'AUC_PR': 0.714, 'AUC_ROC': 0.916},
        'DCheck': {'Pre': 0.619, 'Rec': 0.15, 'F1': 0.241, 'AUC_PR': 0.289, 'AUC_ROC': 0.690},
        'DCheck_adjust': {'Pre': 0.916, 'Rec': 1, 'F1': 0.956, 'AUC_PR': 0.733, 'AUC_ROC': 0.921},
    },
    'NASA-SMAP': {
        'Pred': {'Pre': 0.602, 'Rec': 0.388, 'F1': 0.472, 'AUC_PR': 0.475, 'AUC_ROC': 0.781},
        'Pred_adjust': {'Pre': 0.793, 'Rec': 983, 'F1': 0.878, 'AUC_PR': 0.892, 'AUC_ROC': 0.970},
        'DCheck': {'Pre': 0.8, 'Rec': 0.395, 'F1': 0.529, 'AUC_PR': 0.586, 'AUC_ROC': 0.828},
        'DCheck_adjust': {'Pre': 0.909, 'Rec': 0.983, 'F1': 0.945, 'AUC_PR': 0.955, 'AUC_ROC': 0.984},
    },
    'UCR': {
        'Pred': {'Pre': 0.827, 'Rec': 0.774, 'F1': 0.8, 'AUC_PR': 0.769, 'AUC_ROC': 0.961},
        'Pred_adjust': {'Pre': 0.861, 'Rec': 1, 'F1': 0.925, 'AUC_PR': 0.930, 'AUC_ROC': 0.998},
        'DCheck': {'Pre': 0.827, 'Rec': 0.774, 'F1': 0.8, 'AUC_PR': 0.773, 'AUC_ROC': 0.967},
        'DCheck_adjust': {'Pre': 0.861, 'Rec': 0.1, 'F1': 0.925, 'AUC_PR': 0.930, 'AUC_ROC': 0.998},
    },
}

def eval_normal_reference(plot_enable=False):
    eval_log_root = f"/home/zhuangjiaxin/workspace/TensorTSL/TimeLLM/log/Ablation/few-shot"
    processed_data_root = "/home/zhuangjiaxin/workspace/TensorTSL/TimeLLM/output"
    dataset_name_list = ['NASA-SMAP', 'UCR']
    default_confidence = 9
    default_PAT = 0

    for dataset_name in dataset_name_list:
        dataset_info = dataset_info_map[dataset_name]
        window_size = dataset_info['window']
        stride = dataset_info['stride']
        NR_list = [0, 1, 2, 3, 4]
        plot_data_map = {}
        for key in ['Pred', 'Pred_adjust', 'DCheck', 'DCheck_adjust']:
            tabel = [
                ['Name', 'P', 'R', 'F1', 'AUCPR', 'AUCROC']
            ]
            plot_data_map = {
                'F1': [],
                'AUCPR': [],
                'AUCROC': [],
            }
            for nr in NR_list:
               
                log_path = os.path.join(eval_log_root, f"Ablation-{dataset_name}-NR-{nr}")
                if nr == 3:
                    metrics = nr_3_map[dataset_name]
                else:
                    evaluator = Evaluator(dataset_name, stride, processed_data_root, log_root=log_path)
                    metrics = metrics = evaluator.calculate_f1_aucpr_aucroc(default_confidence, default_PAT)
                # print(metrics.keys())
                name = f"{key}-{nr}"
                pre = metrics[key]['Pre']
                rec = metrics[key]['Rec']
                f1 = metrics[key]['F1']
                aucpr = metrics[key]['AUC_PR']
                aucroc = metrics[key]['AUC_ROC']
                plot_data_map['F1'].append(f1)
                plot_data_map['AUCPR'].append(aucpr)
                plot_data_map['AUCROC'].append(aucroc)
                tabel.append([name, f"{pre:.3f}", f"{rec:.3f}", f"{f1:.3f}", f"{aucpr:.3f}", f"{aucroc:.3f}"])
            print(f"\nDataset: {dataset_name}, Method: {key}")
            print(tabulate.tabulate(tabel, headers='firstrow', tablefmt='fancy_grid'))
            if plot_enable:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(NR_list, plot_data_map['F1'], label='F1', marker='x')
                ax.plot(NR_list, plot_data_map['AUCPR'], label='AUC-PR', marker='o')
                ax.plot(NR_list, plot_data_map['AUCROC'], label='AUC-ROC', marker='s')
                ax.set_xticks(NR_list)
                ax.set_xlim([0, 4])
                ax.set_ylim([0, 1])
                ax.set_xlabel('Numbers of normal references')
                ax.set_ylabel('Score')
                ax.set_title(f"Few-shot ablation study on {dataset_name} ({key})")
                ax.grid(linestyle='--', color='gray', alpha=0.5)
                ax.legend()
                fig.savefig(f"ablation_NR_{dataset_name}_{key}.png", bbox_inches='tight')
                plt.close(fig)
                
def eval_window_size(plot_enable=False):
    period_list = [1,2,3,4]
    dataset_list = ['UCR']
    eval_log_root = f"/home/zhuangjiaxin/workspace/TensorTSL/TimeLLM/log/Ablation/window_size"
    processed_data_root = "/home/zhuangjiaxin/workspace/TensorTSL/TimeLLM/output/Ablation"
    default_confidence = 9
    default_PAT = 0
    for dataset_name in dataset_list:
        metrics_map = {
            'Pred': {'Pre': [], 'Rec': [], 'F1': [], 'AUC_PR': [], 'AUC_ROC': []},
            'Pred_adjust': {'Pre': [], 'Rec': [], 'F1': [], 'AUC_PR': [], 'AUC_ROC': []},
            'DCheck': {'Pre': [], 'Rec': [], 'F1': [], 'AUC_PR': [], 'AUC_ROC': []},
            'DCheck_adjust': {'Pre': [], 'Rec': [], 'F1': [], 'AUC_PR': [], 'AUC_ROC': []},
        }
        for period in period_list:
            if period == 3:
                metrics = nr_3_map[dataset_name]
            else:
                name = f"{dataset_name}-{period}T"
                # data_path = os.path.join(processed_data_root, f"")
                data_path = processed_data_root
                log_path = os.path.join(eval_log_root, f"Ablation-{dataset_name}-window_size-{period}T")
                evaluator = Evaluator(dataset_name, period, data_path, log_root=log_path, processed_path_name=name)
                metrics = evaluator.calculate_f1_aucpr_aucroc(default_confidence, default_PAT)
            for key in metrics.keys():
                name = f"{key}-{period}"
                pre = metrics[key]['Pre']
                rec = metrics[key]['Rec']
                f1 = metrics[key]['F1']
                aucpr = metrics[key]['AUC_PR']
                aucroc = metrics[key]['AUC_ROC']
                metrics_map[key]['Pre'].append(pre)
                metrics_map[key]['Rec'].append(rec)
                metrics_map[key]['F1'].append(f1)
                metrics_map[key]['AUC_PR'].append(aucpr)
                metrics_map[key]['AUC_ROC'].append(aucroc)
        for key in metrics_map.keys():
            tabel = [
                ['Name', 'P', 'R', 'F1', 'AUCPR', 'AUCROC'],
                [f"{key}-1T", f"{metrics_map[key]['Pre'][0]:.3f}", f"{metrics_map[key]['Rec'][0]:.3f}", f"{metrics_map[key]['F1'][0]:.3f}", f"{metrics_map[key]['AUC_PR'][0]:.3f}", f"{metrics_map[key]['AUC_ROC'][0]:.3f}"],
                [f"{key}-2T", f"{metrics_map[key]['Pre'][1]:.3f}", f"{metrics_map[key]['Rec'][1]:.3f}", f"{metrics_map[key]['F1'][1]:.3f}", f"{metrics_map[key]['AUC_PR'][1]:.3f}", f"{metrics_map[key]['AUC_ROC'][1]:.3f}"],
                [f"{key}-3T", f"{metrics_map[key]['Pre'][2]:.3f}", f"{metrics_map[key]['Rec'][2]:.3f}", f"{metrics_map[key]['F1'][2]:.3f}", f"{metrics_map[key]['AUC_PR'][2]:.3f}", f"{metrics_map[key]['AUC_ROC'][2]:.3f}"],
                [f"{key}-4T", f"{metrics_map[key]['Pre'][3]:.3f}", f"{metrics_map[key]['Rec'][3]:.3f}", f"{metrics_map[key]['F1'][3]:.3f}", f"{metrics_map[key]['AUC_PR'][3]:.3f}", f"{metrics_map[key]['AUC_ROC'][3]:.3f}"],
            ]
            print(f"\nDataset: {dataset_name}, Method: {key}")
            print(tabulate.tabulate(tabel, headers='firstrow', tablefmt='fancy_grid'))
            if plot_enable:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(period_list, metrics_map[key]['F1'], label='F1', marker='x')
                ax.plot(period_list, metrics_map[key]['AUC_PR'], label='AUC-PR', marker='o')
                ax.plot(period_list, metrics_map[key]['AUC_ROC'], label='AUC-ROC', marker='s')
                ax.set_xticks(period_list)
                ax.set_xlim([1, 4])
                ax.set_ylim([0, 1])
                ax.set_xlabel('window size (period)')
                ax.set_ylabel('Score')
                ax.set_title(f"Window size ablation study on {dataset_name} ({key})")
                ax.grid(linestyle='--', color='gray', alpha=0.5)
                ax.legend()
                fig.savefig(f"ablation_window_size_{dataset_name}_{key}.png", bbox_inches='tight')
                plt.close(fig)
            
def plot_image():
    pass

if __name__=='__main__':
    # eval_normal_reference(True)
    eval_window_size(True)