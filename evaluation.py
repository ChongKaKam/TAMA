import os
from Datasets.Dataset import EvalDataLoader, Evaluator
import numpy as np
import make_dataset
import tabulate
import yaml
import matplotlib.pyplot as plt
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
dataset_name = 'NASA-MSL'
# key_name = 'DCheck' + '_adjust'
# dataset_info = make_dataset.dataset_config[dataset_name]
# window_size = dataset_info['window']
# stride = dataset_info['stride']
config = {
    "UCR": {
        # 'log_root': './log/gpt-4o-mini',
        # 'log_root': './log/log_nas/anomaly_detection/UCR',
        # 'log_root': '/home/zhuangjiaxin/workspace/TensorTSL/TimeLLM/log/',
        'log_root': '/home/zhuangjiaxin/workspace/TensorTSL/TimeLLM/log/TextModality',
        # 'log_root': '/home/zhuangjiaxin/workspace/TensorTSL/TimeLLM/log/AuxLine-3', 
        # 'log_root': '/home/zhuangjiaxin/workspace/TensorTSL/TimeLLM/log/TAMA-3',
        'data_id_list': ['135','136','137','138'],
    },
    "NASA-MSL": {
        # 'log_root': './log/log_nas/anomaly_detection/NASA-MSL-All-1',
        # 'log_root': '/home/zhuangjiaxin/workspace/TensorTSL/TimeLLM/log/Rotation',
        # 'log_root': '/home/zhuangjiaxin/workspace/TensorTSL/TimeLLM/log/TextModality',
        'log_root': '/home/zhuangjiaxin/workspace/TensorTSL/TimeLLM/log/text_modality',
        'data_id_list': ['3', '9', '10', '11', '15', '23', '24'],
    },
    "NormA": {
        # 'log_root': './log/log_nas/anomaly_detection/NormA-1_4_7_13',
        'log_root': '/home/zhuangjiaxin/workspace/TensorTSL/TimeLLM/log/TextModality',
        'data_id_list': ['1', '4', '7', '13'],
    },
    "NASA-SMAP": {
        # 'log_root': './log/', 
        # 'log_root': '/home/zhuangjiaxin/workspace/TensorTSL/TimeLLM/log/AuxLine-1',
        'log_root': '/home/zhuangjiaxin/workspace/TensorTSL/TimeLLM/log/TextModality',
        # 'log_root': '/home/zhuangjiaxin/workspace/TensorTSL/TimeLLM/log/TAMA-3',
        # 'log_root': './log/log_nas/anomaly_detection/NASA-SMAP-2_24_27_37_45',
        # 'log_root': '/home/zhuangjiaxin/workspace/TensorTSL/TimeLLM/log/text_modality',
        'data_id_list': ['2', '24', '27', '37', '45'],
    },
    "synthetic_datasets": {
        'log_root': './log/log_nas/anomaly_detection/synthetic_datasets-all-2',
        'data_id_list': ['ecg-frequency-0', 'ecg-frequency-1', 'ecg-frequency-2', 'square-frequency-0'],
    },
    "Dodgers": {
        # 'log_root': './log/log_nas/anomaly_detection/Dodgers-v2',
        'log_root': '/home/zhuangjiaxin/workspace/TensorTSL/TimeLLM/log/TextModality',
        'data_id_list': ['101-freeway-traffic'],
    },
    "ECG": {
        # 'log_root': './log/log_nas/anomaly_detection/ECG-v1',
        'log_root': '/home/zhuangjiaxin/workspace/TensorTSL/TimeLLM/log/TextModality',
        'data_id_list': ['CS-MBA-ECG803-data', 'CS-MBA-ECG806-data', 'CS-MBA-ECG820-data', 'RW-MBA-ECG14046-data-12', 'RW-MBA-ECG14046-data-44', 'RW-MBA-ECG803-data', 'WN-MBA-ECG14046-data-12', 'WN-MBA-ECG14046-data-5', 'WN-MBA-ECG803-data'],
    },
    "MSD-1": {
        # 'log_root': './log/MSD-1-machine-1-1',
        'log_root': '/home/zhuangjiaxin/workspace/TensorTSL/TimeLLM/log/TextModality',
        'data_id_list': ['machine-1-1-10', 'machine-1-1-11', 'machine-1-1-12', 'machine-1-1-13', 'machine-1-1-14', 'machine-1-1-15', 'machine-1-1-23', 'machine-1-1-25', 'machine-1-1-26', 'machine-1-1-28', 'machine-1-1-32', 'machine-1-1-33', 'machine-1-1-5', 'machine-1-1-6', 'machine-1-1-8', 'machine-1-1-9'],
    },
}
def old_eval():
    dataset_name = 'MSD-1'
    dataset_info = make_dataset.dataset_config[dataset_name]
    window_size = dataset_info['window']
    stride = dataset_info['stride']
    log_path =  config[dataset_name]['log_root']
    eval = EvalDataLoader(dataset_name, processed_data_root, log_root=log_path)
    eval.set_plot_config(**image_config)
    eval.eval(window_size, stride, 2, point_adjust_enable=True, plot_enable=True, channel_shared=False)

def evaluate_each_data_id():
    dataset_name = 'NormA'
    subtask_name = 'TextModality'
    processed_data_root = os.path.join('/home/zhuangjiaxin/workspace/TensorTSL/TimeLLM/output/', subtask_name)
    dataset_info = make_dataset.dataset_config[dataset_name]
    window_size = dataset_info['window']
    # stride = dataset_info['stride']
    stride = 150
    key_name = 'Pred' + '_adjust'
    log_path =  config[dataset_name]['log_root']
    print(f'SubTask: {subtask_name}, {dataset_name}, log: {log_path}, stride: {stride}')
    print(f'Processed data root: {processed_data_root}')
    # input()
    default_confidence_thres = 9
    evaluator = Evaluator(dataset_name, stride, processed_data_root, log_root=log_path)
    # evaluator.calculate_TP_FP_TN_FN(default_confidence_thres, 0, data_id_list=config[dataset_name]['data_id_list'], show_results=False)
    # res = evaluator.calculate_TP_FP_TN_FN(confidence_thres=default_confidence_thres, data_id_list=config[dataset_name]['data_id_list'],show_results=False)
    pre_list = []
    rec_list = []
    f1_list = []
    auc_pr_list = []
    auc_roc_list = []
    for data_id in config[dataset_name]['data_id_list']:
        print(data_id)
        metrics = evaluator.calculate_f1_aucpr_aucroc(default_confidence_thres, 0, data_id_list=[data_id])
        # print(metrics)
        tabel = [
            ['Name', 'P', 'R', 'F1', 'AUCPR', 'AUCROC']
        ]
        for name in metrics:
            pre = metrics[name]['Pre']
            rec = metrics[name]['Rec']
            f1 = metrics[name]['F1']
            aucpr = metrics[name]['AUC_PR']
            aucroc = metrics[name]['AUC_ROC']
            tabel.append([name, f"{pre:.3f}", f"{rec:.3f}", f"{f1:.3f}", f"{aucpr:.3f}", f"{aucroc:.3f}"])
        pre_list.append(metrics[key_name]['Pre'])
        rec_list.append(metrics[key_name]['Rec'])
        f1_list.append(metrics[key_name]['F1'])
        auc_pr_list.append(metrics[key_name]['AUC_PR'])
        auc_roc_list.append(metrics[key_name]['AUC_ROC'])

        print(tabulate.tabulate(tabel, headers='firstrow', tablefmt='fancy_grid'))

    res = evaluator.calculate_f1_aucpr_aucroc(default_confidence_thres, 0, data_id_list=config[dataset_name]['data_id_list'])
    table = [
        ['Name', 'P', 'R', 'F1', 'AUCPR', 'AUCROC']
    ]
    for name in res:
        pre = res[name]['Pre']
        rec = res[name]['Rec']
        f1 = res[name]['F1']
        aucpr = res[name]['AUC_PR']
        aucroc = res[name]['AUC_ROC']
        table.append([name, f"{pre:.3f}", f"{rec:.3f}", f"{f1:.3f}", f"{aucpr:.3f}", f"{aucroc:.3f}"])
    print(tabulate.tabulate(table, headers='firstrow', tablefmt='fancy_grid'))

    print(f"Method: {key_name}")
    # print(pre_list)
    print(f'Precision: {max(pre_list):.3f} / {res[key_name]["Pre"]:.3f} / {np.std(pre_list):.3f}')
    print(f'Recall: {max(rec_list):.3f} /  {res[key_name]["Rec"]:.3f} / {np.std(rec_list):.3f}')
    print(f'F1: {max(f1_list):.3f} / {res[key_name]["F1"]:.3f} / {np.std(f1_list):.3f}')
    print(f'AUC_PR: {max(auc_pr_list):.3f} / {res[key_name]["AUC_PR"]:.3f} / {np.std(auc_pr_list):.3f}')
    print(f'AUC_ROC: {max(auc_roc_list):.3f} / {res[key_name]["AUC_ROC"]:.3f} / {np.std(auc_roc_list):.3f}')

# evaluator.calculate_roc_pr_auc(config[dataset_name]['data_id_list'])

def plot_AUC_PR_PAT():
    baseline_info = yaml.safe_load(open('/nas/datasets/ysc/TranAD/Processed_results/Reeval/auc_pr_curves_TimesNet.yaml'))
    save_results = {}
    subtask_name = 'TAMA'
    processed_data_root = os.path.join('/home/zhuangjiaxin/workspace/TensorTSL/TimeLLM/output/', subtask_name)
    for dataset_name in baseline_info:
        if dataset_name == 'SMD' or dataset_name == 'SMD-1':
            dataset_name = 'MSD-1'
        dataset_info = make_dataset.dataset_config[dataset_name]
        window_size = dataset_info['window']
        if dataset_name == 'MSD-1':
            stride = dataset_info['stride']
        else:
            stride = dataset_info['stride']
        log_path =  config[dataset_name]['log_root']
        evaluator = Evaluator(dataset_name, stride, processed_data_root, log_root=log_path)
        ours_auc_pr = evaluator.calculate_adjust_PR_curve_auc(config[dataset_name]['data_id_list'])
        tabel = [
            ['Name', 'thres=0.0', 'thres=0.2', 'thres=0.4', 'thres=0.6', 'thres=0.8', 'thres=1.0'],
            ['Ours', f'{ours_auc_pr[0]:.3f}', f'{ours_auc_pr[1]:.3f}', f'{ours_auc_pr[2]:.3f}', f'{ours_auc_pr[3]:.3f}', f'{ours_auc_pr[4]:.3f}', f'{ours_auc_pr[5]:.3f}']
        ]
        # if dataset_name == 'MSD-1':
        #     dataset_name = 'SMD'
        for baseline_model in baseline_info[dataset_name]:
            baseline_auc_pr = baseline_info[dataset_name][baseline_model]
            baseline_auc_pr.sort(reverse=True)
            tabel.append([baseline_model, f'{baseline_auc_pr[0]:.3f}', f'{baseline_auc_pr[1]:.3f}', f'{baseline_auc_pr[2]:.3f}', f'{baseline_auc_pr[3]:.3f}', f'{baseline_auc_pr[4]:.3f}', f'{baseline_auc_pr[5]:.3f}'])
        print(f"Dataset: {dataset_name}")
        print(tabulate.tabulate(tabel, headers='firstrow', tablefmt='fancy_grid'))
        # save in yaml
        result_item = {
            'Ours': [f'{ours_auc_pr[0]:.3f}', f'{ours_auc_pr[1]:.3f}', f'{ours_auc_pr[2]:.3f}', f'{ours_auc_pr[3]:.3f}', f'{ours_auc_pr[4]:.3f}', f'{ours_auc_pr[5]:.3f}'],
        }
        for baseline_model in baseline_info[dataset_name]:
            baseline_auc_pr = baseline_info[dataset_name][baseline_model]
            baseline_auc_pr.sort(reverse=True)
            result_item[baseline_model] = [f'{baseline_auc_pr[0]:.3f}', f'{baseline_auc_pr[1]:.3f}', f'{baseline_auc_pr[2]:.3f}', f'{baseline_auc_pr[3]:.3f}', f'{baseline_auc_pr[4]:.3f}', f'{baseline_auc_pr[5]:.3f}']
        save_results[dataset_name] = result_item
        x_tick = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        fig, ax = plt.subplots(figsize=(13, 6))
        # ax.plot(ours_auc_pr, label='Ours')
        ax.plot(x_tick, ours_auc_pr, label='Ours', marker='o', alpha=0.8, markersize=4)
        for baseline_model in baseline_info[dataset_name]:
            baseline_auc_pr = baseline_info[dataset_name][baseline_model]
            # ax.plot(baseline_auc_pr, label=baseline_model)
            baseline_auc_pr.sort(reverse=True)
            ax.plot(x_tick, baseline_auc_pr, label=baseline_model, marker='o', alpha=0.8, markersize=4)
        # ax.set_xticks(range(0, 1.01, 0.2))
        ax.set_xlim([0, 1.01])
        ax.set_ylim([0, 1.01])
        ax.set_xlabel('Point-adjustment threshold', fontsize=14)
        ax.set_ylabel('AUC-PR', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_title(f'AUC-PR-PAT curve of {dataset_name}', fontsize=16)
        ax.grid(linestyle='--', color='gray', alpha=0.5)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        fig.savefig(f"point-adjustment-auc-pr-{dataset_name}", bbox_inches='tight')
        plt.close(fig)
    with open('point_adjustment_auc_pr.yaml', 'w') as f:
        yaml.dump(save_results, f)

def evaluate_metrics():
    dataset_name = 'synthetic_datasets'
    subtask_name = ''
    processed_data_root = os.path.join('/home/zhuangjiaxin/workspace/TensorTSL/TimeLLM/output/', subtask_name)
    dataset_info = make_dataset.dataset_config[dataset_name]
    window_size = dataset_info['window']
    stride = dataset_info['stride']
    # stride = 300
    log_path =  config[dataset_name]['log_root']
    # log_path = '/home/zhuangjiaxin/workspace/TensorTSL/TimeLLM/log/synthetic_datasets-all-2'
    print(f'SubTask: {subtask_name}, {dataset_name}, log: {log_path}, stride: {stride}')
    print(f'Processed data root: {processed_data_root}')
    evaluator = Evaluator(dataset_name, stride, processed_data_root, log_root=log_path)

    res = evaluator.calculate_f1_aucpr_aucroc(6, 0, data_id_list=config[dataset_name]['data_id_list'])
    table = [
        ['Name', 'P', 'R', 'F1', 'AUCPR', 'AUCROC']
    ]
    for name in res:
        pre = res[name]['Pre']
        rec = res[name]['Rec']
        f1 = res[name]['F1']
        aucpr = res[name]['AUC_PR']
        aucroc = res[name]['AUC_ROC']
        table.append([name, f"{pre:.3f}", f"{rec:.3f}", f"{f1:.3f}", f"{aucpr:.3f}", f"{aucroc:.3f}"])
    print(tabulate.tabulate(table, headers='firstrow', tablefmt='fancy_grid'))
    # res = evaluator.calculate_TP_FP_TN_FN(3, 0, data_id_list=config[dataset_name]['data_id_list'], show_results=True)

def fix_log():
    from Datasets.Dataset import ProcessedDataset
    import numpy as np
    dataset_name = 'synthetic_datasets'
    original_log = yaml.safe_load(open('/home/zhuangjiaxin/workspace/TensorTSL/TimeLLM/log/log_nas/anomaly_detection/synthetic_datasets-all/synthetic_datasets_log_origin.yaml'))
    processed_data_path = '/home/zhuangjiaxin/workspace/TensorTSL/TimeLLM/output'
    normal_reference_base = os.path.join(processed_data_path, dataset_name)
    dataset = ProcessedDataset(os.path.join(processed_data_path, dataset_name), mode='test')
    for data_id in original_log:
        for stride_idx in original_log[data_id]:
            for ch in original_log[data_id][stride_idx]:
                labels = dataset.get_label(data_id, stride_idx, ch)
                labels_index = np.where(labels >= 1)[0].tolist()
                print(labels_index)
                original_log[data_id][stride_idx][ch]['labels'] = str(labels_index)
    with open('/home/zhuangjiaxin/workspace/TensorTSL/TimeLLM/log/log_nas/anomaly_detection/synthetic_datasets-all/synthetic_datasets_log.yaml', 'w') as f:
        yaml.dump(original_log, f)

def evaluation_with_classification():
    dataset_name = 'NASA-SMAP'
    dataset_info = make_dataset.dataset_config[dataset_name]
    window_size = dataset_info['window']
    stride = dataset_info['stride']
    log_path =  config[dataset_name]['log_root']
    print(f'{dataset_name}, log: {log_path}')
    evaluator = Evaluator(dataset_name, stride, processed_data_root, log_root=log_path)
    default_confidence_thres = 3
    # type_id = 
    for type_id in range(1, 6):
        res = evaluator.calculate_metrics_with_classification('/nas/datasets/VisualTimeSeries/output/classifications',
                                                            default_confidence_thres, type_id, data_id_list=config[dataset_name]['data_id_list'])
        print(f'Type: {type_id}')
        print(res)


if __name__=='__main__':
    # old_eval()
    # evaluate_metrics()
    # input('press any key to continue')
    evaluate_each_data_id() 
    # fix_log()
    # plot_AUC_PR_PAT()
    # evaluation_with_classification()