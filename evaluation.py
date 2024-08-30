import os
from Datasets.Dataset import EvalDataLoader, Evaluator
import make_dataset

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
log_root = f'./log/log_nas/anormaly_detection/UCR'
image_config= {
    'width': 2000,
    'height': 480,
    'x_ticks': 50,
    'dpi': 100,
}
dataset_name = 'NASA-SMAP'
dataset_info = make_dataset.dataset_config[dataset_name]
window_size = dataset_info['window']
stride = dataset_info['stride']
config = {
    "UCR": {
        'log_root': './log/log_nas/anormaly_detection/UCR',
        'data_id_list': [],
    },
    "NASA-MSL": {
        'log_root': './log/log_nas/anormaly_detection/NASA-MSL-All-1',
        'data_id_list': [],
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
# start to evaluate
# evaluator = EvalDataLoader(dataset_name, processed_data_root, log_root)
# evaluator.set_plot_config(**image_config)
# results = evaluator.eval(window_size, stride, vote_thres, 
#                          point_adjust_enable=point_adjust_enable, 
#                          plot_enable=plot_enable,
#                          channel_shared=channel_shared)

evaluator = Evaluator(dataset_name, stride, processed_data_root, config[dataset_name]['log_root'])
# evaluator.calculate_f1_score(confidence_thres=9, data_id_list=config[dataset_name]['data_id_list'])
evaluator.calculate_roc_pr_auc(config[dataset_name]['data_id_list'])
# evaluator.calculate_adjust_PR_curve_auc(config[dataset_name]['data_id_list'])