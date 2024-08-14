import os
from Datasets.Dataset import EvalDataLoader
import make_dataset

# configuration
dataset_name = 'SMD'
vote_thres = 1
point_adjust_enable = True
plot_enable = True
processed_data_root = f'./output/'
log_root = f'./log/'
dataset_info = make_dataset.dataset_config[dataset_name]
window_size = dataset_info['window']
stride = dataset_info['stride']

# start to evaluate
evaluator = EvalDataLoader(dataset_name, processed_data_root, log_root)
results = evaluator.eval(window_size, stride, vote_thres, 
                         point_adjust_enable=point_adjust_enable, 
                         plot_enable=plot_enable)
