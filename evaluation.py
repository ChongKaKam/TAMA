import os
from Datasets.Dataset import EvalDataLoader
import make_dataset

# configuration
dataset_name = 'UCR'
vote_thres = 2
point_adjust_enable = True
plot_enable = True
channel_shared = False
processed_data_root = f'./output/'
log_root = f'./log/'
image_config= {
    'width': 2000,
    'height': 480,
    'x_ticks': 50,
    'dpi': 100,
}
dataset_info = make_dataset.dataset_config[dataset_name]
window_size = dataset_info['window']
stride = dataset_info['stride']

# start to evaluate
evaluator = EvalDataLoader(dataset_name, processed_data_root, log_root)
evaluator.set_plot_config(**image_config)
results = evaluator.eval(window_size, stride, vote_thres, 
                         point_adjust_enable=point_adjust_enable, 
                         plot_enable=plot_enable,
                         channel_shared=channel_shared)
