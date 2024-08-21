import yaml
import os
from Datasets.Dataset import RawDataset, ImageConvertor
# configuration
dataset_config = {
    'MSL': { # 1096 - 2264 - 6100
        'sample_rate': 1,
        'normalization_enable': True,
        'window': 500,
        'stride': 250,
    },
    'psm': { # 87840
        'sample_rate': 1,
        'normalization_enable': True,
        'window': 500,
        'stride': 250,
    },
    'SMAP': { # 4693 - 8640
        'sample_rate': 1,
        'normalization_enable': True,
        'window': 1000,
        'stride': 500,
    },
    'SMD': { # 23694 - 28479, 38
        'sample_rate': 1,
        'normalization_enable': True,
        'window': 7500,
        'stride': 2500,
        'drop_last': False,
        'image_config': {
            'width': 2400,
            'height': 320,
            'x_ticks': 50,
            'dpi': 100,
            'aux_enable': True,
        }
    },
    'SWaT': { # 449919
        'sample_rate': 1,
        'normalization_enable': False,
        'window': 50000,
        'stride': 25000,
        'drop_last': False,
        'image_config': {
            'width': 2400,
            'height': 320,
            'dpi': 100,
            'x_ticks': 1000,
            'aux_enable': False,
        }
    },
    'UCR': { # 6301
        'sample_rate': 1,
        'normalization_enable': True,
        'window': 600,
        'stride': 200,
        'drop_last': False,
        'image_config': {
            'width': 2000,
            'height': 320,
            'dpi': 100,
            'x_ticks': 5,
            'aux_enable': True,
        }
    },
    'wadi': { # 17281
        'sample_rate': 1,
        'normalization_enable': True,
        'window': 6450,
        'stride': 2150,
        'drop_last': False,
        'image_config': {
            'width': 1800,
            'height': 400,
            'x_ticks': 150,
            'dpi': 100,
            'aux_enable': True,
        }
    },
}
if __name__ == '__main__':
    # configuration
    task_config = yaml.safe_load(open('./configs/task_config.yaml', 'r'))
    mode = 'test'
    dataset_name = task_config['dataset_name']
    data_id_list = task_config['data_id_list']
    output_dir = task_config['processed_data_path']
    
    item = dataset_config[dataset_name]
    sample_rate = item['sample_rate']
    window = item['window']
    stride = item['stride']
    normal_enable = item['normalization_enable']
    drop_last = item.get('drop_last', False)
    
    dataset = RawDataset(dataset_name, sample_rate=sample_rate, normalization_enable=normal_enable)
    dataset.convert_data(output_dir, mode, window, stride, ImageConvertor, item['image_config'], drop_last=drop_last, data_id_list=data_id_list)