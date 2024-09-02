import yaml
import os
from Datasets.Dataset import RawDataset, ImageConvertor, TextConvertor
import argparse

def args_parse():
    parser = argparse.ArgumentParser(description='Command line interface for main.py')
    parser.add_argument('--dataset', type=str, default='NormA', help='Dataset name')
    parser.add_argument('--data_id_list', type=str, default='', help='Data id list')
    parser.add_argument('--window_size', type=int, default=600, help='Window size',required=False)
    parser.add_argument('--stride', type=int, default=200, help='Stride', required=False)
    parser.add_argument('--mode', type=str, default='test', help='Mode', required=False)
    parser.add_argument('--modality', type=str, default='image', help='Modality: [image, text]', required=False)
    args = parser.parse_args()
    # dump data_id_list_str
    data_id_list_str = args.data_id_list
    data_id_list = data_id_list_str.strip('][').replace(' ', '').split(',')
    args.data_id_list = data_id_list
    if args.modality == 'text':
        args.modality = TextConvertor
    elif args.modality == 'image':
        args.modality = ImageConvertor
    else:
        raise ValueError(f'Invalid modality: {args.modality}')
    return args

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
            'height': 400,
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
    'NAB': { # 17281
        'sample_rate': 1,
        'normalization_enable': True,
        'window': 4200,
        'stride': 1400,
        'drop_last': False,
        'image_config': {
            'width': 1800,
            'height': 400,
            'x_ticks': 150,
            'dpi': 100,
            'aux_enable': True,
        }
    },
    'KDD-TSAD': { # 17281
        'sample_rate': 1,
        'normalization_enable': True,
        'window': 1200,
        'stride': 400,
        'drop_last': False,
        'image_config': {
            'width': 1800,
            'height': 400,
            'x_ticks': 10,
            'dpi': 100,
            'aux_enable': True,
        }
    },
    'NormA': { 
        'sample_rate': 1,
        'normalization_enable': True,
        'window': 900,
        'stride': 300,
        'drop_last': False,
        'image_config': {
            'width': 2000,
            'height': 400,
            'x_ticks': 5,
            'dpi': 100,
            'aux_enable': True,
        }
    },
    'NASA-MSL':{ 
        'sample_rate': 1,
        'normalization_enable': True,
        'window': 600,
        'stride': 200,
        'drop_last': False,
        'image_config': {
            'width': 1800,
            'height': 400,
            'x_ticks': 10,
            'dpi': 100,
            'aux_enable': True,
        }
    },
    'NASA-SMAP':{ 
        'sample_rate': 1,
        'normalization_enable': True,
        'window': 600,
        'stride': 200,
        'drop_last': False,
        'image_config': {
            'width': 1800,
            'height': 400,
            'x_ticks': 10,
            'dpi': 100,
            'aux_enable': True,
        }
    },
}
if __name__ == '__main__':
    args = args_parse()
    mode = args.mode
    dataset_name = args.dataset
    data_id_list = args.data_id_list
    window = args.window_size
    stride = args.stride
    modality_convertor = args.modality
    # configuration
    task_config = yaml.safe_load(open('./configs/task_config.yaml', 'r'))
    output_dir = task_config['processed_data_path']
    
    item = dataset_config[dataset_name]
    sample_rate = item['sample_rate']
    # window = item['window']
    # stride = item['stride']
    normal_enable = item['normalization_enable']
    drop_last = item.get('drop_last', False)

    image_config = item['image_config']
    image_config['width'] = (2000/600 * window)
    
    if args.modality == TextConvertor:
        image_config = {}

    dataset = RawDataset(dataset_name, sample_rate=sample_rate, normalization_enable=normal_enable)
    dataset.convert_data(output_dir, mode, window, stride, modality_convertor, image_config, drop_last=drop_last, data_id_list=data_id_list)