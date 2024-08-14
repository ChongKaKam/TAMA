from Datasets.Dataset import RawDataset, ImageConvertor


id_list = ['MSL', 'psm', 'SMAP', 'SMD', 'SWaT', 'UCR', 'wadi']
dataset_config = {
    'MSL': {
        'sample_rate': 1,
        'normalization_enable': True,
        'window': 500,
        'stride': 250,
    },
    'psm': {
        'sample_rate': 1,
        'normalization_enable': True,
        'window': 500,
        'stride': 250,
    },
    'SMAP': {
        'sample_rate': 1,
        'normalization_enable': True,
        'window': 1000,
        'stride': 500,
    },
    'SMD': { # 23694 - 28479, 38
        'sample_rate': 1,
        'normalization_enable': True,
        'window': 9000,
        'stride': 3000,
        'image_config': {
            'width': 1200,
            'height': 320,
            'dpi': 100,
            'x_ticks': 250,
            'aux_enable': False,
        }
    },
    'SWaT': { # 449919
        'sample_rate': 1,
        'normalization_enable': False,
        'window': 20000,
        'stride': 10000,
        'image_config': {
            'width': 2400,
            'height': 320,
            'dpi': 100,
            'x_ticks': 500,
            'aux_enable': False,
        }
    },
    'UCR': { # 6301
        'sample_rate': 1,
        'normalization_enable': True,
        'window': 600,
        'stride': 300,
        'image_config': {
            'width': 2400,
            'height': 320,
            'dpi': 100,
            'x_ticks': 5,
            'aux_enable': True,
        }
    },
    'wadi': {
        'sample_rate': 1,
        'normalization_enable': True,
        'window': 1000,
        'stride': 500,
    },
}
if __name__ == '__main__':
    mode = 'test'
    # dataset_id = ''
    # for id in dataset_config:
    #     item = dataset_config[id]
    #     sample_rate = item['sample_rate']
    #     window = item['window']
    #     stride = item['stride']
    #     normal_enable = item['normalization_enable']
    #     output_dir = f'./output/'
    #     dataset = RawDataset(id, sample_rate=sample_rate, normalization_enable=normal_enable)
    #     dataset.convert_data(output_dir, mode, window, stride, ImageConvertor)
    id = 'SMD'
    item = dataset_config[id]
    sample_rate = item['sample_rate']
    window = item['window']
    stride = item['stride']
    normal_enable = item['normalization_enable']
    output_dir = f'./output/'
    dataset = RawDataset(id, sample_rate=sample_rate, normalization_enable=normal_enable)
    dataset.convert_data(output_dir, mode, window, stride, ImageConvertor, item['image_config'])