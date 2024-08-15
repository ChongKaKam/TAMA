from Datasets.Dataset import RawDataset, ImageConvertor


id_list = ['MSL', 'psm', 'SMAP', 'SMD', 'SWaT', 'UCR', 'wadi']
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
        'window': 50000,
        'stride': 25000,
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
        'stride': 300,
        'image_config': {
            'width': 2400,
            'height': 320,
            'dpi': 100,
            'x_ticks': 5,
            'aux_enable': True,
        }
    },
    'wadi': { # 17281
        'sample_rate': 1,
        'normalization_enable': True,
        'window': 10000,
        'stride': 7000,
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
    id = 'SWaT'
    item = dataset_config[id]
    sample_rate = item['sample_rate']
    window = item['window']
    stride = item['stride']
    normal_enable = item['normalization_enable']
    output_dir = f'./output/'
    dataset = RawDataset(id, sample_rate=sample_rate, normalization_enable=normal_enable)
    dataset.convert_data(output_dir, mode, window, stride, ImageConvertor, item['image_config'])