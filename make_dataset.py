from Datasets.Dataset import RawDataset, ImageConvertor

if __name__ == '__main__':
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
        'SMD': {
            'sample_rate': 0.5,
            'normalization_enable': True,
            'window': 1000,
            'stride': 500,
        },
        'SWaT': {
            'sample_rate': 0.01,
            'normalization_enable': True,
            'window': 1000,
            'stride': 500,
        },
        'UCR': {
            'sample_rate': 1,
            'normalization_enable': True,
            'window': 320,
            'stride': 160,
        },
        'wadi': {
            'sample_rate': 1,
            'normalization_enable': True,
            'window': 1000,
            'stride': 500,
        },
    }
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
    id = 'UCR'
    item = dataset_config[id]
    sample_rate = item['sample_rate']
    window = item['window']
    stride = item['stride']
    normal_enable = item['normalization_enable']
    output_dir = f'./output/'
    dataset = RawDataset(id, sample_rate=sample_rate, normalization_enable=normal_enable)
    dataset.convert_data(output_dir, mode, window, stride, ImageConvertor)