import yaml
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from sklearn.metrics import auc, roc_curve, roc_auc_score
'''
Tools: Register the dataset
Info:  Read datsets in root_path and record key information in dataset.yaml
The format of dataset.yaml:
    - dataset_name:
        - path: xxx/xxx/
        - type: distributed / centralized
        - file_list (distributed):
            - id-1:
                - train: shape
                - test: shape
                - labels: shape
            - id-2:
                - train: shape
                - test: shape
                - labels: shape
        - file_list (centralized):
            - train-id: shape
            - test-id: shape
            - labels-id: shape
        - (TODO)background: the background information of this dataset
'''
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT_PATH = os.path.join(os.path.dirname(CURRENT_PATH), 'data')   # ../data
DEFAULT_YAML_PATH = os.path.join(CURRENT_PATH, 'dataset.yaml')  # ./dataset.yaml
'''
Tools: Register the dataset
'''
def AutoRegister(root_path:str=DEFAULT_ROOT_PATH, yaml_path:str=DEFAULT_YAML_PATH):
    dataset_map = {}
    for dataset_name in os.listdir(root_path):
        dataset_path = os.path.join(root_path, dataset_name)
        file_list = os.listdir(dataset_path)
        # filter
        file_list = list(filter(lambda x: x.endswith('.npy'), file_list))
        file_list = list(filter(lambda x: not x.startswith(dataset_name), file_list))
        file_list.sort()
        # centralized
        if 'train.npy' in file_list and 'test.npy' in file_list and 'labels.npy' in file_list:
            dataset_map[dataset_name] = {
                'path': dataset_path,
                'type': 'centralized',
                'file_list': {
                    'data': {
                        'train': list(np.load(os.path.join(dataset_path, 'train.npy')).shape),
                        'test': list(np.load(os.path.join(dataset_path, 'test.npy')).shape),
                        'labels': list(np.load(os.path.join(dataset_path, 'labels.npy')).shape),
                    }
                },
                'background': '',
            }
        # dataset with meta_data.yaml
        elif 'meta_data.yaml' in os.listdir(dataset_path):
            meta_info = yaml.safe_load(open(os.path.join(dataset_path, 'meta_data.yaml'), 'r'))
            data_id_list = meta_info['mapping'].keys()
            id_map = {}
            for id_name in data_id_list:
                id_name = str(id_name)
                if id_name not in dataset_map:
                    id_map[id_name] = {}
                if os.path.exists(os.path.join(dataset_path, f'{id_name}.train.npy')):
                    id_map[id_name]['train'] = list(np.load(os.path.join(dataset_path, f'{id_name}.train.npy')).shape)
                if os.path.exists(os.path.join(dataset_path, f'{id_name}.test.npy')):
                    id_map[id_name]['test'] = list(np.load(os.path.join(dataset_path, f'{id_name}.test.npy')).shape)
                if os.path.exists(os.path.join(dataset_path, f'{id_name}.labels.npy')):
                    id_map[id_name]['labels'] = list(np.load(os.path.join(dataset_path, f'{id_name}.labels.npy')).shape)
                dataset_map[dataset_name] = {
                    'path': dataset_path,
                    'type': 'meta_data',
                    'file_list': id_map,
                    'background': '',
                }
        # distributed
        else:
            id_map = {}
            for file_name in file_list:
                id_name = file_name.split('.')[0].split('_')[0]
                if id_name not in id_map:
                    id_map[id_name] = {}
                id_map[id_name][file_name.split('.')[0].split('_')[1]] = list(np.load(os.path.join(dataset_path, file_name)).shape)
            dataset_map[dataset_name] = {
                'path': dataset_path,
                'type': 'distributed',
                'file_list': id_map,
                'background': '',
            }
        # dump
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_map, f)

    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_map, f)

'''
Tools: check dataset shape
'''
def check_shape(dataset_name:str, root_path:str=DEFAULT_ROOT_PATH):
    dataset_path = os.path.join(root_path, dataset_name)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset {dataset_name} not found in {root_path}")
    files = os.listdir(dataset_path)
    shape_map = {}
    for file in files:
        data = np.load(os.path.join(dataset_path, file))
        shape_map[file] = data.shape
    for i in shape_map:
        print(i, shape_map[i])

'''
Info: Base Class. Convert the data to the format that the model can use
'''
class ConvertorBase:
    output_type = 'index'
    def __init__(self, save_path:str):
        self.save_path = save_path
        self.normal_save_path = os.path.join(self.save_path, 'normal')
        self.abnormal_save_path = os.path.join(self.save_path, 'abnormal')
        self.ensure_dir()
    def ensure_dir(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        if not os.path.exists(self.normal_save_path):
            os.makedirs(self.normal_save_path)
        if not os.path.exists(self.abnormal_save_path):
            os.makedirs(self.abnormal_save_path)

    def convert_and_save(self, data, name:str):
        return data
    def load(self, idx:int)->dict:
        res = {
            'type': 'index',
            'data': idx,
        }
        return res
'''
Info: Convert time series data to image
'''
class ImageConvertor(ConvertorBase):
    output_type = 'image'
    def __init__(self, save_path:str):
        super().__init__(save_path)
        self.width = 1500
        self.height = 320
        self.dpi = 100
        self.x_ticks = 100
        self.aux_enable = True
        self.line_color = 'blue'
        self.x_rotation = 90
        plt.rcParams.update({'font.size': 8})
        # convert to inches
        self.figsize = (self.width/self.dpi, self.height/self.dpi)

    def set_config(self, width:int, height:int, dpi:int, x_ticks:int, aux_enable:bool):
        self.width = width
        self.height = height
        self.dpi = dpi
        self.x_ticks = x_ticks
        self.aux_enable = aux_enable
        # convert to inches
        self.figsize = (self.width/self.dpi, self.height/self.dpi)

    def convert_and_save(self, data, name:int, separate:str = ''):
        # check if the shape of data is correct
        if len(data.shape) > 2:
            raise ValueError(f"Only accept 1D data, but got {(data.shape)}")
        else:
            data_checked = data.reshape(-1)
        # new figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        # auxiliary line
        if self.aux_enable:
            for x in range(0, len(data_checked)+1, self.x_ticks):
                ax.axvline(x=x, color='lightgray', linestyle='--', linewidth=0.5)
        ax.plot(data_checked, color=self.line_color)
        plt.xticks(rotation=self.x_rotation)
        ax.set_xticks(range(0, len(data_checked)+1, self.x_ticks))
        ax.set_xlim(0, len(data_checked)+1)
        fig.tight_layout(pad=0.1)
        if separate == 'normal':
            fig.savefig(os.path.join(self.normal_save_path, f"{name}.png"), bbox_inches='tight')
        elif separate == 'abnormal':
            fig.savefig(os.path.join(self.abnormal_save_path, f"{name}.png"), bbox_inches='tight')
        else:
            fig.savefig(os.path.join(self.save_path, f"{name}.png"), bbox_inches='tight')
        plt.close()

    def load(self, name:int):
        image_path = os.path.join(self.save_path, f"{name}.png")
        res = {
            'type': 'image',
            'data': image_path
        }
        return res
'''
Info: Convert time series data to text
'''
class TextConvertor(ConvertorBase):
    output_type = 'text'
    def __init__(self, save_path:str):
        super().__init__(save_path)
    def convert_and_save(self, data, name:int):
        with open(os.path.join(self.save_path, f"{name}.txt"), 'w') as f:
            formatted_data = self.format(data)
            f.write(formatted_data)
    def load(self, name:int):
        text_path = os.path.join(self.save_path, f"{name}.txt")
        res = {
            'type': 'text',
            'data': text_path
        }
        return res
    def format(self, data):
        formatted_data = np.array2string(data, separator=',', precision=3, suppress_small=True)
        return formatted_data
'''
Utils
'''
# padding with nan
def padding(array, target_len):
    current_len = array.shape[0]
    channels = 1 if len(array.shape) == 1 else array.shape[1]
    if current_len >= target_len:
        return array
    else:
        padding_len = target_len - current_len
        padding_array = np.full((padding_len, channels), np.nan)
        return np.concatenate((array, padding_array), axis=0)
# remove nan padding
def remove_padding(array):
    if len(array.shape) == 1:
        return array[~np.isnan(array)]
    else:
        return array[~np.isnan(array).all(axis=1)]
'''
Raw Data Loader
'''

class RawDataset:
    def __init__(self, dataset_name:str, sample_rate:float=1, normalization_enable:bool=True, yaml_path:str=DEFAULT_YAML_PATH) -> None:
        self.dataset_name = dataset_name
        self.yaml_path = yaml_path
        dataset_map = yaml.safe_load(open(self.yaml_path, 'r'))
        if dataset_name not in dataset_map:
            raise FileNotFoundError(f"Dataset {dataset_name} not found in {yaml_path}")
        self.dataset_info = dataset_map[dataset_name]
        self.sample_rate = sample_rate
        self.normalization_enable = normalization_enable

    def get_background_info(self):
        return str(self.dataset_info['background'])
    
    def ensure_dir(self, path:str):
        if not os.path.exists(path):
            os.makedirs(path)

    def sampling(self, data):
        interval = int(1/self.sample_rate)
        return data[::interval]

    def normalize(self, data):
        mean = np.mean(data)
        std = np.std(data)
        return (data - mean) / std

    def separate_label_make(self, dataset_output_dir, id, mode, window_size, stride, convertor_class, image_config:dict={}, drop_last:bool=True):
        if id == 'data':
            data_path = os.path.join(self.dataset_info['path'], f"{mode}.npy")
        else:
            data_path = os.path.join(self.dataset_info['path'], f"{id}_{mode}.npy")
            if not os.path.exists(data_path):
                id = str(id)
                data_path = os.path.join(self.dataset_info['path'], f"{id}.{mode}.npy")
        
        # 1. sampling & normalization
        data = np.load(data_path)
        data = self.sampling(data)
        if self.normalization_enable:
            data = self.normalize(data)

        # 2. get the save path of the convertor & init the convertor
        convertor_save_path = os.path.join(dataset_output_dir, id, mode, convertor_class.output_type)
        self.ensure_dir(convertor_save_path)
        convertor = convertor_class(save_path=convertor_save_path)
        if image_config != {}:
            convertor.set_config(**image_config)

        # 3. convert & save
        # data .npy format: [num_stride, window_size, data_channels]
        window_data_save_path = os.path.join(dataset_output_dir, id, mode, 'data.npy')
        window_data_array = []
        num_stride = (len(data)-window_size) // stride + 1 if len(data) >= window_size else 1
        data_channels = 1 if len(data.shape) == 1 else data.shape[1]
        if data_channels == 1:
            data = data.reshape(-1, 1)
        # label
        window_label_save_path = os.path.join(dataset_output_dir, id, mode, 'labels.npy')
        if mode == 'test':
            if id == 'data':
                labels_path = os.path.join(self.dataset_info['path'], f"labels.npy")
            else:
                labels_path = os.path.join(self.dataset_info['path'], f"{id}_labels.npy")
                if not os.path.exists(labels_path):
                    labels_path = os.path.join(self.dataset_info['path'], f"{id}.labels.npy")
            # sampling 
            labels = np.load(labels_path)
            labels = self.sampling(labels)
            # label .npy format: [num_stride, window_size, label_channels]
            window_label_array = []
            label_channels = 1 if len(labels.shape) == 1 else labels.shape[1]
            if label_channels == 1:
                labels = labels.reshape(-1, 1)
            for i in range(num_stride):
                start = i * stride
                window_label = labels[start:start+window_size]
                window_label = padding(window_label, window_size)
                window_label_array.append(window_label)
            if not drop_last:
                start = num_stride * stride
                window_label = labels[start:]
                window_label = padding(window_label, window_size)
                window_label_array.append(window_label)
            window_label_array = np.array(window_label_array)
            np.save(window_label_save_path, window_label_array)
        # data
        separate_id_list = {'normal': [], 'abnormal': [],'save_path': convertor.save_path}
        for i in range(num_stride):
            start = i * stride
            window_data = data[start:start+window_size]
            window_data = padding(window_data, window_size)
            window_data_array.append(window_data)
            # convert & save
            for ch in range(data_channels):
                if mode == 'test'and window_label_array[i].sum() == 0:
                    separate_id_list['normal'].append(f'{id}-{i}-{ch}')
                    # convertor.convert_and_save(window_data[:,ch], f'{i}-{ch}', separate='normal')
                else:
                    separate_id_list['abnormal'].append(f'{id}-{i}-{ch}')
                    # convertor.convert_and_save(window_data[:,ch], f'{i}-{ch}', separate='abnormal')
                convertor.convert_and_save(remove_padding(window_data[:,ch]), f'{i}-{ch}')
        if not drop_last:
            start = num_stride * stride
            window_data = data[start:]
            padded_window_data = padding(window_data, window_size)
            window_data_array.append(padded_window_data)
            for ch in range(data_channels):
                convertor.convert_and_save(remove_padding(window_data[:,ch]), f'{num_stride}-{ch}')

        window_data_array = np.array(window_data_array)
        np.save(window_data_save_path, window_data_array)
        
        # background
        background_save_path = os.path.join(dataset_output_dir, 'background.txt')
        with open(background_save_path, 'w') as f:
            f.write(self.get_background_info())
        return separate_id_list

    def make(self, dataset_output_dir, id, mode, window_size, stride, convertor_class, image_config:dict={}, drop_last:bool=True):
        # check data_path
        if id == 'data':
            data_path = os.path.join(self.dataset_info['path'], f"{mode}.npy")
        else:
            data_path = os.path.join(self.dataset_info['path'], f"{id}_{mode}.npy")
        
        # 1. sampling & normalization
        data = np.load(data_path)
        data = self.sampling(data)
        if self.normalization_enable:
            data = self.normalize(data)

        # 2. get the save path of the convertor & init the convertor
        convertor_save_path = os.path.join(dataset_output_dir, id, mode, convertor_class.output_type)
        self.ensure_dir(convertor_save_path)
        convertor = convertor_class(save_path=convertor_save_path)
        if image_config != {}:
            convertor.image_config(**image_config)

        # 3. convert & save
        # data .npy format: [num_stride, window_size, data_channels]
        window_data_save_path = os.path.join(dataset_output_dir, id, mode, 'data.npy')
        window_data_array = []
        num_stride = (len(data)-window_size) // stride + 1
        data_channels = 1 if len(data.shape) == 1 else data.shape[1]
        # data
        for i in range(num_stride):
            start = i * stride
            window_data = data[start:start+window_size]
            window_data_array.append(window_data)
            # convert & save
            for ch in range(data_channels):
                convertor.convert_and_save(window_data[:,ch], f'{i}-{ch}')
        if not drop_last:
            start = num_stride * stride
            window_data = data[start:]
            padded_window_data = padding(window_data, window_size)
            window_data_array.append(padded_window_data)
            for ch in range(data_channels):
                convertor.convert_and_save(window_data[:,ch], f'{num_stride}-{ch}')

        window_data_array = np.array(window_data_array)
        np.save(window_data_save_path, window_data_array)
        # label
        window_label_save_path = os.path.join(dataset_output_dir, id, mode, 'labels.npy')
        if mode == 'test':
            if id == 'data':
                labels_path = os.path.join(self.dataset_info['path'], f"labels.npy")
            else:
                labels_path = os.path.join(self.dataset_info['path'], f"{id}_labels.npy")
            # sampling 
            labels = np.load(labels_path)
            labels = self.sampling(labels)
            # label .npy format: [num_stride, window_size, label_channels]
            window_label_array = []
            label_channels = 1 if len(labels.shape) == 1 else labels.shape[1]
            for i in range(num_stride):
                start = i * stride
                window_label = labels[start:start+window_size]
                window_label_array.append(window_label)
            if not drop_last:
                start = num_stride * stride
                window_label = labels[start:]
                window_label = padding(window_label, window_size)
                window_label_array.append(window_label)
            window_label_array = np.array(window_label_array)
            np.save(window_label_save_path, window_label_array)
        # background
        background_save_path = os.path.join(dataset_output_dir, 'background.txt')
        with open(background_save_path, 'w') as f:
            f.write(self.get_background_info())
    '''
    output directory: "output_dir/dataset_name/id/name/convertor_type"
    '''
    def convert_data(self, output_dir:str, mode:str, window_size:int, stride:int, convertor_class:ConvertorBase, 
                     image_config:dict={}, drop_last:bool=True, data_id_list:list=[]):
        data_id_list = [] if data_id_list == [''] else data_id_list
        dataset_output_dir = os.path.join(output_dir, self.dataset_name)
        self.ensure_dir(dataset_output_dir)
        if self.dataset_info['type'] == 'centralized' or self.dataset_info['type'] == 'distributed':
            id_list = self.dataset_info['file_list'].keys() if data_id_list == [] else data_id_list
        elif self.dataset_info['type'] == 'meta_data':
            id_list = self.dataset_info['file_list'].keys() if data_id_list == [] else data_id_list
        else:
            raise ValueError(f"Invalid dataset type: {self.dataset_info['type']}")
        structure = {}
        for id in id_list:
            id = str(id)
            # print(f'id: {id}')
            # self.make(dataset_output_dir, id, mode, window_size, stride, convertor_class, image_config=image_config, drop_last=drop_last)
            id_idx_list = self.separate_label_make(dataset_output_dir, id, mode, window_size, stride, convertor_class, image_config=image_config, drop_last=drop_last)
            structure[id] = id_idx_list
        with open(os.path.join(dataset_output_dir, f"{mode}_structure.yaml"), 'w') as f:
            yaml.dump(structure, f)
        

'''
Proccessed Data Loader
'''
class ProcessedDataset:
    def __init__(self, dataset_path:str, mode:str='train'):
        self.dataset_path = dataset_path
        # print(f"Dataset Path: {dataset_path}");exit()
        self.id_list = list(filter(lambda x: os.path.isdir(os.path.join(dataset_path, x)), os.listdir(dataset_path)))
        self.id_list.sort()
        self.background = open(os.path.join(dataset_path, 'background.txt'), 'r').read()
        self.mode = mode
        self.count_data_num()

    def get_id_list(self):
        return self.id_list

    def get_instances(self, balanced:bool=True, ratio:float=0.5,refined:bool=False):
        import yaml
        import random
        structure = yaml.safe_load(open(os.path.join(self.dataset_path, f"{self.mode}_structure.yaml"), 'r'))
        pos = []
        neg = []
        if refined:
            if balanced:
                for id in structure.keys():
                    pos_len = len(structure[id]['abnormal'])
                    neg_len = len(structure[id]['normal'])
                    if neg_len < pos_len:
                        neg += structure[id]['normal']
                        pos += random.sample(structure[id]['abnormal'], neg_len)
                    else:
                        pos += structure[id]['abnormal']
                        neg += random.sample(structure[id]['normal'], pos_len)
                pos = random.sample(pos, int(len(pos)*ratio))
                neg = random.sample(neg, int(len(neg)*ratio))
            else:
                for id in structure.keys():
                    pos += random.sample(structure[id]['abnormal'], int(len(structure[id]['abnormal'])*ratio))
                    neg += random.sample(structure[id]['normal'], int(len(structure[id]['normal'])*ratio))
        else:
            if balanced:
                for id in structure.keys():
                    pos_len = len(structure[id]['abnormal'])
                    neg_len = len(structure[id]['normal'])
                    if neg_len < pos_len:
                        neg += structure[id]['normal']
                        pos += random.sample(structure[id]['abnormal'], neg_len)
                    else:
                        pos += structure[id]['abnormal']
                        neg += random.sample(structure[id]['normal'], pos_len)
            else:
                pos = 0
                neg = 0
                for id in structure.keys():
                    pos += len(structure[id]['abnormal'])
                    neg += len(structure[id]['normal'])
                print(f"Positive: {pos}, Negative: {neg}")
        return pos, neg


    def get_background_info(self):
        if self.background == '':
            self.background = 'No background information'
        return self.background
    
    def count_data_num(self):
        self.data_id_info = {}
        self.total_data_num = 0
        for data_id in self.id_list:
            item = {
                'num_stride': 0,
                'window_size': 0,
                'data_channels': 0,
                'label_channels': 0,
            }
            label_path = os.path.join(self.dataset_path, data_id, 'test', 'labels.npy')
            data_path = os.path.join(self.dataset_path, data_id, 'test', 'data.npy')
            labels = np.load(label_path)
            data = np.load(data_path)
            item['num_stride'] = data.shape[0]
            item['window_size'] = data.shape[1]
            item['data_channels'] = data.shape[2]
            item['label_channels'] = labels.shape[2]
            self.total_data_num += int(item['num_stride']*item['data_channels'])
            self.data_id_info[data_id] = item

    def get_total_data_num(self):
        return self.total_data_num
    
    def get_data_id_info(self, data_id):
        try:
            info = self.data_id_info[data_id]
        except:
            info = self.data_id_info[f'{data_id}']
        return info
    
    # TODO: 后续可以优化读取的次数，提升运行速度
    def get_data(self, data_id, num_stride, ch):
        data_path = os.path.join(self.dataset_path, data_id, self.mode, 'data.npy')
        data = np.load(data_path)
        return remove_padding(data[num_stride, :, ch])
    
    def get_image(self, data_id, num_stride, ch):
        image_path = os.path.join(self.dataset_path, data_id, self.mode, 'image', f'{num_stride}-{ch}.png')
        return image_path
    
    def get_text(self, data_id, num_stride, ch):
        text_path = os.path.join(self.dataset_path, data_id, self.mode, 'text', f'{num_stride}-{ch}.txt')
        return text_path

    def get_label(self, data_id, num_stride, ch):
        label_channels = self.data_id_info[data_id]['label_channels']
        label_ch = 0 if label_channels == 1 else ch
        label_path = os.path.join(self.dataset_path, data_id, self.mode, 'labels.npy')
        labels = np.load(label_path)
        return remove_padding(labels[num_stride, :, label_ch])
'''
Dataset loader for Evaluation
'''
DEFAULT_LOG_ROOT = os.path.join(os.path.dirname(CURRENT_PATH), 'log')
# DEFAULT_PROCESSED_DATA_ROOT = os.path.join(os.path.dirname(CURRENT_PATH), 'output')
class EvalDataLoader:
    def __init__(self, dataset_name:str, processed_data_root:str, log_root:str=DEFAULT_LOG_ROOT):
        self.dataset_info = yaml.safe_load(open(DEFAULT_YAML_PATH, 'r'))[dataset_name]
        self.dataset_name = dataset_name
        self.log_root = log_root
        self.log_file_path = os.path.join(log_root, f"{dataset_name}_log.yaml")
        self.dataset_path = os.path.join(processed_data_root, dataset_name)
        self.processed_dataset = ProcessedDataset(self.dataset_path, mode='test')
        self.eval_image_path = os.path.join(log_root, f'{dataset_name}_image')
        if not os.path.exists(self.eval_image_path):
            os.makedirs(self.eval_image_path)
        # load 
        self.output_log = yaml.safe_load(open(self.log_file_path, 'r'))
        self.plot_default_config = {
            'width': 1024,
            'height': 320,
            'dpi': 100,
            'x_ticks': 100,
        }

    def label_to_list(self, info:str):
        if info == '[]':
            return []
        else:
            return list(map(int, info.strip('[]').split(',')))
        
    def abnormal_index_to_range(self, info:str):
        # (start, end)/confidence/abnormal_type
        pattern_range = r'\(\d+,\s\d+\)/\d/[a-z]+'
        pattern_single = r'\(\d+\)/\d/[a-z]+'
        # pattern_range = r'\(\d+,\s\d+\)/\d'
        # pattern_single = r'\(\d+\)/\d'
        if info == '[]':
            return [[]]
        else:
            abnormal_ranges = re.findall(pattern_range, info)
            range_list = []
            type_list = []
            for range_tuple_confidence in abnormal_ranges:
                range_tuple, confidence, abnormal_type = range_tuple_confidence.split('/')
                range_tuple = range_tuple.strip('()')
                start, end = map(int, range_tuple.split(','))
                confidence = int(confidence)
                type_list.append(abnormal_type)
                range_list.append((start, end, confidence))
            abnomral_singles = re.findall(pattern_single, info)
            for single_point_confidence in abnomral_singles:
                single_point, confidence, abnormal_type = single_point_confidence.split('/')
                single_point = single_point.strip('()')
                confidence = int(confidence)
                single_point = int(single_point)
                type_list.append(abnormal_type)
                range_list.append((single_point,confidence))
            return range_list
        
    def map_pred_window_index_to_global_index(self, window_index, offset:int=0):
        global_index_set = set()
        for start_end_confidence in window_index:
            if isinstance(start_end_confidence, tuple) or isinstance(start_end_confidence, list):
                # (A, B, C) or (A, C) or [A, B, C] or [A, C]
                if len(start_end_confidence) == 0:
                    continue
                elif len(start_end_confidence) == 2:
                    point, confidence = start_end_confidence
                    if confidence <= 2:
                        continue
                    # single point
                    global_index_set.add(point+offset)
                elif len(start_end_confidence) == 3:
                    # a range of points
                    start, end, confidence = start_end_confidence
                    if confidence <= 2:
                        continue
                    for i in range(start, end+1):
                        global_index_set.add(i+offset)
                else:
                    raise ValueError(f"Invalid abnormal index format: {start_end_confidence}")
            else:
                global_index_set.add(start_end_confidence+offset)
        return global_index_set
    
    def map_label_window_index_to_global_index(self, window_index, offset:int=0):
        global_index_set = set()
        for start_end in window_index:
            if isinstance(start_end, tuple) or isinstance(start_end, list):
                # (A, B) or (A,) or [A] or [A, B]
                if len(start_end) == 0:
                    continue
                elif len(start_end) == 1:
                    # single point
                    global_index_set.add(start_end[0]+offset)
                elif len(start_end) == 2:
                    # a range of points
                    start, end = start_end
                    for i in range(start, end+1):
                        global_index_set.add(i+offset)
                else:
                    raise ValueError(f"Invalid abnormal index format: {start_end}")
            else:
                global_index_set.add(start_end+offset)
        global_index_set = list(global_index_set)
        global_index_set.sort()
        return global_index_set
    
    def map_pred_window_index_to_global_index_with_confidence(self, window_index, offset:int=0):
        global_index_set = {confidence: set() for confidence in range(1, 5)}
        for start_end_confidence in window_index:
            if isinstance(start_end_confidence, tuple) or isinstance(start_end_confidence, list):
                # (A, B, C) or (A, C) or [A, B, C] or [A, C]
                if len(start_end_confidence) == 0:
                    continue
                elif len(start_end_confidence) == 2:
                    point, confidence = start_end_confidence
                    # single point
                    if confidence not in global_index_set:
                        continue
                    global_index_set[confidence].add(point+offset)
                elif len(start_end_confidence) == 3:
                    # a range of points
                    start, end, confidence = start_end_confidence
                    for i in range(start, end+1):
                        if confidence not in global_index_set:
                            continue
                        global_index_set[confidence].add(i+offset)
                else:
                    raise ValueError(f"Invalid abnormal index format: {start_end_confidence}")
            else:
                # global_index_set[confidence].add(start_end_confidence+offset)
                raise ValueError(f"Invalid abnormal index format: {start_end_confidence}")
        for confidence in global_index_set:
            global_index_set[confidence] = list(global_index_set[confidence])
            global_index_set[confidence].sort()
        return global_index_set

    def set_plot_config(self, width:int, height:int, dpi:int, x_ticks:int, aux_enable:bool=False):
        plt.rcParams.update({'font.size': 8})
        self.plot_default_config['width'] = width
        self.plot_default_config['height'] = height
        self.plot_default_config['dpi'] = dpi
        self.plot_default_config['x_ticks'] = x_ticks

    def get_fill_ranges(self, points, continue_thre=1):
        # print(points)
        if points == []:
            return []
        start_idx = points[0]
        fill_range_list = []
        for i in range(1, len(points)):
            if points[i] - points[i-1] > continue_thre:
                fill_range_list.append((start_idx, points[i-1]+1))
                start_idx = points[i]
        fill_range_list.append((start_idx, points[-1]+1))
        # print(fill_range_list)
        return fill_range_list

    def plot_figure(self, data, label, pred_points, image_name:str):
        figsize = (self.plot_default_config['width']/self.plot_default_config['dpi'], self.plot_default_config['height']/self.plot_default_config['dpi'])
        fig, ax = plt.subplots(figsize=figsize, dpi=self.plot_default_config['dpi'])
        ax.plot(data, label='data')

        alpha = 0.2
        # for point in pred:
        #     ax.fill_between([point, point+0.6], np.min(data), np.max(data), color='green', alpha=alpha, label='pred' if point == pred[0] else '')
        
        # label_points = np.where(label == 1)[0]
        # for point in label_points:
        #     ax.fill_between([point, point+0.6], np.min(data), np.max(data), color='orange', alpha=alpha, label='label' if point == label_points[0] else '')
        pred_ranges = self.get_fill_ranges(pred_points)
        for start, end in pred_ranges:
            ax.fill_between(range(start, end), np.min(data), np.max(data), color='green', alpha=alpha, label='pred' if start == pred_ranges[0][0] else '')
        label_points = np.where(label == 1)[0].tolist()
        label_ranges = self.get_fill_ranges(label_points)
        for start, end in label_ranges:
            ax.fill_between(range(start, end), np.min(data), np.max(data), color='red', alpha=alpha, label='label' if start == label_ranges[0][0] else '')

        ax.legend()
        ax.set_xticks(range(0, len(data)+1, self.plot_default_config['x_ticks']))
        ax.set_xlim(0, len(data)+1)
        plt.xticks(rotation=90)
        ax.set_title(image_name)
        fig.tight_layout(w_pad=0.1, h_pad=0)
        plt.savefig(os.path.join(self.eval_image_path, f"{image_name}.png"))
        plt.close()
    
    def plot_figure_with_confidence(self, data, label, pred_points:dict, image_name:str):
        figsize = (self.plot_default_config['width']/self.plot_default_config['dpi'], self.plot_default_config['height']/self.plot_default_config['dpi'])
        fig, ax = plt.subplots(figsize=figsize, dpi=self.plot_default_config['dpi'])
        ax.plot(data, label='data')

        alpha = 0.2
        # for point in pred:
        #     ax.fill_between([point, point+0.6], np.min(data), np.max(data), color='green', alpha=alpha, label='pred' if point == pred[0] else '')
        
        # label_points = np.where(label == 1)[0]
        # for point in label_points:
        #     ax.fill_between([point, point+0.6], np.min(data), np.max(data), color='orange', alpha=alpha, label='label' if point == label_points[0] else '')
        color_map = {1: 'gray', 2: 'blue', 3: 'yellow', 4: 'green'}
        for confidence in pred_points:
            pred_ranges = self.get_fill_ranges(pred_points[confidence])
            for start, end in pred_ranges:
                ax.fill_between(range(start, end), np.min(data), np.max(data), color=color_map[confidence], alpha=alpha, label=f'pred(confidence={confidence})' if start == pred_ranges[0][0] else '')
        label_points = np.where(label == 1)[0].tolist()
        label_ranges = self.get_fill_ranges(label_points)
        for start, end in label_ranges:
            ax.fill_between(range(start, end), np.min(data), np.max(data), color='red', alpha=alpha, label='label' if start == label_ranges[0][0] else '')

        ax.legend()
        ax.set_xticks(range(0, len(data)+1, self.plot_default_config['x_ticks']))
        ax.set_xlim(0, len(data)+1)
        plt.xticks(rotation=90)
        ax.set_title(image_name)
        fig.tight_layout(w_pad=0.1, h_pad=0)
        plt.savefig(os.path.join(self.eval_image_path, f"{image_name}.png"))
        plt.close()

    def adjust_anomaly_detection_results(self, results, labels):
        """
        Adjust anomaly detection results based on the ground-truth labels.
        
        Args:
            results (np.ndarray): The anomaly detection results (0 or 1).
            labels (np.ndarray): The ground-truth labels (0 or 1).
        
        Returns:
            np.ndarray: The adjusted anomaly detection results.
        """
        adjusted_results = results.copy()
        in_anomaly = False
        start_idx = 0
        
        for i in range(len(labels)):
            if labels[i] == 1 and not in_anomaly:
                in_anomaly = True
                start_idx = i
            elif labels[i] == 0 and in_anomaly:
                in_anomaly = False
                if np.any(results[start_idx:i] == 1):
                    adjusted_results[start_idx:i] = 1
        
        # Handle the case where the last segment is an anomaly
        if in_anomaly and np.any(results[start_idx:] == 1):
            adjusted_results[start_idx:] = 1
        return adjusted_results
    
    def eval(self, window_size, stride, vote_thres:int, point_adjust_enable:bool=False, plot_enable:bool=False, channel_shared:bool=False):
        eval_logger = {}
        for data_id in self.output_log:
            data_id_info = self.processed_dataset.get_data_id_info(data_id)
            data_channels = data_id_info['data_channels']
            num_stride = data_id_info['num_stride']
            data_shape = self.dataset_info['file_list'][data_id]['test']
            # [globa_index, channel]
            ch_global_pred_array = np.zeros(data_shape)
            ch_global_label_array = np.zeros(data_shape)
            if data_channels == 1:
                ch_global_pred_array = ch_global_pred_array.reshape(-1, 1)
                ch_global_label_array = ch_global_label_array.reshape(-1, 1)
            for ch in range(data_channels):
                for stride_idx in range(num_stride):
                    if stride_idx not in self.output_log[data_id]:
                        continue
                    if self.output_log[data_id][stride_idx]=={}:
                        continue
                    item = self.output_log[data_id][stride_idx][ch]
                    labels = self.label_to_list(item['labels'])
                    if 'abnormal_index' not in item:
                        item['abnormal_index'] = '[]'
                    item['abnormal_index'] = str(item['abnormal_index'])
                    abnormal_index = self.abnormal_index_to_range(item['abnormal_index'])
                    abnormal_description = item['abnormal_description']
                    image_path = item['image']
                    # confidence = int(item['confidence'])
                    # if confidence <= 3:
                    #     abnormal_index = []
                    # map to global index
                    offset = stride_idx * stride
                    abnormal_point_set = self.map_pred_window_index_to_global_index(abnormal_index, offset)
                    label_point_set = self.map_label_window_index_to_global_index(labels, offset)
                    # mark point    
                    for label_point in label_point_set:
                        if label_point < ch_global_label_array.shape[0]:    # 
                            ch_global_label_array[label_point, ch] = 1
                    for abnormal_point in abnormal_point_set:
                        if abnormal_point < ch_global_pred_array.shape[0]:
                            ch_global_pred_array[abnormal_point, ch] += 1
                    # plot
                    if plot_enable:
                        plot_data = self.processed_dataset.get_data(data_id, stride_idx, ch)
                        plot_label = self.processed_dataset.get_label(data_id, stride_idx, ch)
                        plot_pred = self.map_pred_window_index_to_global_index_with_confidence(abnormal_index, 0)
                        # print(plot_pred)
                        self.plot_figure_with_confidence(plot_data, plot_label, plot_pred, f"{data_id}_{stride_idx}_{ch}")
                
                # vote in channel
                ch_global_pred_array[:, ch] = (ch_global_pred_array[:, ch] >= vote_thres).astype(int)
                # adjust anomaly detection results
                if point_adjust_enable:
                    ch_global_pred_array[:, ch] = self.adjust_anomaly_detection_results(ch_global_pred_array[:, ch], ch_global_label_array[:, ch])
                # plot
            # count TP, FP, TN, FN
            if channel_shared:
                global_pred_array = np.sum(ch_global_pred_array, axis=1)
                global_label_array = np.sum(ch_global_label_array, axis=1)
                # print(global_pred_array.shape)
                global_pred_array = (global_pred_array >= 1).astype(int)
                global_label_array = (global_label_array >= 1).astype(int)
                eval_logger[data_id] = {
                    'TP': np.sum((global_pred_array == 1) & (global_label_array == 1)),
                    'FP': np.sum((global_pred_array == 1) & (global_label_array == 0)),
                    'TN': np.sum((global_pred_array == 0) & (global_label_array == 0)),
                    'FN': np.sum((global_pred_array == 0) & (global_label_array == 1)),
                }
            else:
                global_pred_array = ch_global_pred_array
                global_label_array = ch_global_label_array
                # print(global_pred_array.shape)
                eval_logger[data_id] = {
                    'TP': 0,
                    'FP': 0,
                    'TN': 0,
                    'FN': 0,
                }
                for ch in range(data_channels):
                    global_pred_array[:, ch] = (global_pred_array[:, ch] >= 1).astype(int)
                    global_label_array[:, ch] = (global_label_array[:, ch] >= 1).astype(int)
                    eval_logger[data_id]['TP'] += np.sum((global_pred_array[:, ch] == 1) & (global_label_array[:, ch] == 1))
                    eval_logger[data_id]['FP'] += np.sum((global_pred_array[:, ch] == 1) & (global_label_array[:, ch] == 0))
                    eval_logger[data_id]['TN'] += np.sum((global_pred_array[:, ch] == 0) & (global_label_array[:, ch] == 0))
                    eval_logger[data_id]['FN'] += np.sum((global_pred_array[:, ch] == 0) & (global_label_array[:, ch] == 1))
                # mean
                eval_logger[data_id]['TP'] /= data_channels
                eval_logger[data_id]['FP'] /= data_channels
                eval_logger[data_id]['TN'] /= data_channels
                eval_logger[data_id]['FN'] /= data_channels
                
        # all metrics
        TP = sum([eval_logger[data_id]['TP'] for data_id in eval_logger])
        FP = sum([eval_logger[data_id]['FP'] for data_id in eval_logger])
        TN = sum([eval_logger[data_id]['TN'] for data_id in eval_logger])
        FN = sum([eval_logger[data_id]['FN'] for data_id in eval_logger])
        accuracy = (TP + TN) / (TP + FP + TN + FN)
        precision = TP / (TP + FP) if TP + FP != 0 else 0
        recall = TP / (TP + FN) if TP + FN != 0 else 0
        F1_score = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
        print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
        print(f"Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1_score: {F1_score:.3f}")

'''
New version of evaluator
'''
class Evaluator:
    def __init__(self, dataset_name:str, stride_length:int, processed_data_root:str, log_root:str=DEFAULT_LOG_ROOT, processed_path_name:str=''):
        self.dataset_name = dataset_name
        self.stride_length = stride_length
        self.processed_data_root = processed_data_root
        self.log_root = log_root
        self.dataset_info = yaml.safe_load(open(DEFAULT_YAML_PATH, 'r'))[dataset_name]
        if processed_path_name == '':
            name = dataset_name
        else:
            name = processed_path_name
        self.processed_dataset = ProcessedDataset(os.path.join(processed_data_root, name), mode='test')
        
        log_file_path = os.path.join(log_root, f"{dataset_name}_log.yaml")
        self.load_log_file(log_file_path)

    def get_ranges_from_points(self, point_list:list, continue_thres:int=1):
        if point_list == []:
            return []
        start_idx = point_list[0]
        ranges_list = []
        for i in range(1, len(point_list)):
            if point_list[i] - point_list[i-1] > continue_thres:
                ranges_list.append((start_idx, point_list[i-1]+1))
                start_idx = point_list[i]
        ranges_list.append((start_idx, point_list[-1]+1))
        return ranges_list

    def decode_label(self, label:str):
        if label == '[]':
            return []
        else:
            label_point_list =  list(map(int, label.strip('[]').split(',')))
            return label_point_list
    
    def decode_abnormal_prediction(self, log_item:dict, channel:int, offset:int=0):
        pattern_range = r'\(\d+,\s\d+\)/\d/[a-z]+'
        pattern_single = r'\(\d+\)/\d/[a-z]+'
        abnormal_index = str(log_item['abnormal_index'])
        double_check_index = log_item.get('double_check', {'fixed_abnormal_index': '[]'})
        fixed_abnormal_index = double_check_index.get('fixed_abnormal_index', '[]')
        output_dict = {
            'prediction': [],
            'double_check': [],
        }

        # prediction of ranges
        abnormal_ranges = re.findall(pattern_range, abnormal_index)
        for item in abnormal_ranges:
            range_tuple, confidence, abnormal_type = item.split('/')
            range_tuple = range_tuple.strip('()')
            start, end = map(int, range_tuple.split(','))
            confidence = int(confidence)
            pred = {
                'channel': channel,
                'start': start+offset,
                'end': end+offset,
                'confidence': confidence,
                'type': abnormal_type,
            }
            output_dict['prediction'].append(pred)
        # prediction of single points
        abnormal_singles = re.findall(pattern_single, abnormal_index)
        for item in abnormal_singles:
            single_point, confidence, abnormal_type = item.split('/')
            single_point = single_point.strip('()')
            confidence = int(confidence)
            single_point = int(single_point)
            pred = {
                'channel': channel,
                'start': single_point+offset,
                'end': single_point+1+offset,  # range(single_pint, single_point+1) == [single_point]
                'confidence': confidence,
                'type': abnormal_type,
            }
            output_dict['prediction'].append(pred)
        # double check of ranges
        double_check_ranges = re.findall(pattern_range, fixed_abnormal_index)
        for item in double_check_ranges:
            range_tuple, confidence, abnormal_type = item.split('/')
            range_tuple = range_tuple.strip('()')
            start, end = map(int, range_tuple.split(','))
            confidence = int(confidence)
            pred = {
                'channel': channel,
                'start': start+offset,
                'end': end+offset,
                'confidence': confidence,
                'type': abnormal_type,
            }
            output_dict['double_check'].append(pred)
        # double check of single points
        double_check_singles = re.findall(pattern_single, fixed_abnormal_index)
        for item in double_check_singles:
            single_point, confidence, abnormal_type = item.split('/')
            single_point = single_point.strip('()')
            confidence = int(confidence)
            single_point = int(single_point)
            pred = {
                'channel': channel,
                'start': single_point+offset,
                'end': single_point+1+offset,  # range(single_pint, single_point+1) == [single_point]
                'confidence': confidence,
                'type': abnormal_type,
            }
            output_dict['double_check'].append(pred)
        # output:
        # prediction: [{'start': int, 'end': int, 'confidence': int, 'type': str}, ...]
        # double_check: [{'start': int, 'end': int, 'confidence': int, 'type': str}, ...]
        return output_dict

    def load_log_file(self, log_file_path:str):
        self.raw_log = yaml.safe_load(open(log_file_path, 'r'))
        self.parsed_log = {}
        for data_id in self.raw_log:
            data_id_info = self.processed_dataset.get_data_id_info(data_id)
            data_channels = data_id_info['data_channels']
            num_stride = data_id_info['num_stride']
            data_shape = self.dataset_info['file_list'][data_id]['test']
            # [globa_index, channel]
            self.parsed_log[data_id] = {}
            self.parsed_log[data_id]['raw_data'] = np.zeros(data_shape).reshape(-1, data_channels)
            self.parsed_log[data_id]['label'] = np.zeros(data_shape).reshape(-1, data_channels)
            self.parsed_log[data_id]['prediction'] = []
            self.parsed_log[data_id]['double_check'] = []
            for ch in range(data_channels):
                for stride_idx in range(num_stride):
                    offset = int(stride_idx * self.stride_length)
                    if stride_idx not in self.raw_log[data_id]:
                        continue
                    # print(data_id, stride_idx, ch, self.raw_log[data_id].keys())
                    log_item = self.raw_log[data_id][stride_idx][ch]
                    # raw
                    raw_label = self.decode_label(log_item['labels'])
                    raw_data = self.processed_dataset.get_data(data_id, stride_idx, ch)
                    # map to global 
                    for point in raw_label:
                        self.parsed_log[data_id]['label'][point+offset, ch] = 1
                    self.parsed_log[data_id]['raw_data'][offset:offset+len(raw_data), ch] = raw_data
                    # get prediction and double_check
                    parsed_dict = self.decode_abnormal_prediction(log_item, ch, offset)
                    prediction_list = parsed_dict['prediction']
                    double_check_list = parsed_dict['double_check']
                    self.parsed_log[data_id]['prediction'] += prediction_list
                    self.parsed_log[data_id]['double_check'] += double_check_list

    @staticmethod
    def point_adjustment(results, labels, thres_percentage:float=0.0):
        """
        Adjust anomaly detection results based on the ground-truth labels.
        
        Args:
            results (np.ndarray): The anomaly detection results (0 or 1).
            labels (np.ndarray): The ground-truth labels (0 or 1).
        
        Returns:
            np.ndarray: The adjusted anomaly detection results.
        """
        adjusted_results = results.copy()
        in_anomaly = False
        start_idx = 0
        
        for i in range(len(labels)):
            if labels[i] == 1 and not in_anomaly:
                in_anomaly = True
                start_idx = i
            elif labels[i] == 0 and in_anomaly:
                in_anomaly = False
                # if np.any(results[start_idx:i] == 1):
                #     adjusted_results[start_idx:i] = 1
                thres = (i - start_idx) * thres_percentage
                if np.sum(results[start_idx:i]) > thres:    # threshold
                    adjusted_results[start_idx:i] = 1
        
        # Handle the case where the last segment is an anomaly
        if in_anomaly and np.any(results[start_idx:] == 1):
            adjusted_results[start_idx:] = 1
        return adjusted_results

    @staticmethod
    def get_metrics(TP, FP, TN, FN):
        accuracy = (TP + TN) / (TP + FP + TN + FN)
        precision = TP / (TP + FP) if TP + FP != 0 else 0
        recall = TP / (TP + FN) if TP + FN != 0 else 0
        F1_score = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
        return accuracy, precision, recall, F1_score

    @staticmethod
    def get_tpr_fpr(TP, FP, TN, FN):
        TPR = TP / (TP + FN) if TP + FN != 0 else 0
        FPR = FP / (FP + TN) if FP + TN != 0 else 0
        return TPR, FPR

    def calculate_TP_FP_TN_FN(self, confidence_thres=9, thres_percentage:float=0.0, data_id_list:list=[], show_results:bool=False):
        if data_id_list == []:
            data_id_list = self.parsed_log.keys()
        count_log = {}
        
        # print(f"\nDataset: {self.dataset_name}, Confidence Threshold: {confidence_thres}")
        for data_id in data_id_list:
            count_log[data_id] = {
                "pred": {
                    "TP": 0,
                    "FP": 0,
                    "TN": 0,
                    "FN": 0,
                },
                "pred_adjust": {
                    "TP": 0,
                    "FP": 0,
                    "TN": 0,
                    "FN": 0,
                },
                "double_check": {
                    "TP": 0,
                    "FP": 0,
                    "TN": 0,
                    "FN": 0,
                },
                "double_check_adjust": {
                    "TP": 0,
                    "FP": 0,
                    "TN": 0,
                    "FN": 0,
                }
            }
            data_id_log = self.parsed_log[data_id]
            raw_data = data_id_log['raw_data']  # raw data
            raw_label = data_id_log['label']    # [0,1] label array
            pred_list = data_id_log['prediction']
            double_check_list = data_id_log['double_check']

            # vote array
            pred_vote_array = np.zeros(raw_data.shape)
            # confidence array
            pred_confidence_array = np.zeros(raw_data.shape)
            # pred
            for pred_item in pred_list:
                ch = pred_item['channel']
                start = pred_item['start']
                end = pred_item['end']
                confidence = pred_item['confidence']
                type = pred_item['type']
                # vote 
                pred_vote_array[start:end, ch] += 1
                # confidence
                pred_confidence_array[start:end, ch] += confidence
            
            # vote array
            double_check_vote_array = np.zeros(raw_data.shape)
            # confidence array
            double_check_confidence_array = np.zeros(raw_data.shape)
            # double check
            for doubel_check_item in double_check_list:
                ch = doubel_check_item['channel']
                start = doubel_check_item['start']
                end = doubel_check_item['end']
                confidence = doubel_check_item['confidence']
                type = doubel_check_item['type']
                # vote 
                double_check_vote_array[start:end, ch] += 1
                # confidence
                double_check_confidence_array[start:end, ch] += confidence
        
            # threshold
            pred_vote_array = (pred_confidence_array >= confidence_thres).astype(int)
            double_check_vote_array = (double_check_confidence_array >= confidence_thres).astype(int)

            # calculate F1 score for each data_id
            for ch in range(raw_data.shape[1]):
                TP = np.sum((pred_vote_array[:, ch] == 1) & (raw_label[:, ch] == 1))
                FP = np.sum((pred_vote_array[:, ch] == 1) & (raw_label[:, ch] == 0))
                TN = np.sum((pred_vote_array[:, ch] == 0) & (raw_label[:, ch] == 0))
                FN = np.sum((pred_vote_array[:, ch] == 0) & (raw_label[:, ch] == 1))
                count_log[data_id]['pred']['TP'] += TP
                count_log[data_id]['pred']['FP'] += FP
                count_log[data_id]['pred']['TN'] += TN
                count_log[data_id]['pred']['FN'] += FN
                acc, pre, rec, f1 = self.get_metrics(TP, FP, TN, FN)
                # print(f"data_id: {data_id}, channel: {ch}")
                # print(f"\tTP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
                # print(f"\tPrediction >> Acc: {acc:.3f}, Pre: {pre:.3f}, Rec: {rec:.3f}, F1: {f1:.3f}")
                tabel_format = [
                    ["Type", "TP", "FP", "TN", "FN", "Acc", "Pre", "Rec", "F1"],
                    ["Pred", TP, FP, TN, FN, f"{acc:.3f}", f"{pre:.3f}", f"{rec:.3f}", f"{f1:.3f}"],
                ]
                # point-adjustment
                point_adjusted_pred_array = self.point_adjustment(pred_vote_array[:, ch], raw_label[:, ch], thres_percentage)
                TP = np.sum((point_adjusted_pred_array == 1) & (raw_label[:, ch] == 1))
                FP = np.sum((point_adjusted_pred_array == 1) & (raw_label[:, ch] == 0))
                TN = np.sum((point_adjusted_pred_array == 0) & (raw_label[:, ch] == 0))
                FN = np.sum((point_adjusted_pred_array == 0) & (raw_label[:, ch] == 1))
                count_log[data_id]['pred_adjust']['TP'] += TP
                count_log[data_id]['pred_adjust']['FP'] += FP
                count_log[data_id]['pred_adjust']['TN'] += TN
                count_log[data_id]['pred_adjust']['FN'] += FN
                acc, pre, rec, f1 = self.get_metrics(TP, FP, TN, FN)
                tabel_format.append(["Pred(adjust)", TP, FP, TN, FN, f"{acc:.3f}", f"{pre:.3f}", f"{rec:.3f}", f"{f1:.3f}"])

                # double check
                TP = np.sum((double_check_vote_array[:, ch] == 1) & (raw_label[:, ch] == 1))
                FP = np.sum((double_check_vote_array[:, ch] == 1) & (raw_label[:, ch] == 0))
                TN = np.sum((double_check_vote_array[:, ch] == 0) & (raw_label[:, ch] == 0))
                FN = np.sum((double_check_vote_array[:, ch] == 0) & (raw_label[:, ch] == 1))
                count_log[data_id]['double_check']['TP'] += TP
                count_log[data_id]['double_check']['FP'] += FP
                count_log[data_id]['double_check']['TN'] += TN
                count_log[data_id]['double_check']['FN'] += FN
                acc, pre, rec, f1 = self.get_metrics(TP, FP, TN, FN)
                # print(f"data_id: {data_id}, channel: {ch}")
                # print(f"\tTP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
                # print(f"\tDouble_check >> Acc: {acc:.3f}, Pre: {pre:.3f}, Rec: {rec:.3f}, F1: {f1:.3f}")
                tabel_format.append(["DCheck", TP, FP, TN, FN, f"{acc:.3f}", f"{pre:.3f}", f"{rec:.3f}", f"{f1:.3f}"])

                # point-adjustment
                point_adjusted_double_check_array = self.point_adjustment(double_check_vote_array[:, ch], raw_label[:, ch], thres_percentage)
                TP = np.sum((point_adjusted_double_check_array == 1) & (raw_label[:, ch] == 1))
                FP = np.sum((point_adjusted_double_check_array == 1) & (raw_label[:, ch] == 0))
                TN = np.sum((point_adjusted_double_check_array == 0) & (raw_label[:, ch] == 0))
                FN = np.sum((point_adjusted_double_check_array == 0) & (raw_label[:, ch] == 1))
                count_log[data_id]['double_check_adjust']['TP'] += TP
                count_log[data_id]['double_check_adjust']['FP'] += FP
                count_log[data_id]['double_check_adjust']['TN'] += TN
                count_log[data_id]['double_check_adjust']['FN'] += FN
                acc, pre, rec, f1 = self.get_metrics(TP, FP, TN, FN)
                tabel_format.append(["DCheck(adjust)", TP, FP, TN, FN, f"{acc:.3f}", f"{pre:.3f}", f"{rec:.3f}", f"{f1:.3f}"])
                if show_results:
                    print(f"\ndata_id: {data_id}, channel: {ch}")
                    print(tabulate(tabel_format, headers='firstrow', tablefmt='fancy_grid'))

        # calculate F1 score for all data_id
        TP = sum([count_log[data_id]['pred']['TP'] for data_id in count_log])
        FP = sum([count_log[data_id]['pred']['FP'] for data_id in count_log])
        TN = sum([count_log[data_id]['pred']['TN'] for data_id in count_log])
        FN = sum([count_log[data_id]['pred']['FN'] for data_id in count_log])
        acc, pre, rec, f1 = self.get_metrics(TP, FP, TN, FN)
        # print(f"All data_id: ")
        # print(f"\tPrediction >> Acc: {acc:.3f}, Pre: {pre:.3f}, Rec: {rec:.3f}, F1: {f1:.3f}")
        tabel_format = [
            ["Type", "TP", "FP", "TN", "FN", "Acc", "Pre", "Rec", "F1"],
            ["Pred", TP, FP, TN, FN, f"{acc:.3f}", f"{pre:.3f}", f"{rec:.3f}", f"{f1:.3f}"],
        ]
        Pred_item = {
            "TP": TP,
            "FP": FP,
            "TN": TN,
            "FN": FN,
        }
        # adjust
        TP = sum([count_log[data_id]['pred_adjust']['TP'] for data_id in count_log])
        FP = sum([count_log[data_id]['pred_adjust']['FP'] for data_id in count_log])
        TN = sum([count_log[data_id]['pred_adjust']['TN'] for data_id in count_log])
        FN = sum([count_log[data_id]['pred_adjust']['FN'] for data_id in count_log])
        acc, pre, rec, f1 = self.get_metrics(TP, FP, TN, FN)
        tabel_format.append(["Pred(adjust)", TP, FP, TN, FN, f"{acc:.3f}", f"{pre:.3f}", f"{rec:.3f}", f"{f1:.3f}"])
        Pred_adjust_item = {
            "TP": TP,
            "FP": FP,
            "TN": TN,
            "FN": FN,
        }
        # double_check
        TP = sum([count_log[data_id]['double_check']['TP'] for data_id in count_log])
        FP = sum([count_log[data_id]['double_check']['FP'] for data_id in count_log])
        TN = sum([count_log[data_id]['double_check']['TN'] for data_id in count_log])
        FN = sum([count_log[data_id]['double_check']['FN'] for data_id in count_log])
        acc, pre, rec, f1 = self.get_metrics(TP, FP, TN, FN)
        # print(f"All data_id: ")
        # print(f"Double_check >> Acc: {acc:.3f}, Pre: {pre:.3f}, Rec: {rec:.3f}, F1: {f1:.3f}")
        tabel_format.append(["DCheck", TP, FP, TN, FN, f"{acc:.3f}", f"{pre:.3f}", f"{rec:.3f}", f"{f1:.3f}"])
        DCheck_item = {
            "TP": TP,
            "FP": FP,
            "TN": TN,
            "FN": FN,
        }
        # adjust
        TP = sum([count_log[data_id]['double_check_adjust']['TP'] for data_id in count_log])
        FP = sum([count_log[data_id]['double_check_adjust']['FP'] for data_id in count_log])
        TN = sum([count_log[data_id]['double_check_adjust']['TN'] for data_id in count_log])
        FN = sum([count_log[data_id]['double_check_adjust']['FN'] for data_id in count_log])
        acc, pre, rec, f1 = self.get_metrics(TP, FP, TN, FN)
        tabel_format.append(["DCheck(adjust)", TP, FP, TN, FN, f"{acc:.3f}", f"{pre:.3f}", f"{rec:.3f}", f"{f1:.3f}"])
        DCheck_adjust_item = {
            "TP": TP,
            "FP": FP,
            "TN": TN,
            "FN": FN,
        }
        # output
        output_metrics = {
            "Pred": Pred_item,
            "Pred_adjust": Pred_adjust_item,
            "DCheck": DCheck_item,
            "DCheck_adjust": DCheck_adjust_item,
        }
        if show_results:
            print(f"\nAll data_id: ")
            print(tabulate(tabel_format, headers='firstrow', tablefmt='fancy_grid'))
        return output_metrics
    
    def calculate_roc_pr_auc(self,data_id_list:list=[]):
        # self.parsed_log:
        #   - data_id
        #      + raw_data: [global_index, channel]
        #      + label: [global_index, channel]
        #      + prediction: [{'channel': ch, 'start': int, 'end': int, 'confidence': int, 'type': str}, ...]
        #      + double_check: ['channel': ch, 'start': int, 'end': int, 'confidence': int, 'type': str}, ...]
        if data_id_list == []:
            data_id_list = self.parsed_log.keys()
        # auc
        TPR_FPR_map = {
            'Pred': [],
            'Pred_adjust': [],
            'DCheck': [],
            'DCheck_adjust': [],
        }
        PR_map = {
            'Pred': [],
            'Pred_adjust': [],
            'DCheck': [],
            'DCheck_adjust': [],
        }
        AUC_PR_map = {
            'Pred': 0,
            'Pred_adjust': 0,
            'DCheck': 0,
            'DCheck_adjust': 0,
        }
        AUC_ROC_map = {
            'Pred': 0,
            'Pred_adjust': 0,
            'DCheck': 0,
            'DCheck_adjust': 0,
        }
        for conf in range(0,13):
            res = self.calculate_TP_FP_TN_FN(conf, data_id_list=data_id_list)
            for key in res:
                item = res[key]
                TPR, FPR = self.get_tpr_fpr(item['TP'], item['FP'], item['TN'], item['FN'])
                TPR_FPR_map[key].append((FPR, TPR))
                acc, pre, rec, f1 = self.get_metrics(item['TP'], item['FP'], item['TN'], item['FN'])
                PR_map[key].append((rec, pre))
        # plot ROC curve
        fig, ax = plt.subplots(figsize=(8, 5))
        for key in TPR_FPR_map:
            auc_score = auc([x[0] for x in TPR_FPR_map[key]], [x[1] for x in TPR_FPR_map[key]])
            AUC_ROC_map[key] = auc_score
            ax.plot([x[0] for x in TPR_FPR_map[key]], [x[1] for x in TPR_FPR_map[key]], label=f'{key} (AUC={auc_score:.3f})', marker='x')
        ax.legend()
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(np.arange(0, 1.05, 0.1))
        ax.set_yticks(np.arange(0, 1.05, 0.1))
        ax.grid(True, linestyle='--', color='lightgray', linewidth=0.5)
        ax.set_title(f"{self.dataset_name}-ROC curve")
        fig.savefig(f"./{self.dataset_name}_ROC_curve.png", bbox_inches='tight')
        plt.close()
        # plot PR curve
        fig, ax = plt.subplots(figsize=(7, 5))
        for key in PR_map:
            recall_list = [x[0] for x in PR_map[key]]
            recall_list.append(0)
            precision_list = [x[1] for x in PR_map[key]]
            precision_list.append(1)
            auc_score = auc(recall_list, precision_list)
            AUC_PR_map[key] = auc_score
            ax.plot(recall_list, precision_list, label=f'{key} (AUC={auc_score:.3f})', marker='x')
        ax.legend()
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(np.arange(0, 1.05, 0.1))
        ax.set_yticks(np.arange(0, 1.05, 0.1))
        ax.grid(True, linestyle='--', color='lightgray', linewidth=0.5)
        ax.set_title(f"{self.dataset_name}-PR curve")
        fig.savefig(f"./{self.dataset_name}_PR_curve.png", bbox_inches='tight')
        plt.close()
        return AUC_ROC_map, AUC_PR_map


    def calculate_adjust_PR_curve_auc(self, data_id_list:list=[]):
        if data_id_list == []:
            data_id_list = self.parsed_log.keys()
        # auc
        
        # default_confidence_thres = 9
        fig, ax = plt.subplots(figsize=(13, 7)) 
        for thres_percentage in np.arange(0, 1.05, 0.2):
            PR_map = {
                # 'Pred': [],
                'Pred_adjust': [],
                # 'DCheck': [],
                # 'DCheck_adjust': [],
            }
            for confidence in range(0, 13):
                res = self.calculate_TP_FP_TN_FN(confidence, thres_percentage, data_id_list)
                for key in PR_map:
                    item = res[key]
                    acc, pre, rec, f1 = self.get_metrics(item['TP'], item['FP'], item['TN'], item['FN'])
                    PR_map[key].append((rec, pre))
            # plot PR curve
            for key in PR_map:
                recall_list = [x[0] for x in PR_map[key]]
                recall_list.append(0)
                precision_list = [x[1] for x in PR_map[key]]
                precision_list.append(1)
                auc_score = auc(recall_list, precision_list)
                ax.plot(recall_list, precision_list, label=f'{key} (thres={thres_percentage:.2f}, AUC={auc_score:.3f})', marker='x')
                # print(f"{key} >> {PR_map[key]}")
        ax.legend()
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_xlim(0, 1.01)
        ax.set_ylim(0, 1.01)
        ax.set_xticks(np.arange(0, 1.01, 0.1))
        ax.set_yticks(np.arange(0, 1.01, 0.1))
        ax.grid(True, linestyle='--', color='lightgray', linewidth=0.5)
        ax.set_title(f"{self.dataset_name}-PR curve (point-adjustment)")
        fig.savefig(f"./{self.dataset_name}_PR_curve_point_adjustment.png", bbox_inches='tight')
        plt.close()

    def calculate_f1_aucpr_aucroc(self, confidence_thres, point_adjustment_thres, data_id_list:list=[]):
        if data_id_list == []:
            data_id_list = self.parsed_log.keys()
        output_metrics = {}
        res = self.calculate_TP_FP_TN_FN(confidence_thres, point_adjustment_thres, data_id_list)
        auc_roc_map, auc_pr_map = self.calculate_roc_pr_auc(data_id_list)
        for key in res:
            item = res[key]
            acc, pre, rec, f1 = self.get_metrics(item['TP'], item['FP'], item['TN'], item['FN'])
            auc_roc = auc_roc_map[key]
            auc_pr = auc_pr_map[key]
            output_metrics[key] = {
                'Acc': acc,
                'Pre': pre,
                'Rec': rec,
                'F1': f1,
                'AUC_ROC': auc_roc,
                'AUC_PR': auc_pr,
            }
        return output_metrics
        

            

if __name__ == '__main__':
    # check_shape('MSL')
    AutoRegister()
    # dataset = RawDataset('UCR', sample_rate=1, normalization_enable=True)
    # dataset.convert_data('./output/test-1-300', 'test', 1000, 500, ImageConvertor)
