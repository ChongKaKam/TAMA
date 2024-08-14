import yaml
import os
import numpy as np
import matplotlib.pyplot as plt

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
        self.ensure_dir()
    def ensure_dir(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
    def convert_and_save(self, data):
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
        self.width = 2400
        self.height = 320
        self.dpi = 100
        self.x_ticks = 5
        self.aux_enable = True
        self.line_color = 'blue'
        plt.rcParams.update({'font.size': 6})
        # convert to inches
        self.figsize = (self.width/self.dpi, self.height/self.dpi)

    def convert_and_save(self, data, idx:int):
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
        ax.set_xticks(range(0, len(data_checked)+1, self.x_ticks))
        ax.set_xlim(0, len(data_checked)+1)
        fig.tight_layout(pad=0)
        fig.savefig(os.path.join(self.save_path, f"{idx}.png"), bbox_inches='tight')
        plt.close()

    def load(self, idx:int):
        image_path = os.path.join(self.save_path, f"{idx}.png")
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
    def convert_and_save(self, data, idx:int):
        with open(os.path.join(self.save_path, f"{idx}.txt"), 'w') as f:
            formatted_data = self.format(data)
            f.write(formatted_data)
    def load(self, idx:int):
        text_path = os.path.join(self.save_path, f"{idx}.txt")
        res = {
            'type': 'text',
            'data': text_path
        }
        return res
    def format(self, data):
        # TODO: how to format the time series data to text ????
        raise NotImplementedError

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

    def make(self, dataset_output_dir, id, mode, window_length, stride, convertor_class):
        if id == 'data':
            data_path = os.path.join(self.dataset_info['path'], f"{mode}.npy")
        else:
            data_path = os.path.join(self.dataset_info['path'], f"{id}_{mode}.npy")
        data = np.load(data_path)
        data = self.sampling(data)
        if self.normalization_enable:
            data = self.normalize(data)
        
        convertor_save_path = os.path.join(dataset_output_dir, id, mode, convertor_class.output_type)
        self.ensure_dir(convertor_save_path)
        convertor = convertor_class(save_path=convertor_save_path)
        raw_data_save_path = os.path.join(dataset_output_dir, id, mode, 'data.npy')
        raw_data_array = []
        cnt = 0
        for i in range(0, len(data)-window_length, stride):
            window_data = data[i:i+window_length]
            if len(window_data.shape) == 1:
                num_dim = 1
            else:
                num_dim = window_data.shape[1]
            for j in range(num_dim):
                convertor.convert_and_save(window_data[:, j], cnt)
                raw_data_array.append(window_data[:, j])
                cnt += 1
        raw_data_array = np.array(raw_data_array)
        np.save(raw_data_save_path, raw_data_array)

        label_save_path = os.path.join(dataset_output_dir, id, mode, 'labels.npy')
        if mode == 'test':
            if id == 'data':
                labels_path = os.path.join(self.dataset_info['path'], f"labels.npy")
            else:
                labels_path = os.path.join(self.dataset_info['path'], f"{id}_labels.npy")
            labels = np.load(labels_path)
            labels = self.sampling(labels)
            label_array = []
            for i in range(0, len(data)-window_length, stride):
                window_label = labels[i:i+window_length]
                if len(window_data.shape) == 1:
                    num_dim = 1
                else:
                    num_dim = window_data.shape[1]
                for j in range(num_dim):
                    label_array.append(window_label[:, j])
            label_array = np.array(label_array)
            np.save(label_save_path, label_array)
        # background
        background_save_path = os.path.join(dataset_output_dir, 'background.txt')
        with open(background_save_path, 'w') as f:
            f.write(self.get_background_info())
    '''
    output directory: "output_dir/dataset_name/id/name/convertor_type"
    '''
    def convert_data(self, output_dir:str, mode:str, window_length:int, stride:int, convertor_class:ConvertorBase):
        dataset_output_dir = os.path.join(output_dir, self.dataset_name)
        self.ensure_dir(dataset_output_dir)
        for id in self.dataset_info['file_list']:
            self.make(dataset_output_dir, id, mode, window_length, stride, convertor_class)
'''
Proccessed Data Loader
'''
class ProcessedDataset:
    def __init__(self, dataset_path:str, mode:str='train'):
        self.dataset_path = dataset_path
        self.id_list = list(filter(lambda x: os.path.isdir(os.path.join(dataset_path, x)), os.listdir(dataset_path)))
        self.id_list.sort()
        self.background = open(os.path.join(dataset_path, 'background.txt'), 'r').read()
        self.mode = mode
        self.get_data_num()

    def get_id_list(self):
        return self.id_list
    
    def get_background_info(self):
        if self.background == '':
            self.background = 'No background information'
        return self.background
    
    def get_data_num(self):
        if not hasattr(self, 'data_num'):
            self.data_num = 0
            self.id_data_num = {}
            for id in self.id_list:
                self.id_data_num[id] = 0
                label_path = os.path.join(self.dataset_path, id, 'test', 'labels.npy')
                if not os.path.exists(label_path):
                    raise FileNotFoundError(f"Labels not found in {label_path}")
                labels = np.load(label_path)
                self.data_num += int(labels.shape[0])
                self.id_data_num[id] = int(labels.shape[0])
        return self.data_num
    
    def get_data_by_id_index(self, id, index):
        data_dir = os.path.join(self.dataset_path, id, self.mode)
        image_path = os.path.join(data_dir, 'image', f"{index}.png")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image {image_path} not found")
        return image_path
    
    def get_data_by_index(self, index):
        for id in self.id_list:
            if index < self.id_data_num[id]:
                return self.get_data_by_id_index(id, index)
            index -= self.id_data_num[id]
        raise IndexError(f"Index {index} out of range")

    def get_label_by_id_index(self, id, index):
        label_path = os.path.join(self.dataset_path, id, 'test', 'labels.npy')
        labels = np.load(label_path)
        return labels[index]
    
    def get_label_by_index(self, index):
        for id in self.id_list:
            if index < self.id_data_num[id]:
                return self.get_label_by_id_index(id, index)
            index -= self.id_data_num[id]
        raise IndexError(f"Index {index} out of range")


if __name__ == '__main__':
    # check_shape('MSL')
    # AutoRegister()
    dataset = RawDataset('UCR', sample_rate=1, normalization_enable=True)
    dataset.convert_data('../output/test-1-300', 'test', 1000, 500, ImageConvertor)