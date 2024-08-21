import yaml
import os
import re
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

    def image_config(self, width:int, height:int, dpi:int, x_ticks:int, aux_enable:bool):
        self.width = width
        self.height = height
        self.dpi = dpi
        self.x_ticks = x_ticks
        self.aux_enable = aux_enable
        # convert to inches
        self.figsize = (self.width/self.dpi, self.height/self.dpi)

    def convert_and_save(self, data, name:int):
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
        # TODO: how to format the time series data to text ????
        raise NotImplementedError
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
        dataset_output_dir = os.path.join(output_dir, self.dataset_name)
        self.ensure_dir(dataset_output_dir)
        id_list = self.dataset_info['file_list'].keys() if data_id_list == [] else data_id_list
        for id in id_list:
            self.make(dataset_output_dir, id, mode, window_size, stride, convertor_class, image_config=image_config, drop_last=drop_last)
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
        self.count_data_num()

    def get_id_list(self):
        return self.id_list
    
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
        return self.data_id_info[data_id]
    
    # TODO: 后续可以优化读取的次数，提升运行速度
    def get_data(self, data_id, num_stride, ch):
        data_path = os.path.join(self.dataset_path, data_id, self.mode, 'data.npy')
        data = np.load(data_path)
        return remove_padding(data[num_stride, :, ch])
    
    def get_image(self, data_id, num_stride, ch):
        image_path = os.path.join(self.dataset_path, data_id, self.mode, 'image', f'{num_stride}-{ch}.png')
        return image_path

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
        pattern_range = r'\(\d+,\s\d+\)'
        pattern_single = r'\(\d+\)'
        if info == '[]':
            return [[]]
        else:
            abnormal_ranges = re.findall(pattern_range, info)
            range_list = []
            for range_tuple in abnormal_ranges:
                range_tuple = range_tuple.strip('()')
                start, end = map(int, range_tuple.split(','))
                range_list.append((start, end))
            abnomral_singles = re.findall(pattern_single, info)
            for single_point in abnomral_singles:
                single_point = single_point.strip('()')
                range_list.append((int(single_point),))
            return range_list
        
    def map_window_index_to_global_index(self, window_index, offset:int=0):
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
            for ch in range(data_channels):
                for stride_idx in range(num_stride):
                    item = self.output_log[data_id][stride_idx][ch]
                    labels = self.label_to_list(item['labels'])
                    abnormal_index = self.abnormal_index_to_range(item['abnormal_index'])
                    abnormal_type = item['abnormal_type']
                    abnormal_description = item['abnormal_description']
                    image_path = item['image']
                    confidence = int(item['confidence'])
                    if confidence <= 90:
                        abnormal_index = []
                    # map to global index
                    offset = stride_idx * stride
                    abnormal_point_set = self.map_window_index_to_global_index(abnormal_index, offset)
                    label_point_set = self.map_window_index_to_global_index(labels, offset)
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
                        plot_pred = [point-offset for point in abnormal_point_set]
                        self.plot_figure(plot_data, plot_label, plot_pred, f"{data_id}_{stride_idx}_{ch}")
                
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

if __name__ == '__main__':
    # check_shape('MSL')
    # AutoRegister()
    dataset = RawDataset('UCR', sample_rate=1, normalization_enable=True)
    dataset.convert_data('../output/test-1-300', 'test', 1000, 500, ImageConvertor)