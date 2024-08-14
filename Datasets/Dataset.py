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
        self.width = 1500
        self.height = 320
        self.dpi = 100
        self.x_ticks = 100
        self.aux_enable = True
        self.line_color = 'blue'
        plt.rcParams.update({'font.size': 6})
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

    def make(self, dataset_output_dir, id, mode, window_length, stride, convertor_class, image_config:dict={}):
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
        if image_config != {}:
            convertor.image_config(**image_config)
        raw_data_save_path = os.path.join(dataset_output_dir, id, mode, 'data.npy')
        raw_data_array = []
        cnt = 0
        num_stride = (len(data)-window_length) // stride + 1
        # print(len(data), window_length, stride, num_stride);exit()
        for i in range(num_stride):
            start = i * stride
            window_data = data[start:start+window_length]
            if len(window_data.shape) == 1:
                channel = 1
            else:
                channel = window_data.shape[1]
            for ch in range(channel):
                convertor.convert_and_save(window_data[:, ch], cnt)
                raw_data_array.append(window_data[:, ch])
                # print(window_data[:, ch].shape)
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
            for i in range(num_stride):
                start = i * stride
                window_label = labels[start:start+window_length]
                if len(window_data.shape) == 1:
                    channel = 1
                else:
                    channel = window_data.shape[1]
                for ch in range(channel):
                    label_array.append(window_label[:, ch])
            label_array = np.array(label_array)
            np.save(label_save_path, label_array)
        # background
        background_save_path = os.path.join(dataset_output_dir, 'background.txt')
        with open(background_save_path, 'w') as f:
            f.write(self.get_background_info())
    '''
    output directory: "output_dir/dataset_name/id/name/convertor_type"
    '''
    def convert_data(self, output_dir:str, mode:str, window_length:int, stride:int, convertor_class:ConvertorBase, image_config:dict={}):
        dataset_output_dir = os.path.join(output_dir, self.dataset_name)
        self.ensure_dir(dataset_output_dir)
        for id in self.dataset_info['file_list']:
            self.make(dataset_output_dir, id, mode, window_length, stride, convertor_class, image_config=image_config)
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
                # print(label_path);print(labels.shape);exit()
                self.data_num += int(labels.shape[0])
                self.id_data_num[id] = int(labels.shape[0])
        return self.data_num
    
    def get_id_data_num(self):
        if not hasattr(self, 'id_data_num'):
            self.get_data_num()
        return self.id_data_num

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
        self.data_path = os.path.join(processed_data_root, dataset_name)
        self.eval_image_path = os.path.join(log_root, f'{dataset_name}_image')
        
        # load 
        self.output_log = yaml.safe_load(open(self.log_file_path, 'r'))

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

    def plot_figure(self, index, pred, image_name):
        # print(self.plot_data.shape);exit()
        data = self.plot_data[index]
        label = self.plot_label[index]
        prediction = pred.copy()

        prediction[prediction == 1] = np.max(data)
        prediction[prediction == 0] = np.min(data)
        label_points = np.where(label == 1)[0]

        fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
        ax.plot(data, label='RawData')
        ax.plot(prediction, label='Prediction', color='green')
        for point in label_points:
            ax.plot(point, data[point], 'o', color='red', label='Anomaly' if point == label_points[0] else '')
        title = image_name.split('.')[0]
        plt.title(f'{title}')
        plt.legend()
        plt.savefig(os.path.join(self.eval_image_path, image_name))
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
    
    def eval(self, window_size:int, stride:int, vote_thres:int, point_adjust_enable:bool=False, plot_enable:bool=False):
        eval_results = {}
        for data_id in self.output_log:
            data_shape = self.dataset_info['file_list'][data_id]['test']
            if len(data_shape) == 1:
                channels = 1
            else:
                channels = data_shape[1]

            global_pred_array = np.zeros(data_shape)
            global_label_array = np.zeros(data_shape)
            eval_results[data_id] = {
                'TP': 0,
                'FP': 0,
                'TN': 0,
                'FN': 0,
            }
            for index in self.output_log[data_id]:
                item = self.output_log[data_id][index]
                labels = self.label_to_list(item['labels'])
                abnormal_index = self.abnormal_index_to_range(item['abnormal_index'])
                abnormal_type = item['abnormal_type']
                abnormal_description = item['abnormal_description']
                image_path = item['image']
                # when making dataset, sliding window first and then separating channels.
                # In another word, every N channels are in the same window. (if there are N channels)
                # therefore, if there are N channels, the true window stride should be (index//N)*stride
                # the channel can be calculated by (index%N)
                num_stride = index // channels
                ch = index % channels
                offset = num_stride * stride

                # map to global index
                abnormal_point_set = self.map_window_index_to_global_index(abnormal_index, offset)
                label_point_set = self.map_window_index_to_global_index(labels, offset)

                # mark point
                for label_point in label_point_set:
                    global_label_array[label_point, ch] = 1
                for abnormal_point in abnormal_point_set:
                    global_pred_array[abnormal_point, ch] += 1
                
                # plot
                if plot_enable:
                    if not hasattr(self, 'plot_data') or not hasattr(self, 'plot_label'):
                        plot_data_path = os.path.join(self.data_path, data_id, 'test', 'data.npy')
                        plot_label_path = os.path.join(self.data_path, data_id, 'test', 'labels.npy')
                        self.plot_data = np.load(plot_data_path)
                        self.plot_label = np.load(plot_label_path)
                    # global_index = item['global_index']
                    pred = global_pred_array[offset:offset+window_size,ch]
                    self.plot_figure(index, pred, f'{data_id}-{index}-{ch}.png')
                
            # vote
            global_pred_array[global_pred_array >= vote_thres] = 1
            global_pred_array[global_pred_array < vote_thres] = 0

            # adjust anomaly detection results
            if point_adjust_enable:
                for ch in range(channels):
                    global_pred_array[:, ch] = self.adjust_anomaly_detection_results(global_pred_array[:, ch], global_label_array[:, ch])
                # global_pred_array = self.adjust_anomaly_detection_results(global_pred_array, global_label_array)

            # calculate TP, FP, TN, FN
            for i in range(data_shape[0]):
                for ch in range(channels):
                    if global_pred_array[i, ch] == 1 and global_label_array[i, ch] == 1:
                        eval_results[data_id]['TP'] += 1
                    elif global_pred_array[i, ch] == 1 and global_label_array[i, ch] == 0:
                        eval_results[data_id]['FP'] += 1
                    elif global_pred_array[i, ch] == 0 and global_label_array[i, ch] == 0:
                        eval_results[data_id]['TN'] += 1
                    elif global_pred_array[i, ch] == 0 and global_label_array[i, ch] == 1:
                        eval_results[data_id]['FN'] += 1
        # all metrics
        TP = sum([eval_results[data_id]['TP'] for data_id in eval_results])
        FP = sum([eval_results[data_id]['FP'] for data_id in eval_results])
        TN = sum([eval_results[data_id]['TN'] for data_id in eval_results])
        FN = sum([eval_results[data_id]['FN'] for data_id in eval_results])
        accuracy = (TP + TN) / (TP + FP + TN + FN)
        precision = TP / (TP + FP) if TP + FP != 0 else 0
        recall = TP / (TP + FN) if TP + FN != 0 else 0
        F1_score = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
        print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
        print(f"Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1_score: {F1_score:.3f}")
        return eval_results

if __name__ == '__main__':
    # check_shape('MSL')
    AutoRegister()
    # dataset = RawDataset('UCR', sample_rate=1, normalization_enable=True)
    # dataset.convert_data('../output/test-1-300', 'test', 1000, 500, ImageConvertor)