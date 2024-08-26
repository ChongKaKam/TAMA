import os
import yaml
import numpy as np
import time
import json
import random
import io
import base64
from PIL import Image
import argparse
import csv

from BigModel.Base import BigModelBase
from BigModel.Azure import Chat_AzureGPT4o
from Datasets.Dataset import ProcessedDataset

'''
Args Parser
'''
def args_parse():
    parser = argparse.ArgumentParser(description='Command line interface for main.py')
    parser.add_argument('--dataset', type=str, default='NormA', help='Dataset name')
    parser.add_argument('--refined', default=False, action='store_true')
    parser.add_argument('--balanced', default=False, action='store_true')
    parser.add_argument('--ratio', type=float, default=1.0, required=False)
    parser.add_argument('--data_id_list', type=str, default='', help='Data id list')
    args = parser.parse_args()
    # dump data_id_list_str
    data_id_list_str = args.data_id_list
    data_id_list = data_id_list_str.strip('][').replace(' ', '').split(',')
    args.data_id_list = data_id_list
    return args

'''
Message Helper
'''
class MessageHelper:
    def __init__(self):
        self.message_list = []
    @staticmethod
    def image_base64_encode(image_path):
        with Image.open(image_path) as img:
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
    def set_system_message(self, message):
        self.sys_message = {'role': 'system', 'content': message}
        self.message_list.append(self.sys_message)
    def add_message(self, role:str, text:str, image_path_list:list=[]):
        new_message = {
            'role': role,
            'content': [{"type": "text", "text": text}],
        }
        for image_path in image_path_list:
            image_base64 = self.image_base64_encode(image_path)
            new_message['content'].append({
                "type": "image_url", 
                "image_url": {"url": f"data:image/png;base64,{image_base64}", "detail": "high"},
            })
        self.message_list.append(new_message)
    def add_user_message(self, text:str, image_path_list:list=[]):
        self.add_message('user', text, image_path_list)
    def add_chatbot_answer(self, text:str, image_path_list:list=[]):
        self.add_message('assistant', text, image_path_list)
    def get_message(self):
        return self.message_list
    def clean_message(self):
        self.message_list = []
        self.message_list.append(self.sys_message)
    def copy_message(self):
        new_helper = MessageHelper()
        new_helper.message_list = self.message_list.copy()
        return new_helper
'''
Normal Reference Helper
'''
class NormalReferenceHelper:
    def __init__(self, normal_reference_base:str, structure_yaml_path:str):
        self.normal_reference_base = normal_reference_base
        self.structure_yaml_path = structure_yaml_path
        self.structure_info = yaml.safe_load(open(structure_yaml_path, 'r'))
    def find_normal_reference(self, data_id, channel, num:int=3, fixed:bool=False)->list:
        if os.path.exists(os.path.join(self.normal_reference_base, data_id, 'train', 'image')):
            file_list = os.listdir(os.path.join(self.normal_reference_base, data_id, 'train', 'image'))
            normal_list = []
            for i in range(len(file_list)):
                image_name = file_list[i]
                if image_name.endswith('.png'):
                    normal_list.append(os.path.join(self.normal_reference_base, data_id, 'train', 'image', image_name))
            if num > len(normal_list):
                num = len(normal_list)
            sample_normal_list = normal_list[:num] if fixed else random.sample(normal_list, num)
            return sample_normal_list
        else:
            normal_list = self.structure_info[data_id]['normal']
            if len(normal_list) == 0:
                raise Exception(f'No normal reference found in {self.structure_yaml_path} for data_id {data_id}.')
            if num > len(normal_list):
                num = len(normal_list)
            sample_normal_list = normal_list[:num] if fixed else random.sample(normal_list, num)
            print(f'Normal reference list: {sample_normal_list}')
            for idx in range(num):
                parsed_id = sample_normal_list[idx].split('-')
                image_path = os.path.join(self.normal_reference_base, data_id, 'test', 'image', f'{parsed_id[1]}-{parsed_id[2]}.png')
                sample_normal_list[idx] = image_path
            return sample_normal_list
'''
Logger
'''
# class Logger:
#     def __init__(self, save_path:str, split_char:str='@'):
#         self.save_path = save_path
#         self.split_char = split_char
#         self.mode = 'a' # append mode
#         if os.path.exists(save_path):
#             ans = input(f'File {save_path} already exists, overwrite? (y/n)')
#             if ans.lower() != 'y':
#                 raise Exception('File already exists.')
#             self.mode = 'w' # write mode
#         self.file = open(save_path, self.mode)
#         self.writer = csv.writer(self.file)
class Logger:
    def __init__(self, save_path:str):
        self.save_path = save_path
        self.log_info = {}

    def log(self, data_id, stride, channel, info:dict):
        if data_id not in self.log_info:
            self.log_info[data_id] = {}
        if stride not in self.log_info[data_id]:
            self.log_info[data_id][stride] = {}
        self.log_info[data_id][stride][channel] = info

    def save(self):
        with open(self.save_path, 'w') as f:
            yaml.dump(self.log_info, f)
'''
Chat Controller
Features:
    - timing
    - token usage
    - retry
    - rate control
'''
class ChatController:
    def __init__(self, chatbot:BigModelBase):
        self.chatbot = chatbot
        self.max_tokens_per_minute = chatbot.tokens_per_minute
        self.used_tokens_in_minute = 0
        self.reset_minute_timing = True
        self.last_used_token = 0
        self.token_usage = {
            'completion_tokens': 0,
            'prompt_tokens': 0,
            'total_tokens': 0
        }
        self.start_time = 0
        self.sample_time_usage = []
    def start_timming(self):
        self.start_time = time.time()
    def get_used_time(self):
        return time.time() - self.start_time
    def get_used_token(self):
        return self.token_usage
    def get_last_used_token(self):
        return self.last_used_token
    def get_last_sample_time_usage(self):
        return self.sample_time_usage[-1]
    def check_rate_limit(self):
        # check if token limit reached
        used_seconds = self.get_used_time()
        sleep_time = 0
        if used_seconds > 59:
            self.reset_minute_timing = True
        elif self.used_tokens_in_minute >= self.max_tokens_per_minute:
            self.reset_minute_timing = True
            sleep_time = 60 - used_seconds
        # reset timing and token usage
        if self.reset_minute_timing:
            self.used_tokens_in_minute = 0
            self.reset_minute_timing = False
            time.sleep(sleep_time)
            self.start_timming()
    def chat_with_rate_control(self, message:list, max_retry:int=3)->dict:
        # chat with retry
        for _ in range(max_retry):
            try:
                sample_time_start = time.time()
                self.last_used_token = 0
                self.check_rate_limit()
                response = self.chatbot.chat(message)
                self.token_usage = self.chatbot.get_used_token()    # update total token usage
                self.last_used_token = self.chatbot.get_last_used_token()['total_tokens']
                self.used_tokens_in_minute += self.last_used_token  # update used token in this minute
                # dump JSON response
                parsed_response = json.loads(response)
                # record time usage of a sample
                sample_time_end = time.time()
                self.sample_time_usage.append(sample_time_end - sample_time_start)
                return parsed_response
            except Exception as e:
                print(f'Error: {e}, try again.')

    def show_sample_time_usage_statistics(self):
        sample_time_usage = np.array(self.sample_time_usage)
        mean_time = np.mean(sample_time_usage)
        min_time = np.min(sample_time_usage)
        max_time = np.max(sample_time_usage)
        total_time = np.sum(sample_time_usage)
        print('sample time usage statistics:')
        print(f'Total: {total_time}, samples: {len(self.sample_time_usage)}')
        print(f'Mean time: {mean_time}, min time: {min_time}, max time: {max_time}')
        money = self.get_used_token()['total_tokens'] / 1000000 * 5
        print(f'Earned: {money:.2f} USD')


'''
Prompts
'''
normal_reference_prompt = '''
<Background>: 
I have a long time series data with some abnormalities. I have converted the data into plots and I need your help to find the abnormality in the time series data.
This task contains two parts:
    - "Task1": I will give you some "normal reference" time series data slices without any abnormality. And you need to extract some valuable information from them to help me find the abnormality in the following time series data slices.
    - "Task2": I will give you some time series data slices with some abnormalities. You need to find the abnormality in them and provide some structured information.
besides, I will offer you some background information about the data plots:
    - The horizonal axis represents the time series index.
    - The vertical axis represents the value of the time series.
    - all normal reference data slices are from the same data channel but in different strides. Therefore, some patterns based on the position, for example, the position of peaks and the end of the plot, may cause some confusion.
    - all normal reference is a slice of the time series data with a fixed length and the same data channel. Therefore the beginning and the end of the plot may be different but the pattern should be similar.

<Task>: 
Now we are in the "Task1" part: I will give you some "normal reference" time series data slices without any abnormality. And you need to extract some valuable information from them to help me find the abnormality in the following time series data slices.

<Target>: 
Please help me extract some valuable information from them to help me find the abnormality in the following time series data slices.
The output should include some structured information, please output in JSON format:
    - normal_pattern (a 300-400 words paragraph): Try to describe the pattern of all "normal references" . All normal reference data slices are from the same data channel but in different strides. The abnormal pattern caused by truncation might be found at the begining and end of the sequence, do not pay too much attention to them. The description should cover at least the following aspects: period, stability, trend, peak, trough, and other important features.
Last, please double check before you submit your answer.
'''

anormaly_detection_prompt = '''
<Background>: 
I have a long time series data with some abnormalities. I have converted the data into plots and I need your help to find the abnormality in the time series data.
This task contains two parts:
    - "Task1": I will give you some "normal reference" time series data slices without any abnormality. And you need to extrace some valuable information from them to help me find the abnormality in the following time series data slices.
    - "Task2": I will give you some time series data slices with some abnormalities. You need to find the abnormality in them and provide some structured information.
besides, I will offer you some background information about the data plots:
    - The horizonal axis represents the time series index.
    - The vertical axis represents the value of the time series.
    - all normal reference data slices are from the same data channel but in different strides. Therefore, some patterns based on the position, for example, the position of peaks and the end of the plot, may cause some confusion.
    - all normal reference is a slice of the time series data with a fixed length and the same data channel. Therefore the beginning and the end of the plot may be different but the pattern should be similar.

<Task>: 
In "Task1" part, you have already extracted some valuable information from the "normal reference" time series data slices. You can use them to help you find the abnormality in the following time series data slices.
Now we are in "Task2", you are expected to detect the abnormality in the given data. 

<Target>: 
Please help me find the abnormality in this time series data slice and provide some structured information.
The output should include some structured information, please output in JSON format:
    - has_abnormality (string, answer "True" or "False"): Whether there is an abnormality in the time series data slice. The value should be "True" or "False".
    - abnormal_index (the output format should be like "[(start1, end1)/confidence_1, (start2, end2)/confidence_2, ...]", if there are some single outliers, the output should be "[(index1)/confidence_1, (index2)/confidence_2, ...]",if there is no abnormality, you can say "[]". The final output should can be mixed with these three formats.): The abnormality index of the time series. There are some requirements:
        + There may be multiple abnormalities in one stride, please try to find all of them. Pay attention to the range of each abnormality, the range should cover each whole abnormality in a suitable range.
        + Since the x-axis in the image only provides a limited number of tick marks, in order to improve the accuracy of your prediction, please try to estimate the coordinates of any anomaly locations based on the tick marks shown in the image as best as possible.
        + all normal reference data slices are from the same data channel but in different strides. Therefore, some patterns based on the position, for example, the position of peaks and the end of the plot, may cause some confusion.
        + all normal reference is a slice of the time series data with a fixed length and the same data channel. Therefore the beginning and the end of the plot may be different but the pattern should be similar.
    - abnormal_description (a 200-300 words paragraph): Make a brief description of the abnormality, why do you think it is abnormal? 
    - abnormal_type(string, answer from "none", "global", "contextual", "seasonal", "trend", "contextual"): The abnormality type of the time series, choose from [none, shapelet, seasonal, trend]. The detailed explanation is as follows:
        + none: No abnormality
        + global: Global outliers refer to the subsequences with dissimilar patterns compared with the normal reference
        + contextual: Contextual outliers refer to the subsequences with dissimilar patterns compared with the local context
        + shapelet: Shapelet outliers refer to the subsequences with dissimilar basic shapelets compared with the normal shapelet
        + seasonal: Seasonal outliers are the subsequences with unusual seasonalities compared with the overall seasonality
        + trend: Trend outliers indicate the subsequences that significantly alter the trend of the time series, leading to a permanent shift on the mean of the data.
    - abnormal_type_description (a 200-300 words paragraph): Make a brief description of the abnormality type, why do you think this type is suitable for the abnormality?
    - confidence (this must be an integer from 1 to 4): The confidence of your prediction. The value should be a integer between 1 and 4 which represents the confidence level of your prediction. Each level of confidence is explained as follows:
        + 1: No confidence: I am not sure about my prediction
        + 2: Low confidence: Weak evidence supports my prediction 
        + 3: medium confidence: strong evidence supports my prediction
        + 4: high confidence: more than 95% of the evidence supports my prediction
Last, please double check before you submit your answer.
'''
def make_normal_reference_response_prompt(normal_pattern):
    assistant_response_prompt = f'''
    The answer of "Task1" part is as follows: 
        - normal_pattern: {normal_pattern}
    '''
    return assistant_response_prompt
# double check
double_check_prompt = '''
<Background>: 
I have a long time series data with some abnormalities. I have converted the data into plots and I need your help to find the abnormality in the time series data.
There has been a response from another assistant, but I am not sure about the prediction. I need your help to double check the prediction.
besides, I will offer you some background information about the data plots:
    - The horizonal axis represents the time series index.
    - The vertical axis represents the value of the time series.
    - all normal reference data slices are from the same data channel but in different strides. Therefore, some patterns based on the position, for example, the position of peaks and the end of the plot, may cause some confusion.
    - all normal reference is a slice of the time series data with a fixed length and the same data channel. Therefore the beginning and the end of the plot may be different but the pattern should be similar.
<Task>:
Now, I will give you some "normal reference" and you are expected to double check the prediction of the abnormality in the given data.

<Target>:
The prediction of another assistant contains some information as flows:
    - has_abnormality: Whether there is an abnormality in the time series data slice. The value should be "True" or "False".
    - abnormal_index: The abnormality index of the time series. The output format should be like "[(start1, end1)/confidence_1, (start2, end2)/confidence_2, ...]", if there are some single outliers, the output should be "[(index1)/confidence_1, (index2)/confidence_2, ...]",if there is no abnormality, you can say "[]".
    - abnormal_description: Make a brief description of the abnormality, why do you think it is abnormal?
    - confidence: The confidence of your prediction. 
Prediction of another assistant:
    - has_abnormality: {has_abnormality}
    - abnormal_index: {abnormal_index}
    - abnormal_description: {abnormal_description}
    - confidence: {confidence}
Based on the "nomral reference" I gave you, please read the prediction above and double check the prediction. If you find any mistakes, please correct them. The output should include some structured information, please output in JSON format:
    - is_correct: Whether the prediction is correct. The value should be "True" or "False". 
    - fixed_abnormal_index (the output format should be like "[(start1, end1)/confidence_1, (start2, end2)/confidence_2, ...]", if there are some single outliers, the output should be "[(index1)/confidence_1, (index2)/confidence_2, ...]",if there is no abnormality, you can say "[]". The final output should can be mixed with these three formats.): The abnormality index of the time series. There are some requirements:
        + If the prediction is correct, please keep it. If you find mistakes, such as "not suitable range", "missing some abnormality", "wrong abnormality type", please correct them. 
        + Since the x-axis in the image only provides a limited number of tick marks, in order to improve the accuracy of your prediction, please try to estimate the coordinates of any anomaly locations based on the tick marks shown in the image as best as possible.
        + all normal reference data slices are from the same data channel but in different strides. Therefore, some patterns based on the position, for example, the position of peaks and the end of the plot, may cause some confusion.
        + all normal reference is a slice of the time series data with a fixed length and the same data channel. Therefore the beginning and the end of the plot may be different but the pattern should be similar.
    - The reason why you think the prediction is correct or incorrect. (a 200-300 words paragraph): Make a brief description of your double check, why do you think the prediction is correct or incorrect?
    - confidence (this must be an integer from 1 to 4): The confidence of your prediction. The value should be a integer between 1 and 4 which represents the confidence level of your prediction. Each level of confidence is explained as follows:
        + 1: No confidence: I am not sure about my prediction
        + 2: Low confidence: Weak evidence supports my prediction 
        + 3: medium confidence: strong evidence supports my prediction
        + 4: high confidence: more than 95% of the evidence supports my prediction
'''
def make_double_check_prompt(parsed_respon:dict):
    has_abnormality = parsed_respon['has_abnormality'] if 'has_abnormality' in parsed_respon else 'True'
    abnormal_index = parsed_respon['abnormal_index'] if 'abnormal_index' in parsed_respon else '[]'
    abnormal_description = parsed_respon['abnormal_description'] if 'abnormal_description' in parsed_respon else 'there is no abnormality'
    confidence = parsed_respon['confidence'] if 'confidence' in parsed_respon else 0
    prompt = double_check_prompt.format(has_abnormality=has_abnormality, abnormal_index=abnormal_index, abnormal_description=abnormal_description, confidence=confidence)
    return prompt


if __name__ == '__main__':
    # args
    args = args_parse()
    dataset_name = args.dataset
    refined = args.refined
    balanced = args.balanced
    ratio = args.ratio
    data_id_list = args.data_id_list
    print(f'dataset_name: {dataset_name}, refined: {refined}, balanced: {balanced}, ratio: {ratio}')
    # load dataset & normal reference helper
    log_save_path = os.path.join('/home/zhuangjiaxin/workspace/TensorTSL/TimeLLM/log', f'{dataset_name}_log.yaml')
    processed_data_path = '/home/zhuangjiaxin/workspace/TensorTSL/TimeLLM/output'
    normal_reference_base = os.path.join(processed_data_path, dataset_name)
    structure_yaml_path = os.path.join(normal_reference_base, 'test_structure.yaml')
    dataset = ProcessedDataset(os.path.join(processed_data_path, dataset_name), mode='test')
    normal_reference_helper = NormalReferenceHelper(normal_reference_base, structure_yaml_path)
    # init chatbot
    max_tokens=4096
    temperature=0.1
    top_p=0.3
    chatbot = Chat_AzureGPT4o(max_tokens, temperature, top_p)
    chat_controller = ChatController(chatbot)
    # init logger
    logger = Logger(log_save_path)
    # init message helper
    detect_messager = MessageHelper()
    detect_messager.set_system_message('you are a helpful assistant to detect the abnormality in the time series data. Please output in JSON format.')
    # preparation
    if data_id_list == [] or data_id_list == ['']:
        data_id_list = dataset.get_id_list()
    print(f'Data_id_list: {data_id_list}')
    pos, neg = dataset.get_instances(refined=refined, balanced=balanced, ratio=ratio)
    refined_data_id_list = pos + neg
    total_sample_num = 0
    sample_cnt = 1
    for data_id in data_id_list:
        data_info = dataset.get_data_id_info(data_id)
        num_stride = data_info['num_stride']
        num_channel = data_info['data_channels']
        for ch in range(num_channel):
            for stride_idx in range(num_stride):
                if refined or balanced:
                    if f'{data_id}-{stride_idx}-{ch}' in refined_data_id_list:
                        total_sample_num += 1
                else:
                    total_sample_num += 1
    print(f'Total sample number: {total_sample_num}')
    # start to chat
    for data_id in data_id_list:
        data_info = dataset.get_data_id_info(data_id)
        num_stride = data_info['num_stride']
        data_channel = data_info['data_channels']
        label_channel = data_info['label_channels']
        detect_messager.clean_message()
        for ch in range(data_channel):
            normal_reference_image_list = normal_reference_helper.find_normal_reference(data_id, ch)
            detect_messager.add_user_message(normal_reference_prompt, normal_reference_image_list)
            # print(normal_reference_image_list)
            # continue
            normal_reference_response = chat_controller.chat_with_rate_control(detect_messager.get_message())
            print(f'data_id: {data_id}, channel: {ch}, normal_reference_token_usage: {chat_controller.get_last_used_token()} >> Time: {chat_controller.get_last_sample_time_usage():.2f}s')
            normal_pattern = str(normal_reference_response['normal_pattern'])
            normal_reference_response_prompt = make_normal_reference_response_prompt(normal_pattern)
            detect_messager.add_chatbot_answer(normal_reference_response_prompt)
            # for each stride
            for stride_idx in range(num_stride):
                stride_msg_helper = detect_messager.copy_message()
                double_checker_msg_helper = detect_messager.copy_message()
                if refined or balanced:
                    if f'{data_id}-{stride_idx}-{ch}' not in refined_data_id_list:
                        continue
                image_path = dataset.get_image(data_id, stride_idx, ch)
                image_label = dataset.get_label(data_id, stride_idx, ch)
                label_index = np.where(image_label == 1)[0].tolist()
                # chat
                stride_msg_helper.add_user_message(anormaly_detection_prompt, [image_path])
                stride_response = chat_controller.chat_with_rate_control(stride_msg_helper.get_message())
                print(f'[{sample_cnt}/{total_sample_num}]>> data_id: {data_id}, stride: {stride_idx}, channel: {ch}, stride_token_usage: {chat_controller.get_last_used_token()} >> Time: {chat_controller.get_last_sample_time_usage():.2f}s')
                # double check
                double_check_prompt = make_double_check_prompt(stride_response)
                double_checker_msg_helper.add_user_message(double_check_prompt, [image_path])
                double_check_response = chat_controller.chat_with_rate_control(double_checker_msg_helper.get_message())
                print(f'[{sample_cnt}/{total_sample_num}]>> data_id: {data_id}, stride: {stride_idx}, channel: {ch}, double_check_token_usage: {chat_controller.get_last_used_token()} >> Time: {chat_controller.get_last_sample_time_usage():.2f}s')
                # log
                log_item = {
                    'data_id': data_id,
                    'num_stride': stride_idx,
                    'data_channel': ch,
                    'image': image_path,
                    'labels': str(label_index),
                    'abnormal_index': stride_response['abnormal_index'] if 'abnormal_index' in stride_response else '[]',
                    # 'abnormal_type': response.abnormal_type,
                    'has_abnormality': stride_response['has_abnormality'] if 'has_abnormality' in stride_response else 'False',
                    'abnormal_description': stride_response['abnormal_description'] if 'abnormal_description' in stride_response else '',
                    'confidence': stride_response['confidence'] if 'confidence' in stride_response else 0,
                    'normal_reference': {
                        'normal_image_list': str(normal_reference_image_list),
                        # 'normal_range': NR_response.normal_value_range,
                        'normal_pattern': normal_pattern
                    },
                    'double_check': {
                        'is_correct': double_check_response['is_correct'] if 'is_correct' in double_check_response else 'False',
                        'fixed_abnormal_index': double_check_response['fixed_abnormal_index'] if 'fixed_abnormal_index' in double_check_response else '[]',
                        'reason': double_check_response['reason'] if 'reason' in double_check_response else '',
                        'confidence': double_check_response['confidence'] if 'confidence' in double_check_response else 0,
                    }
                }
                logger.log(data_id, stride_idx, ch, log_item)
                sample_cnt += 1
                # if sample_cnt == 2:
                #     logger.save()
                #     chat_controller.show_sample_time_usage_statistics()
                #     exit()
    logger.save()
    chat_controller.show_sample_time_usage_statistics()