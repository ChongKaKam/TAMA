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
import re
import matplotlib.pyplot as plt
import sys

from Datasets.Dataset import ProcessedDataset

from BigModel.Base import BigModelBase
from BigModel.Azure import Chat_AzureGPT4o, Chat_AzureGPT4o_mini
from BigModel.GLM import Chat_GLM4v
from BigModel.Claude import Chat_Claude
from BigModel.Groq import Chat_LLaVA_v1_5
from BigModel.QWen import Chat_Qwen_VL_MAX, Chat_Qwen_VL_PLUS
from BigModel.Gemini import Chat_Gemini_1_5_pro, Chat_Gemini_1_5_flash

ChatBotMAP = {
    'GPT-4o': Chat_AzureGPT4o,
    'GPT-4o-mini': Chat_AzureGPT4o_mini,
    'GLM-4v-plus': Chat_GLM4v,
    'Claude': Chat_Claude,
    'LLaVA': Chat_LLaVA_v1_5,
    'qwen-vl-max': Chat_Qwen_VL_MAX,
    'qwen-vl-plus':Chat_Qwen_VL_PLUS,
    'gemini-1.5-pro': Chat_Gemini_1_5_pro,
    'gemini-1.5-flash': Chat_Gemini_1_5_flash,
}
'''
Args Parser
'''
def args_parse():
    parser = argparse.ArgumentParser(description='Command line interface for main.py')
    parser.add_argument('--dataset', type=str, default='NormA', help='Dataset name')
    parser.add_argument('--refined', default=False, action='store_true')
    parser.add_argument('--balanced', default=False, action='store_true')
    parser.add_argument('--double_check', default=False, action='store_true')
    parser.add_argument('--normal_reference', type=int, default=3, required=False)
    parser.add_argument('--ratio', type=float, default=1.0, required=False)
    parser.add_argument('--data_id_list', type=str, default='', help='Data id list')
    parser.add_argument('--LLM', type=str, default='GPT-4o', help='[GPT-4o, GPT-4o-mini, GLM-4v-plus]')
    parser.add_argument('--subtask', type=str, default='', help='Subtask name', required=False)
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
    def __init__(self, chatbot:BigModelBase):
        self.message_list = []
        self.chatbot = chatbot
    
    def set_system_message(self, message):
        self.sys_message = {'role': 'system', 'content': message}
        self.message_list.append(self.sys_message)

    def add_message(self, role:str, text:str, image_path_list:list=[], text_path_list:list=[]):
        new_message = {
            'role': role,
            'content': [self.chatbot.text_item(text)],
        }
        for image_path in image_path_list:
            new_message['content'].append(self.chatbot.image_item_from_path(image_path))
        cnt = 1
        for text_path in text_path_list:
            with open(text_path, 'r') as f:
                text_data = f.read().replace('\n', ' ').strip('[]')
            new_message['content'].append(
                {
                    'type': 'text',
                    'text': f'data_{cnt}. {text_data}'
                }
            )
            cnt += 1
        self.message_list.append(new_message)

    def add_message_with_base64(self, role:str, text:str, image_base64_list:list=[]):
        new_message = {
            'role': role,
            'content': [self.chatbot.text_item(text)],
        }
        for image_base64 in image_base64_list:
            new_message['content'].append(self.chatbot.image_item_from_base64(image_base64))
        self.message_list.append(new_message)

    def add_user_message_with_texts(self, role:str, text:str, text_list:list):
        new_message = {
            'role': role,
            'content': [{'type': 'text', 'text': text}],
        }
        cnt = 1
        for text_data in text_list:
            new_message['content'].append(
                {
                    'type': 'text',
                    'text': f'data_{cnt}. {text_data}'
                }
            )
            cnt += 1
        self.message_list.append(new_message)

    def add_user_message(self, text:str, image_path_list:list=[], text_path_list:list=[]):
        self.add_message('user', text, image_path_list, text_path_list)
    def add_chatbot_answer(self, text:str, image_path_list:list=[], text_path_list:list=[]):
        self.add_message('assistant', text, image_path_list, text_path_list)
    def get_message(self):
        return self.message_list
    def clean_message(self):
        self.message_list = []
        # self.message_list.append(self.sys_message)
    def copy_message(self):
        new_helper = MessageHelper(self.chatbot)
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
        if os.path.exists(os.path.join(self.normal_reference_base, data_id, 'train', 'text')):
            file_list = os.listdir(os.path.join(self.normal_reference_base, data_id, 'train', 'text'))
            normal_list = []
            for i in range(len(file_list)):
                image_name = file_list[i]
                if image_name.endswith('.txt'):
                    normal_list.append(os.path.join(self.normal_reference_base, data_id, 'train', 'text', image_name))
            if num > len(normal_list):
                num = len(normal_list)
            sample_normal_list = normal_list[:num] if fixed else random.sample(normal_list, num)
            return sample_normal_list
        else:
            normal_list = self.structure_info[data_id]['normal']
            if len(normal_list) == 0:
                # raise Exception(f'No normal reference found in {self.structure_yaml_path} for data_id {data_id}.')
                return []
            if num > len(normal_list):
                num = len(normal_list)
            sample_normal_list = normal_list[:num] if fixed else random.sample(normal_list, num)
            print(f'Normal reference list: {sample_normal_list}')
            for idx in range(num):
                parsed_id = sample_normal_list[idx].split('-')
                image_path = os.path.join(self.normal_reference_base, data_id, 'test', 'text', f'{parsed_id[-2]}-{parsed_id[-1]}.txt')
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
WheelChair of JSON mode
'''
JSON_whitelist = ['GPT-4o', 'GPT-4o-mini']
class JSON_WheelChair:
    def __init__(self):
        self.chatbot = Chat_AzureGPT4o(4096, 0.1, 0.3)
        self.message_helper = MessageHelper(self.chatbot)
        self.prompt = 'You are a helpful JSON formater. \nI will give you a text and you need to format it into JSON format without any change. \nThe key is defined in below:(If there is no content of a key, you can output "")\n'
    def add_key(self, key:str, description:str):
        self.prompt += f'    - "{key}": {description}\n'
    def get_json(self, text:str):
        prompt = self.prompt + 'The text:\n' + text
        self.message_helper.add_user_message(prompt)
        response = self.chatbot.chat_retry(self.message_helper.get_message())
        print(response)
        return response
    
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
        self.start_time = time.time()
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
            print(f'slepp for {sleep_time} s...')
            time.sleep(sleep_time)
            self.start_timming()
    def chat_with_rate_control(self, message:list, max_retry:int=6, JSON_wheelchair=None)->dict:
        # chat with retry
        pause_retry = max_retry // 2
        retry_times = 0
        while retry_times <= max_retry:
            try:
                sample_time_start = time.time()
                self.last_used_token = 0
                self.check_rate_limit()
                response = self.chatbot.chat(message)
                self.token_usage = self.chatbot.get_used_token()    # update total token usage
                self.last_used_token = self.chatbot.get_last_used_token()['total_tokens']
                self.used_tokens_in_minute += self.last_used_token  # update used token in this minute
                if JSON_wheelchair is not None:
                    response = JSON_wheelchair.get_json(response)
                # dump JSON response
                parsed_response = json.loads(response)
                if retry_times > pause_retry:
                    print('Connection recovery...', file=sys.stderr)
                # record time usage of a sample
                sample_time_end = time.time()
                self.sample_time_usage.append(sample_time_end - sample_time_start)
                retry_times += 1
                return parsed_response
            except Exception as e:
                print(f'Error: {e}, try again. {retry_times}/{max_retry}')
                retry_times += 1
                time.sleep(int(5*retry_times))
                if retry_times == pause_retry:
                    print('There is something wrong, please check it and press any key to continue...', file=sys.stderr)
                    cmd = input('type [retry] to retry, [ok] to continue:')
                    if cmd == 'retry':
                        retry_times = 0
                    elif cmd == 'ok':
                        return {}

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
    - The horizontal axis represents the time series index.
    - The vertical axis represents the value of the time series.
    - all normal reference data slices are from the same data channel but in different strides. Therefore, some patterns based on the position, for example, the position of peaks and the end of the plot, may cause some confusion.
    - all normal references are slices of the time series data with a fixed length and the same data channel. Therefore the beginning and the end of the plot may be different but the pattern should be similar.

<Task>: 
Now we are in the "Task1" part: I will give you some "normal reference" time series data slices without any abnormality. And you need to extract some valuable information from them to help me find the abnormality in the following time series data slices.

<Target>: 
Please help me extract some valuable information from them to help me find the abnormality in the following time series data slices.
The output should include some structured information, please output in JSON format:
    - normal_pattern (a 300-400 words paragraph): Try to describe the pattern of all "normal references" . All normal reference data slices are from the same data channel but in different strides. The abnormal pattern caused by truncation might be found at the beginning and end of the sequence, do not pay too much attention to them. The description should cover at least the following aspects: period, stability, trend, peak, trough, and other important features.
Last, please double check before you submit your answer.
'''

anormaly_detection_prompt = '''
<Background>: 
I have a long time series data with some abnormalities. I have converted the data into plots and I need your help to find the abnormality in the time series data.
I will offer you some background information about the data plots:
    - The horizontal axis represents the time series index.
    - The vertical axis represents the value of the time series.
    - all normal reference data slices are from the same data channel but in different strides. Therefore, some patterns based on the position, for example, the position of peaks and the end of the plot, may cause some confusion.
    - all normal references are slices of the time series data with a fixed length and the same data channel. Therefore the beginning and the end of the plot may be different but the pattern should be similar.

<Task>: 
you are expected to detect the abnormality in the given data. 

<Target>: 
Please help me find the abnormality in this time series data slice and provide some structured information.
The output should include some structured information, please output in JSON format:
    - abnormal_index (the output format should be like "[(start1, end1)/confidence_1/abnormal_type_1, (start2, end2)/confidence_2/abnormal_type_2, ...]", if there is no abnormality, you can say "[]". The final output should be mixed with these three formats.): The abnormality index of the time series. There are some requirements:
        + The prediction of abnormal_index should be no more than 5 intervals.
        + Since the x-axis in the image only provides a limited number of tick marks, in order to improve the accuracy of your prediction, please try to estimate the coordinates of any anomaly locations based on the tick marks shown in the image as best as possible.
        + all normal reference data slices are from the same data channel but in different strides. Therefore, some patterns based on the position, for example, the position of peaks and the end of the plot, may cause some confusion.
        + abnormal_type(answer from "global", "contextual", "frequency", "trend", "contextual" "shapelet"): The abnormality type of the time series, choose from [global, contextual, frequency, trend, shapelet]. The detailed explanation is as follows:
            + global: Global outliers refer to the points that significantly deviate from the rest of the points. Try to position the outliers at the center of the interval.
            + contextual: Contextual outliers are the points that deviate from its corresponding context, which is defined as the neighboring time points within certain ranges. Try to position the outliers at the center of the interval.
            + frequency: Frequency outliers refer to changes in frequency, the basic shape of series remains the same. Please focuse on the horizontal axis to find the frequency anomalies.
            + trend: Trend outliers indicate the subsequences that significantly alter the trend of the time series, leading to a permanent shift on the mean of the data. Mark the intervals where the mean of the data significantly changes.
            + shapelet: Shapelet outliers refer to the subsequences with totally different shapes compared to the rest of the time series.
        - confidence (integer, from 1 to 4): The confidence of your prediction. The value should be an integer between 1 and 4, which represents the confidence level of your prediction. Each level of confidence is explained as follows:
            + 1: No confidence: I am not sure about my prediction
            + 2: Low confidence: Weak evidence supports my prediction 
            + 3: medium confidence: strong evidence supports my prediction
            + 4: high confidence: more than 95% of the evidence supports my prediction
        + based on the provided abnormal_type, you should double check the abnormal_index.
Last, please double check before you submit your answer.
'''
def make_normal_reference_response_text(normal_pattern):
    assistant_response_text = f'''
    The answer of "Task1" part is as follows: 
        - normal_pattern: {normal_pattern}
    '''
    return assistant_response_text

def make_anormaly_detection_response_text(parsed_respon:dict):
    abnormal_index = parsed_respon['abnormal_index'] if 'abnormal_index' in parsed_respon else '[]'
    abnormal_description = parsed_respon['abnormal_description'] if 'abnormal_description' in parsed_respon else ''
    abnormal_type_description = parsed_respon['abnormal_type_description'] if 'abnormal_type_description' in parsed_respon else ''
    # confidence = parsed_respon['confidence'] if 'confidence' in parsed_respon else 0
    assistant_response_text = f'''
    The answer of "Task2" part is as follows: 
        - abnormal_index: {abnormal_index}
        - abnormal_description: {abnormal_description}
        - abnormal_type_description: {abnormal_type_description}
    '''
    return assistant_response_text

# double check
double_check_prompt = '''
<Background>: 
I have a long time series data with some abnormalities. I have converted the data into plots and I need your help to find the abnormality in the time series data.
There has been a response from another assistant, but I am not sure about the prediction. I need your help to double check the prediction.
Besides, I will offer you some background information about the data plots:
    - The horizontal axis represents the time series index.
    - The vertical axis represents the value of the time series.
    - all normal reference data slices are from the same data channel but in different strides. Therefore, some patterns based on the position, for example, the position of peaks and the end of the plot, may cause some confusion.
    - all normal references are slices of the time series data with a fixed length and the same data channel. Therefore, the beginning and the end of the plot may be different, but the pattern should be similar.
<Task>:
Now, I will give you some "normal reference" and you are expected to double check the prediction of the abnormality in the given data.

<Target>:
The prediction of another assistant contains some information as follows:
    - abnormal_index: The abnormality index of the time series. The output format should be like "[(start1, end1)/confidence_1/abnormal_type_1, (start2, end2)/confidence_2/abnormal_type_2, ...]", if there are some single outliers, the output should be "[(index1)/confidence_1/abnormal_type_1, (index2)/confidence_2/abnormal_type_2, ...]",if there is no abnormality, you can say "[]".
    - abnormal_description: Make a brief description of the abnormality, why do you think it is abnormal?
Based on the "nomral reference" I gave you, please read the prediction above and double check the prediction. If you find any mistakes, please correct them. The output should include some structured information, please output in JSON format:
    - corrected_abnormal_index (string, the output format should be like "[(start1, end1)/confidence_1/abnormal_type_1, (start2, end2)/confidence_2/abnormal_type_2, ...]", if there are some single outliers, the output should be "[(index1)/confidence_1/abnormal_type_1, (index2)/confidence_2/abnormal_type_2, ...]",if there is no abnormality, you can say "[]". The final output should be mixed with these three formats.): The abnormality index of the time series. There are some requirements:
        + 1. you should check each prediction of the abnormal_type and make sure it is correct based on the abnormality index. If there is a incorrect prediction, you should remove it.
        + 2. you should check each prediction of the abnormal_index according to the image I gave to you. If there is an abnormality in image but not in the prediction, you should add it. The format should keep the same as the original prediction.
    - The reason why you think the prediction is correct or incorrect. (a 200-300 words paragraph): Make a brief description of your double check, why do you think the prediction is correct or incorrect?
'''
def make_double_check_prompt(parsed_respon:dict):
    # has_abnormality = parsed_respon['has_abnormality'] if 'has_abnormality' in parsed_respon else 'True'
    abnormal_index = parsed_respon['abnormal_index'] if 'abnormal_index' in parsed_respon else '[]'
    abnormal_description = parsed_respon['abnormal_description'] if 'abnormal_description' in parsed_respon else 'there is no abnormality'
    confidence = parsed_respon['confidence'] if 'confidence' in parsed_respon else 0
    prompt = double_check_prompt.format(abnormal_index=abnormal_index, abnormal_description=abnormal_description, confidence=confidence)
    return prompt


if __name__ == '__main__':
    # args
    args = args_parse()
    dataset_name = args.dataset
    refined = args.refined
    balanced = args.balanced
    ratio = args.ratio
    LLM_name = args.LLM
    subtask_name = args.subtask
    NormalReferenceNum = args.normal_reference
    doubel_check_enable = args.double_check
    data_id_list = args.data_id_list
    print(f'dataset_name: {dataset_name}, refined: {refined}, balanced: {balanced}, ratio: {ratio}')
    print(f'NormalReferenceNum: {NormalReferenceNum}, doubel_check_enable: {doubel_check_enable}')
    print(f'Subtask: {subtask_name}, Using LLM: {LLM_name}')
    # exit()
    # load dataset & normal reference helper
    # log_save_path = os.path.join('/home/zhuangjiaxin/workspace/TensorTSL/TimeLLM/log/text_modality', f'{dataset_name}_{LLM_name}_log.yaml')
    log_save_path = os.path.join('/home/zhuangjiaxin/workspace/TensorTSL/TimeLLM/log', subtask_name, f'{dataset_name}_{LLM_name}_log.yaml')
    # processed_data_path = '/home/zhuangjiaxin/workspace/TensorTSL/TimeLLM/output'
    processed_data_path = os.path.join('/home/zhuangjiaxin/workspace/TensorTSL/TimeLLM/output', subtask_name)
    normal_reference_base = os.path.join(processed_data_path, dataset_name)
    structure_yaml_path = os.path.join(normal_reference_base, 'test_structure.yaml')
    dataset = ProcessedDataset(os.path.join(processed_data_path, dataset_name), mode='test')
    normal_reference_helper = NormalReferenceHelper(normal_reference_base, structure_yaml_path)
    # init chatbot
    max_tokens=4096
    temperature=0.1
    top_p=0.3
    # chatbot = Chat_AzureGPT4o(max_tokens, temperature, top_p)
    # chatbot = Chat_GLM4v(max_tokens, temperature, top_p)
    chatbot = ChatBotMAP[LLM_name](max_tokens, temperature, top_p)
    # if 'GPT-4o' not in LLM_name:
    #     js_wheelchair = JSON_WheelChair()
    # else:
    #     js_wheelchair = None
    chat_controller = ChatController(chatbot)
    # init logger
    logger = Logger(log_save_path)
    # init message helper
    detect_messager = MessageHelper(chatbot)
    # detect_messager.set_system_message('you are a helpful assistant to detect the abnormality in the time series data. Please output in JSON format.')
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
            normal_reference_path_list = []
            if NormalReferenceNum > 0:
                normal_reference_path_list = normal_reference_helper.find_normal_reference(data_id, ch, NormalReferenceNum)
                if normal_reference_path_list == []:
                    continue
                detect_messager.add_user_message(normal_reference_prompt, text_path_list=normal_reference_path_list)
                # print(normal_reference_image_list)
                # continue
                if LLM_name not in JSON_whitelist:
                    normal_reference_chair = JSON_WheelChair()
                    normal_reference_chair.add_key('normal_pattern', '')
                else:
                    normal_reference_chair = None
                normal_reference_response = chat_controller.chat_with_rate_control(detect_messager.get_message(), JSON_wheelchair=normal_reference_chair)
                if normal_reference_response == {}:
                    continue
                print(f'data_id: {data_id}, channel: {ch}, normal_reference_token_usage: {chat_controller.get_last_used_token()} >> Time: {chat_controller.get_last_sample_time_usage():.2f}s')
                normal_pattern = str(normal_reference_response['normal_pattern'])
                normal_reference_response_prompt = make_normal_reference_response_text(normal_pattern)
                detect_messager.add_chatbot_answer(normal_reference_response_prompt)
            # for each stride
            for stride_idx in range(num_stride):
                stride_msg_helper = detect_messager.copy_message()
                # stride_msg_helper = detect_messager.copy_message()
                if refined or balanced:
                    if f'{data_id}-{stride_idx}-{ch}' not in refined_data_id_list:
                        continue
                text_path = dataset.get_text(data_id, stride_idx, ch)
                image_label = dataset.get_label(data_id, stride_idx, ch)
                label_index = np.where(image_label >= 1)[0].tolist()
                # chat
                stride_msg_helper.add_user_message(anormaly_detection_prompt, text_path_list=[text_path])
                if LLM_name not in JSON_whitelist:
                    stride_chair = JSON_WheelChair()
                    stride_chair.add_key('abnormal_index', 'the format is like "[(start1, end1)/confidence_1/abnormal_type_1, (start2, end2)/confidence_2/abnormal_type_2, ...]", the abnormal type is one of [global, contextual, frequency, trend, shapelet], the number of abnormal index should be no more than 5')
                    # stride_chair.add_key('abnormal_description', 'a 200-300 words paragraph')
                    # stride_chair.add_key('abnormal_type_description', 'a 200-300 words paragraph')
                else:
                    stride_chair = None
                stride_response = chat_controller.chat_with_rate_control(stride_msg_helper.get_message(), JSON_wheelchair=stride_chair)
                if stride_response == {}:
                    continue
                print(f'[{sample_cnt}/{total_sample_num}]>> data_id: {data_id}, stride: {stride_idx}, channel: {ch}, stride_token_usage: {chat_controller.get_last_used_token()} >> Time: {chat_controller.get_last_sample_time_usage():.2f}s')
                # double check
                response_abnormal_index = stride_response.get('abnormal_index', '[]')
                double_check_response = None
                if doubel_check_enable and response_abnormal_index != '[]':
                    # pattern_range = r'\(\d+,\s\d+\)/\d/[a-z]+'
                    # pattern_single = r'\(\d+\)/\d/[a-z]+'
                    # check_ranges = re.findall(pattern_range, stride_response['abnormal_index'])
                    # check_single = re.findall(pattern_single, stride_response['abnormal_index'])
                    # check_list = []
                    # base64_list = []
                    # for item in check_ranges:
                    #     index_range, confidence, index_type = item.split('/')
                    #     start, end = map(int, index_range.strip('()').split(','))
                    #     center = (start + end) // 2
                    #     check_list.append(center)
                    # for item in check_single:
                    #     index, confidence, index_type = item.split('/')
                    #     check_list.append(int(index))
                    # for check_idx in check_list:
                    #     data_for_check = dataset.get_data(data_id, stride_idx, ch)
                    #     window_length = data_for_check.shape[0]
                    #     if check_idx > 2*window_length//3:
                    #         data_for_check = data_for_check[window_length//3:]
                    #         xticks_start = window_length//3
                    #         xticks_end = window_length + 1
                    #     elif check_idx < window_length//3:
                    #         data_for_check = data_for_check[:int(2*window_length//3)]
                    #         xticks_start = 0
                    #         xticks_end = 2*window_length//3 + 1
                    #     else:
                    #         data_for_check = data_for_check[check_idx-window_length//3:check_idx+window_length//3]
                    #         xticks_start = check_idx - window_length//3
                    #         xticks_end = check_idx + window_length//3 + 1
                    data = dataset.get_data(data_id, stride_idx, ch)
                    window_length = data.shape[0]
                    xticks_start_list = [0, window_length//3]
                    xticks_end_list = [2*window_length//3, window_length]
                    base64_list = []
                    for check_idx in range(2):
                        xticks_start = xticks_start_list[check_idx]
                        xticks_end = xticks_end_list[check_idx]
                        data_for_check = data[xticks_start:xticks_end]
                        data_for_check = data_for_check.reshape(-1)
                        # fig, ax = plt.subplots(figsize=(15, 4), dpi=100)
                        # xticks = 5
                        # for x in range(0, len(data_for_check)+1, xticks):
                        #     ax.axvline(x, color='gray', linestyle='--', linewidth=0.5)
                        # ax.plot(data_for_check, color='blue')
                        # plt.xticks(rotation=90)
                        # ax.set_xticks(range(0, len(data_for_check)+1, xticks))
                        # ax.set_xticklabels(range(xticks_start, xticks_end+1, xticks))
                        # fig.tight_layout(pad=0.1)
                        # buffer = io.BytesIO()
                        # fig.savefig(buffer, format='png', bbox_inches='tight')
                        # # fig.savefig('./test.png', bbox_inches='tight')
                        # image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                        # base64_list.append(image_base64)
                        # plt.close(fig)
                        text_data = np.array2string(data_for_check, separator=',', precision=3, suppress_small=True).replace('\n', ' ').strip('[]')
                        base64_list.append(text_data)
                    double_check_msg_helper = stride_msg_helper.copy_message()
                    double_check_msg_helper.add_chatbot_answer(make_anormaly_detection_response_text(stride_response))
                    double_check_prompt = make_double_check_prompt(stride_response)
                    double_check_msg_helper.add_user_message_with_texts('user', double_check_prompt, base64_list)
                    if LLM_name not in JSON_whitelist:
                        double_check_chair = JSON_WheelChair()
                        double_check_chair.add_key('corrected_abnormal_index', 'the format is like "[(start1, end1)/confidence_1/abnormal_type_1, (start2, end2)/confidence_2/abnormal_type_2, ...]"')
                        # double_check_chair.add_key('reason', 'a 200-300 words paragraph')
                    else:
                        double_check_chair = None
                    double_check_response = chat_controller.chat_with_rate_control(double_check_msg_helper.get_message(), JSON_wheelchair=double_check_chair)
                    if double_check_response == {}:
                        continue
                    print(f'Double check >> check_idx: {check_idx}, double_check_token_usage: {chat_controller.get_last_used_token()} >> Time: {chat_controller.get_last_sample_time_usage():.2f}s')
                # log
                log_item = {
                    'data_id': data_id,
                    'num_stride': stride_idx,
                    'data_channel': ch,
                    'image': text_path,
                    'normal_reference_num': len(normal_reference_path_list),
                    'labels': str(label_index),
                    'abnormal_index': stride_response['abnormal_index'] if 'abnormal_index' in stride_response else '[]',
                    'abnormal_type_description': stride_response['abnormal_type_description'] if 'abnormal_type_description' in stride_response else '',
                    'abnormal_description': stride_response['abnormal_description'] if 'abnormal_description' in stride_response else '',
                    'normal_reference': {
                        'normal_image_list': str([]),
                        'normal_pattern': ""
                    },
                    'double_check': {}
                }
                if NormalReferenceNum > 0:
                    log_item['normal_reference'] = {
                        'normal_image_list': str(normal_reference_path_list),
                        'normal_pattern': normal_pattern
                    }
                if doubel_check_enable:
                    if double_check_response is not None:
                        log_item['double_check'] = {
                            # 'is_correct': double_check_response['is_correct'] if 'is_correct' in double_check_response else 'False',
                            'corrected_abnormal_index': double_check_response['corrected_abnormal_index'] if 'corrected_abnormal_index' in double_check_response else '[]',
                            'reason': double_check_response['reason'] if 'reason' in double_check_response else '',
                        }
                logger.log(data_id, stride_idx, ch, log_item)
                sample_cnt += 1
                # if sample_cnt == 3:
                #     logger.save()
                #     chat_controller.show_sample_time_usage_statistics()
                #     exit()
    logger.save()
    chat_controller.show_sample_time_usage_statistics()