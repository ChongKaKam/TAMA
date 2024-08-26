import os
import yaml
import numpy as np
import time
import json
import random
import argparse

from BigModel.Azure import Chat_AzureGPT4o
from Datasets.Dataset import ProcessedDataset

# 1. load configuration
chat_config = yaml.load(open('configs/chat_config.yaml', 'r'), Loader=yaml.FullLoader)
task_config = yaml.load(open('configs/task_config.yaml', 'r'), Loader=yaml.FullLoader)

log_save_path = task_config['log_save_path']
processed_data_path = task_config['processed_data_path']

dataset_name = task_config['dataset_name']
data_id_list = task_config['data_id_list']
refined = task_config['refined']
balanced = task_config['balanced']
data_ratio = task_config['ratio']

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default=dataset_name)
parser.add_argument('--refined', default=False, action='store_true')
parser.add_argument('--balanced', default=False, action='store_true')
parser.add_argument('--ratio', type=float, default=data_ratio, required=False)
args = parser.parse_args()

dataset_name = args.dataset
refined = args.refined
balanced = args.balanced
data_ratio = args.ratio
print(f'Dataset: {dataset_name}, refined: {refined}, balanced: {balanced}, ratio: {data_ratio}')

# exit()
# 2. init chatbot
chatbot = Chat_AzureGPT4o(**chat_config)
token_rate_limit = chatbot.get_tokens_per_minute()

# 3. load dataset
dataset = ProcessedDataset(os.path.join(processed_data_path, dataset_name), mode='test')

# 4. message helper
class MessageHelper:
    def __init__(self):
        self.message_list = []
    
    def add_system_message(self, text:str):
        sys_message = {
            "role": "system",
            "content": text,
        }
        self.message_list.append(sys_message)

    def add_user_message(self, text:str, image_list:list=[], detail='auto'):
        new_message = {
            'role': 'user',
            'content': [{"type": "text", "text": text}],
        }
        for image in image_list:
            base64_image = chatbot.image_encoder_base64(image)
            new_message['content'].append({
                "type": "image_url", 
                "image_url": {"url": f"data:image/png;base64,{base64_image}", "detail": detail,},
            })
        self.message_list.append(new_message)

    def add_chatbot_answer(self, text:str, image_list:list=[], detail='auto'):
        new_message = {
            "role": "assistant",
            "content": [{"type": "text", "text": text,}]
        }
        for image in image_list:
            base64_img = chatbot.image_encoder_base64(image)
            new_message['content'].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_img}", "detail": detail,},
            })
        self.message_list.append(new_message)

    def get_message(self):
        return self.message_list
    
    def copy_message(self):
        new_helper = MessageHelper()
        new_helper.message_list = self.message_list.copy()
        return new_helper
    
    def clean_message(self):
        self.message_list = []

# 5. prompt
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

logger = {}
message_helper = MessageHelper()
if data_id_list == []:
    data_id_list = dataset.get_id_list()
print('Data id list:', data_id_list)

pos, neg = dataset.get_instances(refined=refined, balanced=balanced, ratio=data_ratio)
refined_data_id_list = pos + neg
# print(f'Refined data id list: {refined_data_id_list}')
# exit()
total_num = 0
cnt = 0
for data_id in data_id_list:
    data_id_info = dataset.get_data_id_info(data_id)
    num_stride = data_id_info['num_stride']
    data_channels = data_id_info['data_channels']
    total_num += int(num_stride * data_channels)
print(f'Total num: {total_num}')

time_list = []
current_min = 0
token_per_min = 0

normal_ref_base = os.path.join(processed_data_path, dataset_name)
structure_yaml = yaml.safe_load(open(os.path.join(normal_ref_base, 'test_structure.yaml'), 'r'))
def find_normal_reference(dataset_name, data_id, channel)->list:
    normal_list = structure_yaml[data_id]['normal']
    if len(normal_list) == 0:
        for key in structure_yaml.keys():
            if len(structure_yaml[key]['normal']) > 0:
                normal_list = structure_yaml[key]['normal']
            break
    image_num = 3
    if len(normal_list) < image_num:
        image_num = len(normal_list)
    sample_nomral_list = random.sample(normal_list, image_num)
    print(f'Normal list: {sample_nomral_list}')
    for i in range(image_num):
        parsed = sample_nomral_list[i].split('-')
        image_path = os.path.join(normal_ref_base, data_id, 'test', 'image', f'{parsed[1]}-{parsed[2]}.png')
        sample_nomral_list[i] = image_path
    return sample_nomral_list
    # if dataset_name == 'SMD':
    #     image_path = os.path.join(normal_ref_base, data_id, 'train', 'image', f'6-{channel}.png')
    #     return [image_path]
    # elif dataset_name == 'UCR':
    #     image_list = []
    #     image_num = 3
    #     for i in range(image_num):
    #         image_path = os.path.join(normal_ref_base, data_id, 'train', 'image', f'{i}-{channel}.png')
    #         image_list.append(image_path)
    #     return image_list
    # elif dataset_name == 'NAB':
    #     image_list = []
    #     image_num = 1
    #     for i in range(image_num):
    #         # /home/zhuangjiaxin/workspace/TensorTSL/TimeLLM/output/NAB/0/test/image/0-0.png
    #         image_path = os.path.join(normal_ref_base, data_id, 'test', 'image', f'{i}-0.png')
    #         image_list.append(image_path)
    #     return image_list
    # elif dataset_name == 'KDD-TSAD':
    #     image_list = []
    #     image_num = 1
    #     image_path = os.path.join(f'/home/zhuangjiaxin/workspace/TensorTSL/TimeLLM/output/KDD-TSAD/{data_id}/test/image/0-0.png')
    #     image_list.append(image_path)
    #     return image_list
    # elif dataset_name == 'NormA':
    #     image_list = []
    #     image_num = 3
    #     for i in [0,2,4]:
    #         image_path = os.path.join(normal_ref_base, data_id, 'test', 'image', f'{i}-0.png')
    #         image_list.append(image_path)
    #     return image_list
    # else:
    #     raise ValueError(f'Unknown dataset name: {dataset_name}')

# 6. run
# log_file = open(os.path.join(log_save_path, f'{dataset_name}_log.yaml'), 'w')
start_min = time.time()
message_helper.add_system_message("you are a helpful assistant designed to output JSON.")
for data_id in data_id_list:
    data_id_info = dataset.get_data_id_info(data_id)
    num_stride = data_id_info['num_stride']
    data_channels = data_id_info['data_channels']
    label_channels = data_id_info['label_channels']
    if data_id not in logger:
        logger[data_id] = {}
    message_helper.clean_message()
    for ch in range(data_channels):
        # normal reference
        # normal_reference_image = os.path.join(normal_ref_base, data_id, 'train', 'image', f'6-{ch}.png')    # SMD

        normal_reference_image_list = find_normal_reference(dataset_name, data_id, ch)
        message_helper.add_user_message(normal_reference_prompt, normal_reference_image_list, "high")
        for _ in range(3):
            try:
                NR_response = chatbot.chat(message_helper.get_message())
                used_tokens = chatbot.get_used_token()
                print(f'New Channel: Normal Reference Used tokens: {used_tokens}')
                parsed_response = json.loads(NR_response)
                if 'normal_pattern' in parsed_response:
                    break
            except Exception as e:
                print(f'Error: {e}')
                continue
        channel_normal_reference = str(parsed_response['normal_pattern'])
        normal_reference_knowledge = make_normal_reference_response_prompt(channel_normal_reference)
        message_helper.add_chatbot_answer(normal_reference_knowledge)
        # anormaly detection
        for i in range(num_stride):
            if i not in logger[data_id]:
                logger[data_id][i] = {}
            stride_msg_helper = message_helper.copy_message()
            double_checker = message_helper.copy_message()
            if refined or balanced:
                if f"{data_id}-{i}-{ch}" not in refined_data_id_list:
                    print(f'[{cnt}/{total_num}]>> id: {data_id}, num_stride {i}, channel: {ch} done, Used tokens: {0} --> TIME: 0.0s')
                    cnt += 1
                    continue
            image_path = dataset.get_image(data_id, i, ch)
            image_label = dataset.get_label(data_id, i, ch)
            label_index = np.where(image_label == 1)[0].tolist()

            # Chat with the chatbot
            start_time = time.time()
            stride_msg_helper.add_user_message(anormaly_detection_prompt, [image_path], "high")
            # Add rate limits
            for _ in range(3):
                try:
                    used_min = (start_time - start_min) // 60
                    if used_min > current_min:
                        current_min = used_min
                        token_per_min = token_per_min + used_tokens['total_tokens']
                        if token_per_min > token_rate_limit:
                            current_time = time.time()
                            sleep_time = 60 - (current_time - start_min) % 60
                            print(f'Rate limit reached {token_per_min}/{token_rate_limit}, sleep for {sleep_time}s')
                            token_per_min = 0
                            time.sleep(sleep_time)
                            start_time = time.time()    # reset start time
                    response = chatbot.chat(stride_msg_helper.get_message())
                    parsed_response = json.loads(response)
                    double_check_msg = make_double_check_prompt(parsed_response)
                    double_checker.add_user_message(double_check_msg, [image_path], "high")
                    double_check_response = chatbot.chat(double_checker.get_message())
                    parsed_double_check_response = json.loads(double_check_response)
                    break
                except Exception as e:
                    print(f'Error: {e}')
                    continue
            # print(parsed_response)
            # exit()
            used_tokens = chatbot.get_last_used_token()
            # print(response.has_abnormality, type(response.has_abnormality));exit()
            # if response.has_abnormality == 'True':
            #     print(f'Chatbot: double check... (used token: {used_tokens})')
            #     chatbot_ans = make_double_check_response_prompt(response.has_abnormality, response.confidence, response.abnormal_index, response.abnormal_description)
            #     stride_msg_helper.add_chatbot_message(chatbot_ans)
            #     stride_msg_helper.add_user_message(double_check_prompt, [image_path], "high")
            #     response = chatbot.chat(stride_msg_helper.get_message(), FormatModel=anormaly_detection_response)
            end_time = time.time()
            used_tokens = chatbot.get_last_used_token()

            processing_time = end_time - start_time
            time_list.append(processing_time)
            cnt += 1
            print(f'[{cnt}/{total_num}]>> id: {data_id}, num_stride {i}, channel: {ch} done, Used tokens: {used_tokens} --> TIME: {processing_time:.3f}s')
            # logger
            logger[data_id][i][ch] = {
                'data_id': data_id,
                'num_stride': i,
                'data_channel': ch,
                'image': image_path,
                'labels': str(label_index),
                'abnormal_index': parsed_response['abnormal_index'] if 'abnormal_index' in parsed_response else '[]',
                # 'abnormal_type': response.abnormal_type,
                'has_abnormality': parsed_response['has_abnormality'] if 'has_abnormality' in parsed_response else 'False',
                'abnormal_description': parsed_response['abnormal_description'] if 'abnormal_description' in parsed_response else '',
                'confidence': parsed_response['confidence'] if 'confidence' in parsed_response else 0,
                'normal_reference': {
                    'normal_image_list': str(normal_reference_image_list),
                    # 'normal_range': NR_response.normal_value_range,
                    'normal_pattern': channel_normal_reference
                },
                'double_check': {
                    'is_correct': parsed_double_check_response['is_correct'] if 'is_correct' in parsed_double_check_response else 'False',
                    'fixed_abnormal_index': parsed_double_check_response['fixed_abnormal_index'] if 'fixed_abnormal_index' in parsed_double_check_response else '[]',
                    'reason': parsed_double_check_response['reason'] if 'reason' in parsed_double_check_response else '',
                    'confidence': parsed_double_check_response['confidence'] if 'confidence' in parsed_double_check_response else 0,
                }
            }
            # if cnt == 3:
            #     # print(logger)
            #     break
    # 7.save log
    # yaml.dump({data_id: logger[data_id]}, log_file)

# 7. save log  
with open(os.path.join(log_save_path, f'{dataset_name}_log.yaml'), 'w') as f:
    yaml.dump(logger, f)

print(f'Total num: {total_num}, Total time: {sum(time_list):.3f}s, Average time: {(sum(time_list)/(len(time_list))):.3f}s')
used_tokens = chatbot.get_used_token()
print(f'Used tokens: {used_tokens}')
cost = used_tokens['total_tokens'] / 1000000 * 5
print(f'Earned: {cost:.2f} USD')