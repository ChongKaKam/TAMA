import os
import yaml
import numpy as np
import time
from pydantic import BaseModel, Field

from BigModel.GLM import Chat_GLM4v
from BigModel.OpenAI import Chat_GPT4o, Chat_GPT4o_Structured
from Datasets.Dataset import RawDataset, ProcessedDataset

# 改变当前工作目录为文件所在目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 1. load configuration
chat_config = yaml.safe_load(open('./configs/chat_config.yaml', 'r'))
task_config = yaml.safe_load(open('./configs/task_config.yaml', 'r'))
dataset_name = task_config['dataset_name']
log_save_path = task_config['log_save_path']
data_id_list = task_config['data_id_list']
refined = task_config['refined']
balanced = task_config['balanced']
data_ratio = task_config['ratio']
processed_data_path = task_config['processed_data_path']
normal_reference_lookup = task_config['normal_reference']

# 2. init chatbot
chatbot = Chat_GPT4o_Structured(**chat_config)
token_rate_limit = chatbot.get_tokens_per_minute()
# 3. load dataset
dataset = ProcessedDataset(os.path.join(processed_data_path, dataset_name), mode='test')


# 4. task prompt & message helper
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
The output should include some structured information:
    - normal_value_range: The normal value range of the time series. The format should be like "[min, max]".
    - normal_pattern: Try to describe the pattern of all "normal references" to help me find the abnormality in the following as best as possible. All normal reference data slices are from the same data channel but in different strides. Therefore, some patterns based on the position, for example, the position of peaks and the end of the plot, may cause some confusion. Please pay attention to it. (The words of output should be 200-300 words).
Last, please double check before you submit your answer.
'''
class normal_reference_response(BaseModel):
    normal_value_range: str = Field(description='The normal value range of the time series. The format should be like "[min, max]".')
    normal_pattern: str = Field(description='Try to describe the pattern of all "normal references" to help me find the abnormality in the following as best as possible. The words of output should be 200-300 words.')


def make_normal_reference_response_prompt(normal_value_range, normal_pattern):
    assistant_response_prompt = f'''
    The answer of "Task1" part is as follows: 
        - normal_value_range (the range of sequence may float at a slight level, so the range is not fixed, but the range should be in the same level): {normal_value_range}
        - normal_pattern: {normal_pattern}
    '''
    
    return assistant_response_prompt

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
The output should include some structured information:
    - has_abnormality: Whether there is an abnormality in the time series data slice. The value should be "True" or "False".
    - abnormal_index: The abnormality index of the time series. There are some requirements:
        + the output format should be like "[(start1, end1), (start2, end2), ...]", if there are some single outliers, the output should be "[(index1), (index2), ...]",if there is no abnormality, you can say "[]". The final output should can be mixed with these three formats.
        + Since the x-axis in the image only provides a limited number of tick marks, in order to improve the accuracy of your prediction, please try to estimate the coordinates of any anomaly locations based on the tick marks shown in the image as best as possible.
        + all normal reference data slices are from the same data channel but in different strides. Therefore, some patterns based on the position, for example, the position of peaks and the end of the plot, may cause some confusion.
        + all normal reference is a slice of the time series data with a fixed length and the same data channel. Therefore the beginning and the end of the plot may be different but the pattern should be similar.
    - abnormal_description: Make a brief description of the abnormality, why do you think it is abnormal? 
    - confidence: The confidence of your prediction. The value should be a integer between 1 and 4 which represents the confidence level of your prediction. Each level of confidence is explained as follows:
        + 1: No confidence: I am not sure about my prediction
        + 2: Low confidence: Weak evidence supports my prediction 
        + 3: medium confidence: strong evidence supports my prediction
        + 4: high confidence: more than 95% of the evidence supports my prediction
Last, please double check before you submit your answer.
'''
class anormaly_detection_response(BaseModel):
        has_abnormality: str = Field(description="Whether there is an abnormality in the time series data slice. The value should be 'True' or 'False'.")
        abnormal_index: str = Field(description="the output format should be like '[(start1, end1), (start2, end2), ...]', if there are some single outliers, the output should be '[(index1), (index2), ...]',if there is no abnormality, you can say '[]'. The final output should can be mixed with these three formats.")
        abnormal_description: str = Field(description="Make a brief description of the abnormality, why do you think it is abnormal?")
        confidence: int = Field(description="confidence: The confidence of your prediction. The value should be a integer between 1 and 4 which represents the confidence level of your prediction.")

double_check_prompt = '''
based on your response, you believe there is a abnormality in this data slice. I will give you a high resultion image to help you double check.
Please help me find the abnormality in this time series data slice and provide some structured information:
    - has_abnormality: Whether there is an abnormality in the time series data slice. The value should be "True" or "False".
    - abnormal_index: The abnormality index of the time series. There are some requirements:
        + the output format should be like "[(start1, end1), (start2, end2), ...]", if there are some single outliers, the output should be "[(index1), (index2), ...]",if there is no abnormality, you can say "[]". The final output should can be mixed with these three formats.
        + Since the x-axis in the image only provides a limited number of tick marks, in order to improve the accuracy of your prediction, please try to estimate the coordinates of any anomaly locations based on the tick marks shown in the image as best as possible.
        + all normal reference data slices are from the same data channel but in different strides. Therefore, some patterns based on the position, for example, the position of peaks and the end of the plot, may cause some confusion.
        + all normal reference is a slice of the time series data with a fixed length and the same data channel. Therefore the beginning and the end of the plot may be different but the pattern should be similar.
    - abnormal_description: Make a brief description of the abnormality, why do you think it is abnormal? 
    - confidence: The confidence of your prediction. The value should be a integer between 1 and 4 which represents the confidence level of your prediction. Each level of confidence is explained as follows:
        + 1: No confidence: I am not sure about my prediction
        + 2: Low confidence: Weak evidence supports my prediction 
        + 3: medium confidence: strong evidence supports my prediction
        + 4: high confidence: more than 95% of the evidence supports my prediction
'''
def make_double_check_response_prompt(has_abnormality, confidence, abnormal_index, abnormal_description):
    assistant_response_prompt = f'''
    The answer is as follows: 
        - has_abnormality: {has_abnormality}
        - abnormal_index: {abnormal_index}
        - abnormal_description: {abnormal_description}
        - confidence: {confidence}
    '''
    
    return assistant_response_prompt

# Message Helper: Help to generate the message content and keep the message list
class MessageHelper:
    def __init__(self):
        self.message_list = []

    def add_user_message(self, text:str, image_list:list=[], detail='auto'):
        new_message = {
            "role": "user",
            "content": [{"type": "text", "text": text,}]
        }
        for image in image_list:
            base64_img = chatbot.image_encoder_base64(image)
            new_message['content'].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_img}", "detail": detail,},
            })
        self.message_list.append(new_message)
        
    def add_chatbot_message(self, text:str, image_list:list=[], detail='auto'):
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

# 5. prepare for running
logger = {}
message_helper = MessageHelper()
if data_id_list == []:
    data_id_list = dataset.get_id_list()
# check = dataset.id_list
pos,neg = dataset.get_instances(refined=refined, balanced=balanced, ratio=data_ratio)
refined_data_id_list = pos + neg
total_num = 0
cnt = 0
for data_id in data_id_list:
    data_id_info = dataset.get_data_id_info(data_id)
    num_stride = data_id_info['num_stride']
    data_channels = data_id_info['data_channels']
    total_num += int(num_stride * data_channels)

# used_token = {
#     'completion_tokens': 0,
#     'prompt_tokens': 0,
#     'total_tokens': 0
# }

time_list = []
current_min = 0
token_per_min = 0
# 'SMD': '/home/zhuangjiaxin/workspace/TensorTSL/TimeLLM/output/SMD',
# 'UCR': '/home/zhuangjiaxin/workspace/TensorTSL/TimeLLM/output/UCR'
normal_ref_base = os.path.join(processed_data_path, dataset_name)
def find_normal_reference(dataset_name, data_id, channel)->list:
    if dataset_name == 'SMD':
        image_path = os.path.join(normal_ref_base, data_id, 'train', 'image', f'6-{channel}.png')
        return [image_path]
    elif dataset_name == 'UCR':
        image_list = []
        image_num = 3
        for i in range(image_num):
            image_path = os.path.join(normal_ref_base, data_id, 'train', 'image', f'{i}-{channel}.png')
            image_list.append(image_path)
        return image_list
    elif dataset_name == 'NAB':
        image_list = []
        image_num = 1
        for i in range(image_num):
            # /home/zhuangjiaxin/workspace/TensorTSL/TimeLLM/output/NAB/0/test/image/0-0.png
            image_path = os.path.join(normal_ref_base, data_id, 'test', 'image', f'{i}-0.png')
            image_list.append(image_path)
        return image_list
    else:
        raise ValueError(f'Unknown dataset name: {dataset_name}')
        

# 6. run
log_file = open(os.path.join(log_save_path, f'{dataset_name}_log.yaml'), 'w')
start_min = time.time()
for data_id in data_id_list:
    data_id_info = dataset.get_data_id_info(data_id)
    num_stride = data_id_info['num_stride']
    data_channels = data_id_info['data_channels']
    label_channels = data_id_info['label_channels']
    logger[data_id] = {}
    message_helper.clean_message()
    for ch in range(data_channels):
        # normal reference
        # normal_reference_image = os.path.join(normal_ref_base, data_id, 'train', 'image', f'6-{ch}.png')    # SMD

        normal_reference_image_list = find_normal_reference(dataset_name, data_id, ch)
        message_helper.add_user_message(normal_reference_prompt, normal_reference_image_list, "high")
        NR_response = chatbot.chat(message_helper.get_message(), FormatModel=normal_reference_response)
        used_tokens = chatbot.get_used_token()
        print(f'New Channel: Normal Reference Used tokens: {used_tokens}')
        normal_reference_knowledge = make_normal_reference_response_prompt(NR_response.normal_value_range, NR_response.normal_pattern)
        message_helper.add_chatbot_message(normal_reference_knowledge)
        # anormaly detection
        for i in range(num_stride):
            if i not in logger[data_id]:
                logger[data_id][i] = {}
            stride_msg_helper = message_helper.copy_message()
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
            response = chatbot.chat(stride_msg_helper.get_message(), FormatModel=anormaly_detection_response)
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
                'abnormal_index': response.abnormal_index,
                # 'abnormal_type': response.abnormal_type,
                'has_abnormality': response.has_abnormality,
                'abnormal_description': response.abnormal_description,
                'confidence': response.confidence,
                'normal_reference': {
                    'normal_image_list': str(normal_reference_image_list),
                    'normal_range': NR_response.normal_value_range,
                    'normal_pattern': NR_response.normal_pattern
                }
                
            }
    # 7.save log
    yaml.dump({data_id: logger[data_id]}, log_file)

# 7. save log  
# with open(os.path.join(log_save_path, f'{dataset_name}_log.yaml'), 'w') as f:
#     yaml.dump(logger, f)

print(f'Total num: {total_num}, Total time: {sum(time_list):.3f}s, Average time: {(sum(time_list)/(len(time_list))):.3f}s')
used_tokens = chatbot.get_used_token()
print(f'Used tokens: {used_tokens}')
cost = used_tokens['total_tokens'] / 1000000 * 5
print(f'Cost: {cost:.2f} USD') 