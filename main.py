import os
import yaml
import numpy as np
import time

from BigModel.GLM import Chat_GLM4v
from BigModel.OpenAI import Chat_GPT4o, Chat_GPT4o_Structured
from Datasets.Dataset import RawDataset, ProcessedDataset

chat_config = yaml.safe_load(open('/home/zhuangjiaxin/workspace/TensorTSL/TimeLLM/configs/chat_config.yaml', 'r'))
task_config = yaml.safe_load(open('/home/zhuangjiaxin/workspace/TensorTSL/TimeLLM/configs/task_config.yaml', 'r'))

chatbot = Chat_GPT4o_Structured(**chat_config)
dataset = ProcessedDataset('/home/zhuangjiaxin/workspace/TensorTSL/TimeLLM/output/UCR', mode='test')

task_prompt = f'''
<Task>: Here is a time series data. There may be some abnormality in it. I will offer you some background information about the data:
The background information is as follows: {dataset.get_background_info()}
<Target>: Please help me analyze it. 
The output should include some structured information:
- abnormal_index: The abnormality index of the time series. There are some requirements:
    + the output format should be like "[index1, index2, index3, ...]", if there is no abnormality, you can say "[]"
    + Since the x-axis in the image only provides a limited number of tick marks, in order to improve the accuracy of your prediction, please try to estimate the coordinates of any anomaly locations based on the tick marks shown in the image as best as possible.
- abnormal_type: The abnormality type of the time series, choose from [none, shapelet, seasonal, trend]. The detailed explanation is as follows:
    + none: No abnormality
    + shapelet: Shapelet outliers refer to the subsequences with dissimilar basic shapelets compared with the normal shapelet
    + seasonal: Seasonal outliers are the subsequences with unusual seasonalities compared with the overall seasonality
    + trend: Trend outliers indicate the subsequences that significantly alter the trend of the time series, leading to a permanent shift on the mean of the data.
- abnormal_description: Make a brief description of the abnormality, why do you think it is abnormal? 
'''

def make_message_content(base64_img):
    message_content = [
        {
            "type": "text", 
            "text": task_prompt,
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{base64_img}"},
        }
    ]
    return message_content

# img_path = dataset.get_data_by_index(0)
# print(f'image: {img_path}')
# base64_img = chatbot.image_encoder_base64(img_path)
# message_content = make_message_content(base64_img)
# response = chatbot.chat(message_content)
# print(response)
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
os.environ['all_proxy'] = 'socks5://127.0.0.0.1:7890'
logger = {}
log_save_path = '/home/zhuangjiaxin/workspace/TensorTSL/TimeLLM/log'
time_list = []
for i in range(dataset.data_num):
    start_time = time.time()
    label = dataset.get_label_by_index(i)
    abnormal_index = np.where(label == 1)[0]
    abnormal_index = abnormal_index.tolist()
    # if len(abnormal_index) == 0:
    #     abnormal_index = '[()]'
    img_path = dataset.get_data_by_index(i)
    base64_img = chatbot.image_encoder_base64(img_path)
    message_content = make_message_content(base64_img)
    response = chatbot.chat(message_content)
    # print(f'image: {img_path}')
    
    # print('Models:')
    # print(f'index: {response.abnormal_index}')
    # print(f'type: {response.abnormal_type}')
    # print(f'description: {response.abnormal_description}')
    end_time = time.time()
    processing_time = end_time - start_time
    time_list.append(processing_time)
    print(f'index {i} done -- TIME: {processing_time:.3f}s')
    key_i = f"idx_{i}"
    logger[key_i] = {
        'image': img_path,
        'labels': str(abnormal_index),
        'abnormal_index': response.abnormal_index,
        'abnormal_type': response.abnormal_type,
        'abnormal_description': response.abnormal_description
    }
with open(os.path.join(log_save_path, f'UCR_log.yaml'), 'w') as f:
    yaml.dump(logger, f)
print(f'Total time: {sum(time_list):.3f}s, Average time: {(sum(time_list)/(len(time_list))):.3f}s')
    
