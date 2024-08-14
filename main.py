import os
import yaml
import numpy as np
import time
import subprocess

from BigModel.GLM import Chat_GLM4v
from BigModel.OpenAI import Chat_GPT4o, Chat_GPT4o_Structured
from Datasets.Dataset import RawDataset, ProcessedDataset

chat_config = yaml.safe_load(open('/home/zhuangjiaxin/workspace/TensorTSL/TimeLLM/configs/chat_config.yaml', 'r'))
task_config = yaml.safe_load(open('/home/zhuangjiaxin/workspace/TensorTSL/TimeLLM/configs/task_config.yaml', 'r'))

chatbot = Chat_GPT4o_Structured(**chat_config)
dataset_name = 'SMD'
dataset = ProcessedDataset(f'/home/zhuangjiaxin/workspace/TensorTSL/TimeLLM/output/{dataset_name}', mode='test')

task_prompt = f'''
<Task>: Here is a time series data. There may be some abnormality in it. I will offer you some background information about the data:
The background information is as follows: {dataset.get_background_info()}
The basic image information:
    - The dpi of the image is 100.
    - The x-axis represents the time series index.
    - The y-axis represents the value of the time series.
<Target>: Please help me analyze it. 
The output should include some structured information:
- abnormal_index: The abnormality index of the time series. There are some requirements:
    + the output format should be like "[(start1, end1), (start2, end2), ...]", if there are some single outliers, the output should be "[(index1), (index2), ...]",if there is no abnormality, you can say "[]". The final output should can be mixed with these three formats.
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

log_save_path = '/home/zhuangjiaxin/workspace/TensorTSL/TimeLLM/log'
logger = {}
time_list = []
# for i in range(dataset.data_num):
id_list = dataset.get_id_list()
id_data_num = dataset.get_id_data_num()
global_index = 0
for data_id in id_list:
    for i in range(id_data_num[data_id]):
        # print(f'ID: {data_id}, index {i} start')
        # continue
        # label = dataset.get_label_by_index(i)
        label = dataset.get_label_by_id_index(data_id, i)
        abnormal_index = np.where(label == 1)[0]
        abnormal_index = abnormal_index.tolist()
        # img_path = dataset.get_data_by_index(i)
        img_path = dataset.get_data_by_id_index(data_id, i)

        start_time = time.time()
        base64_img = chatbot.image_encoder_base64(img_path)
        message_content = make_message_content(base64_img)
        response = chatbot.chat(message_content)
        end_time = time.time()
        processing_time = end_time - start_time
        time_list.append(processing_time)
        print(f'ID: {data_id}, index {i} done -- TIME: {processing_time:.3f}s')

        if data_id not in logger:
            logger[data_id] = {}
        logger[data_id][i] = {
            'data_id': data_id,
            'index': i,
            'global_index': global_index,
            'image': img_path,
            'labels': str(abnormal_index),
            'abnormal_index': response.abnormal_index,
            'abnormal_type': response.abnormal_type,
            'abnormal_description': response.abnormal_description
        }
        global_index += 1
    #     break
    # break

with open(os.path.join(log_save_path, f'{dataset_name}_log.yaml'), 'w') as f:
    yaml.dump(logger, f)
print(f'Total time: {sum(time_list):.3f}s, Average time: {(sum(time_list)/(len(time_list))):.3f}s')

