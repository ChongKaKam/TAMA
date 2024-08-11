import os
import yaml

from BigModel.GLM import Chat_GLM4v
from BigModel.OpenAI import Chat_GPT4o, Chat_GPT4o_Structured
from Datasets.Dataset import RawDataset, ProcessedDataset

chat_config = yaml.safe_load(open('/home/zhuangjiaxin/workspace/TensorTSL/TimeLLM/configs/chat_config.yaml', 'r'))
task_config = yaml.safe_load(open('/home/zhuangjiaxin/workspace/TensorTSL/TimeLLM/configs/task_config.yaml', 'r'))

chatbot = Chat_GPT4o_Structured(**chat_config)
dataset = ProcessedDataset('/home/zhuangjiaxin/workspace/TensorTSL/TimeLLM/output/test-1-300/UCR', mode='test')

ask_prompt = f'''
<Task>: Here is a time series data. There may be some abnormality in it. I will offer you some background information about the data:
The background information is as follows: {dataset.get_background_info()}
<Target>: Please help me analyze it. If there is no abnormality, you can say "No abnormality".
The output should include some structured information:
- abnormal_index: The abnormality index of the time series. If there is no abnormality, you can say "No abnormality".
- abnormal_type: The abnormality type of the time series, choose from [amplitude, frequency]
- abnormal_description: Make a brief description of the abnormality, why do you think it is abnormal?
'''

def make_message_content(base64_img):
    message_content = [
        {
            "type": "text", 
            "text": ask_prompt,
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{base64_img}"},
        }
    ]
    return message_content

img_path = dataset.get_data_by_index(0)
print(f'image: {img_path}')
base64_img = chatbot.image_encoder_base64(img_path)
message_content = make_message_content(base64_img)
response = chatbot.chat(message_content)
print(response)

