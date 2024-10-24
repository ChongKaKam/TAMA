import sensenova
import yaml
from BigModel.Base import BigModelBase, DEFAULT_API_PATH

'''
SenseChat-Vision
'''
class Chat_SenseChatVision(BigModelBase):
    def __init__(self, max_tokens=1024, temperature=0.1, top_p=0.5):
        super().__init__(max_tokens, temperature, top_p)
        self.load_my_api('sensenova')
        sensenova.access_key_id = self.api_id
        sensenova.secret_access_key = self.api_key
        self.client = sensenova.ChatCompletion
        self.model_name = 'SenseChat-Vision'
        self.used_token = {
            'completion_tokens': 0,
            'prompt_tokens': 0,
            'total_tokens': 0
        }

    def load_my_api(self, name):
        yaml_info = yaml.safe_load(open(DEFAULT_API_PATH))[name]['api_key']
        self.api_key = yaml_info[name]['api_key']
        self.api_id = yaml_info[name]['api_id']

    
        