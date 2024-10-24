import yaml
import os
import base64
import io
from PIL import Image
import time
import math

DEFAULT_API_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'api_keys.yaml')
'''
Base Model for BigModel
'''
class BigModelBase:
    def __init__(self, max_tokens=1024, temperature=0.1, top_p=0.5):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.used_token = {}
        self.last_used_token = {}
        self.tokens_per_minute = math.inf
        self.image_rotation_angle = 0

    def load_my_api(self, name):
        self.api_key = yaml.safe_load(open(DEFAULT_API_PATH))[name]['api_key']

    def set_image_rotation(self, angle:float=0):
        self.image_rotation_angle = angle

    def image_encoder_base64(self, image_path):
        # print(image_path)
        with Image.open(image_path) as img:
            buffer = io.BytesIO()
            img.rotate(self.image_rotation_angle, expand=True).save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def chat(self, message):
        raise NotImplementedError

    # chat, if fail, try again
    def chat_retry(self, message, max_try=3, wait_time=0.5):
        for t in range(max_try):
            try:
                response = self.chat(message)
                return response
            except Exception as e:
                print(f'Error: {e}')
                time.sleep(wait_time)
                print(f'Try again {t}/{max_try}')

    def get_used_token(self):
        return self.used_token
    
    def get_last_used_token(self):
        return self.last_used_token
    
    def get_tokens_per_minute(self):
        return self.tokens_per_minute
    
    def text_item(self, content):
        pass

    def image_item_from_path(self, image_path):
        pass

    def image_item_from_base64(self, image_base64):
        pass