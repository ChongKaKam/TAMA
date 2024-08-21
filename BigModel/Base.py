import yaml
import os
import base64
import io
from PIL import Image

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

    def load_my_api(self, name):
        self.api_key = yaml.safe_load(open(DEFAULT_API_PATH))[name]['api_key']

    def image_encoder_base64(self, image_path):
        with Image.open(image_path) as img:
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def chat(self, message):
        raise NotImplementedError

    # chat, if fail, try again
    def chat_retry(self, message, max_try=5):
        for t in range(max_try):
            try:
                response = self.chat(message)
                return response
            except Exception as e:
                print(f'Error: {e}')
                print(f'Try again {t}/{max_try}')

    def get_used_token(self):
        return self.used_token