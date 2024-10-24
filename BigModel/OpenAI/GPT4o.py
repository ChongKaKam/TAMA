from openai import OpenAI
from pydantic import BaseModel, Field

from BigModel.Base import BigModelBase

'''
General GPT-4o
'''
class Chat_GPT4o(BigModelBase):
    def __init__(self, max_tokens=1024, temperature=0.1, top_p=0.5):
        super().__init__(max_tokens, temperature, top_p)
        self.load_my_api('openai')
        self.client = OpenAI(api_key=self.api_key)
        self.model_name = 'gpt-4o-2024-08-06'
        self.used_token = {
            'completion_tokens': 0,
            'prompt_tokens': 0,
            'total': 0
        }
        self.tokens_per_minute = 60000

    def chat(self, message:list):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=message,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p
        )
        ans = response.choices[0].message.content
        # last used tokens
        self.last_used_token['completion_tokens'] = response.usage.completion_tokens
        self.last_used_token['prompt_tokens'] = response.usage.prompt_tokens
        self.last_used_token['total_tokens'] = response.usage.total_tokens
        # count the used tokens
        self.used_token['completion_tokens'] += response.usage.completion_tokens
        self.used_token['prompt_tokens'] += response.usage.prompt_tokens
        self.used_token['total_tokens'] += response.usage.total_tokens
        return ans
    
    def text_item(self, content):
        item = {
            "type": "text",
            "text": content
        }
        return item

    def image_item_from_path(self, image_path, detail="high"):
        image_base64 = self.image_encoder_base64(image_path)
        item = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{image_base64}", 
                "detail": detail,
            },
        }
        return item
    
    def image_item_from_base64(self, image_base64, detail="high"):
        item = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{image_base64}", 
                "detail": detail,
            },
        }
        return item
    
'''
GPT-4o with structured output
'''
class Chat_GPT4o_Structured(BigModelBase):
    def __init__(self, max_tokens=1200, temperature=0.1, top_p=0.5):
        super().__init__(max_tokens, temperature, top_p)
        self.load_my_api('openai')
        self.client = OpenAI(api_key=self.api_key)
        self.model_name = 'gpt-4o-2024-08-06'
        self.used_token = {
            'completion_tokens': 0,
            'prompt_tokens': 0,
            'total_tokens': 0
        }
        self.tokens_per_minute = 60000
    def chat(self, message:list, FormatModel:BaseModel):
        response = self.client.beta.chat.completions.parse(
            model=self.model_name,
            messages=message,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            response_format=FormatModel,
        )
        ans = response.choices[0].message.parsed
        # record the last used tokens
        self.last_used_token['completion_tokens'] = response.usage.completion_tokens
        self.last_used_token['prompt_tokens'] = response.usage.prompt_tokens
        self.last_used_token['total_tokens'] = response.usage.total_tokens
        # count the used tokens
        self.used_token['completion_tokens'] += response.usage.completion_tokens
        self.used_token['prompt_tokens'] += response.usage.prompt_tokens
        self.used_token['total_tokens'] += response.usage.total_tokens
        return ans
    
    def text_item(self, content):
        item = {
            "type": "text",
            "text": content
        }
        return item

    def image_item_from_path(self, image_path, detail="high"):
        image_base64 = self.image_encoder_base64(image_path)
        item = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{image_base64}", 
                "detail": detail,
            },
        }
        return item
    
    def image_item_from_base64(self, image_base64, detail="high"):
        item = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{image_base64}", 
                "detail": detail,
            },
        }
        return item