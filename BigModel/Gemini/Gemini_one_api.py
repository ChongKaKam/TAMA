from openai import OpenAI

from BigModel.Base import BigModelBase

'''
Gemini -- OneAPI
'''
class Chat_Gemini_1_5_pro(BigModelBase):
    def __init__(self, max_tokens=1024, temperature=0.1, top_p=0.5):
        super().__init__(max_tokens, temperature, top_p)
        self.load_my_api('oneapi')
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="http://xiao-zy19.top:13010/v1",
        )
        self.model_name = 'gemini-1.5-pro'
        self.used_token = {
            'completion_tokens': 0,
            'prompt_tokens': 0,
            'total_tokens': 0
        }
        self.tokens_per_minute = 28000
    def chat(self, message):
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
Gemini-1.5-flash
'''
class Chat_Gemini_1_5_flash(Chat_Gemini_1_5_pro):
    def __init__(self, max_tokens=1024, temperature=0.1, top_p=0.5):
        super().__init__(max_tokens, temperature, top_p)
        self.model_name = 'gemini-1.5-flash'
        self.tokens_per_minute = 28000