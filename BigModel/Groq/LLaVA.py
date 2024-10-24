from groq import Groq

from BigModel.Base import BigModelBase

class Chat_LLaVA_v1_5(BigModelBase):
    def __init__(self, max_tokens=1024, temperature=0.1, top_p=0.5):
        super().__init__(max_tokens, temperature, top_p)
        self.load_my_api('groq')
        self.client = Groq(api_key=self.api_key)
        self.model_name = "llava-v1.5-7b-4096-preview"
        self.used_token = {
            'completion_tokens': 0,
            'prompt_tokens': 0,
            'total_tokens': 0
        }
        self.tokens_per_minute = 60000

    def chat(self, message:list):
        # clear last used token
        self.last_used_token['completion_tokens'] = 0
        self.last_used_token['prompt_tokens'] = 0
        self.last_used_token['total_tokens'] = 0
        # chat
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=message,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p
        )
        ans = response.choices[0].message.content
        # print(response);exit()
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
            },
        }
        return item
    
    def image_item_from_base64(self, image_base64, detail="high"):
        item = {
            "type": "image",
            "image_url": {
                "url": f"data:image/png;base64,{image_base64}", 
            },
        }
        return item