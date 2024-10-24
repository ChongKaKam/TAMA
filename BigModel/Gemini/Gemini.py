import google.generativeai as genai

from BigModel.Base import BigModelBase

'''
Gemini
'''
class Chat_Gemini_1_5_pro(BigModelBase):
    def __init__(self, max_tokens=1024, temperature=0.1, top_p=0.5):
        super().__init__(max_tokens, temperature, top_p)
        self.load_my_api('gemini')
        genai.configure(api_key=self.api_key) 
        self.model_name = 'gemini-1.5-pro'
        self.client = genai.GenerativeAI(model_name=self.model_name)
        self.used_token = {
            'completion_tokens': 0,
            'prompt_tokens': 0,
            'total_tokens': 0
        }
        self.tokens_per_minute = 28000
    def chat(self, message):
        response = self.client.chat(message)
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
    