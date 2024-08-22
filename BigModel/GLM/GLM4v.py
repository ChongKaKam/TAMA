from zhipuai import ZhipuAI

from BigModel.Base import BigModelBase

'''
General GLM-4v
'''
class Chat_GLM4v(BigModelBase):
    def __init__(self, max_tokens=1024, temperature=0.1, top_p=0.5):
        super().__init__(max_tokens, temperature, top_p)
        self.load_my_api('chatglm')
        self.client = ZhipuAI(api_key=self.api_key)
        self.model_name = 'glm-4v'
        # self.tokens_per_minute = 60000

    def chat(self, messages:list):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p
        )
        ans = response.choices[0].message.content
        # TODO: NoImplementationError
        # last used tokens
        self.last_used_token = response.usage.completion_tokens
        # count the used tokens
        self.used_token['completion_tokens'] += response.usage.completion_tokens
        return ans