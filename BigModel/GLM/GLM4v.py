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

    def chat(self, content:list):
        messages=[
            {
                "role": "user",
                "content": content,
            }
        ]
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p
        )
        ans = response.choices[0].message.content
        return ans