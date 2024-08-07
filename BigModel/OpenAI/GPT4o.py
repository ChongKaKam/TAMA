import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.pydantic_v1 import BaseModel, Field

from BigModel.Base import BigModelBase

'''
Genernal GPT-4o
'''
class Chat_GPT4o(BigModelBase):
    def __init__(self, max_tokens=1024, temperature=0.1, top_p=0.5):
        super().__init__(max_tokens, temperature, top_p)
        self.load_my_api('openai')
        os.environ['OPENAI_API_KEY'] = self.api_key
        self.model = ChatOpenAI(model='gpt-4o-2024-08-06', max_tokens=self.max_tokens, temperature=self.temperature, top_p=self.top_p)

    def chat(self, content:list):
        message = HumanMessage(content=content)
        response = self.model.invoke([message])
        return response
'''
GPT-4o with structured output
'''
class Chat_GPT4o_Structured(BigModelBase):
    '''
    Structured output for GPT-4o
    '''
    class StructuredModel(BaseModel):
        trend: str = Field(description="The trend of the time series")
        seasonality: str = Field(description="The seasonality of the time series")
        abnormal: str = Field(description="The abnormality of the time series")

    def __init__(self, max_tokens=1200, temperature=0.1, top_p=0.5):
        super().__init__(max_tokens, temperature, top_p)
        self.load_my_api('openai')
        os.environ['OPENAI_API_KEY'] = self.api_key
        general_model = ChatOpenAI(model='gpt-4o-2024-08-06', max_tokens=self.max_tokens, temperature=self.temperature, top_p=self.top_p)
        self.model = general_model.with_structured_output(self.StructuredModel)

    def chat(self, content:list):
        message = HumanMessage(content=content)
        response = self.model.invoke([message])
        return response

