from azure.identity import AzureCliCredential, get_bearer_token_provider
from openai import AzureOpenAI

from BigModel.Base import BigModelBase

class Chat_AzureGPT4o(BigModelBase):
    def __init__(self, max_tokens=1024, temperature=0.1, top_p=0.5):
        super().__init__(max_tokens, temperature, top_p)
        self.tokens_per_minute = 75000
        self.credential = AzureCliCredential()
        self.token_provider = get_bearer_token_provider(
            self.credential,
            "https://cognitiveservices.azure.com/.default"
        )
        self.client = AzureOpenAI(
            azure_endpoint="https://msraopenaieastus.openai.azure.com/",
            azure_ad_token_provider=self.token_provider,
            api_version="2024-07-01-preview",
            max_retries=5,
        )
        self.used_token = {
            'completion_tokens': 0,
            'prompt_tokens': 0,
            'total_tokens': 0
        }
    def chat(self, message:list):
        # clear last used token
        self.last_used_token['completion_tokens'] = 0
        self.last_used_token['prompt_tokens'] = 0
        self.last_used_token['total_tokens'] = 0
        # chat
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=message,
            response_format={"type": "json_object"},
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        ans = response.choices[0].message.content
        # token usage
        self.last_used_token['completion_tokens'] = response.usage.completion_tokens
        self.last_used_token['prompt_tokens'] = response.usage.prompt_tokens
        self.last_used_token['total_tokens'] = response.usage.total_tokens
        # # token count
        self.used_token['completion_tokens'] += response.usage.completion_tokens
        self.used_token['prompt_tokens'] += response.usage.prompt_tokens
        self.used_token['total_tokens'] += response.usage.total_tokens

        return ans
    