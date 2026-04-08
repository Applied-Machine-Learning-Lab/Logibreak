from openai import OpenAI
import os
import logging
import time


class LLMCall:
    def __init__(self, model_name, API_key=None, max_tokens=256) -> None:
        super().__init__()
        self.API_key = ""
        self.model_name = model_name
        self.max_tokens = max_tokens  # ★★ 新增：最大生成 token 限制 默认 256

        # ====== API Base + Key 按模型类型自动选择 ======
        if self.model_name.lower().startswith("deepseek") and '7b' in self.model_name.lower():
            API_base = ""
        elif self.model_name.lower().startswith("deepseek") and '32b' in self.model_name.lower():
            API_base = ""
            self.API_key = ""
        elif self.model_name.lower().startswith("qwen") and '7b' in self.model_name.lower():
            API_base = ""
        elif self.model_name.lower().startswith("llama3") and '8b' in self.model_name.lower():
            API_base = ""
        elif self.model_name.lower().startswith('llama3') and '70b' in self.model_name.lower():
            API_base = ""
            self.API_key = ""
        elif self.model_name.lower().startswith('gpt'):
            API_base = ""
            self.API_key = ""
        elif self.model_name == 'r1-api':
            API_base = ""
            self.API_key = ""
        elif self.model_name == 'v3-api':
            API_base = ""
            self.API_key = ""

        self.client = OpenAI(api_key=self.API_key, base_url=API_base)

    # ====== 对外接口 ======
    def call(self, messages, seed=0, temperature=1.0):
        response = None

        while response is None:
            try:
                # ====== Deepseek 7B ======
                if self.model_name.lower().startswith("deepseek") and '7b' in self.model_name.lower():
                    response = self.client.chat.completions.create(
                        model="/data/share_weight/DeepSeek-R1-Distill-Qwen-7B",
                        messages=messages,
                        temperature=temperature,
                        seed=seed,
                        max_tokens=self.max_tokens  # ★★ 新增
                    )

                # ====== Deepseek 32B ======
                elif self.model_name.lower().startswith("deepseek") and '32b' in self.model_name.lower():
                    response = self.client.chat.completions.create(
                        model="deepseek-r1-distill-qwen-32b-250120",
                        messages=messages,
                        temperature=temperature,
                        seed=seed,
                        max_tokens=self.max_tokens
                    )

                # ====== Qwen ======
                elif self.model_name.lower().startswith("qwen") and '7b' in self.model_name.lower():
                    response = self.client.chat.completions.create(
                        model="/data/share_weight/Qwen2.5-7B-Instruct",
                        messages=messages,
                        temperature=temperature,
                        seed=seed,
                        max_tokens=self.max_tokens
                    )

                # ====== Llama3 8B ======
                elif self.model_name.lower().startswith("llama3.1") and '8b' in self.model_name.lower():
                    response = self.client.chat.completions.create(
                        model="/data/share_weight/Llama-3.1-8B-Instruct",
                        messages=messages,
                        temperature=temperature,
                        seed=seed,
                        max_tokens=self.max_tokens
                    )

                elif self.model_name.lower().startswith("llama3.2") and '11b' in self.model_name.lower():
                    response = self.client.chat.completions.create(
                        model="/data/share_weight/Llama-3.2-11B-Vision-Instruct",
                        messages=messages,
                        temperature=temperature,
                        seed=seed,
                        max_tokens=self.max_tokens
                    )

                elif self.model_name.lower().startswith("llama3") and '8b' in self.model_name.lower():
                    response = self.client.chat.completions.create(
                        model="/data/share_weight/Meta-Llama-3-8B-Instruct",
                        messages=messages,
                        temperature=temperature,
                        seed=seed,
                        max_tokens=self.max_tokens
                    )

                # ====== Llama3 70B ======
                elif self.model_name.lower().startswith('llama3') and '70b' in self.model_name.lower():
                    response = self.client.chat.completions.create(
                        model="llama3-70b",
                        messages=messages,
                        temperature=temperature,
                        seed=seed,
                        max_tokens=self.max_tokens
                    )

                # ====== o3-mini ======
                elif 'o3-mini' in self.model_name.lower():
                    response = self.client.chat.completions.create(
                        model='o3-mini',
                        messages=messages,
                        temperature=temperature,
                        max_tokens=self.max_tokens
                    )

                # ====== GPT 系列 ======
                elif 'gpt' in self.model_name.lower():
                    if '4o-mini' in self.model_name:
                        response = self.client.chat.completions.create(
                            model='gpt-4o-mini',
                            messages=messages,
                            temperature=temperature,
                            max_tokens=self.max_tokens
                        )
                    elif 'gpt-4' in self.model_name:
                        response = self.client.chat.completions.create(
                            model='gpt-4-turbo',
                            messages=messages,
                            temperature=temperature,
                            max_tokens=self.max_tokens
                        )
                    elif 'gpt-3.5' in self.model_name:
                        response = self.client.chat.completions.create(
                            model='gpt-3.5-turbo',
                            messages=messages,
                            temperature=temperature,
                            max_tokens=self.max_tokens
                        )

                # ====== r1-api ======
                elif self.model_name == 'r1-api':
                    response = self.client.chat.completions.create(
                        model='deepseek-reasoner',
                        messages=messages,
                        temperature=temperature,
                        max_tokens=self.max_tokens
                    )

                # ====== v3-api ======
                elif self.model_name == 'v3-api':
                    response = self.client.chat.completions.create(
                        model='deepseek-v3-250324',
                        messages=messages,
                        temperature=temperature,
                        max_tokens=self.max_tokens
                    )

                else:
                    return "Unsupported model name."

            except Exception as e:
                logging.warning(e)
                return "Unable to reach the response due to some reasons."

        return response.choices[0].message.content
