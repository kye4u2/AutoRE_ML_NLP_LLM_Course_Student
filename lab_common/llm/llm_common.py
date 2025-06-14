import os
from dataclasses import dataclass
from typing import Optional

import tiktoken
from omegaconf import OmegaConf

from lab_common.common import ROOT_PROJECT_FOLDER_PATH


MAX_TOKEN_LENGTH = 12000



@dataclass
class LLMClientConfig:
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    system_prompt: str = "You are a helpful assistant."
    temperature: float = 0.7
    max_tokens: int = 256
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

@dataclass
class LLMContext:
    """
    Data class to hold the response and token information from the Large Language Model (LLM).

    Attributes:
        response (str): The response text from the LLM.
        completion_tokens (int): The number of tokens in the completion.
        prompt_tokens (int): The number of tokens in the prompt.
        total_tokens (int): The total number of tokens used.
    """
    response: str
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int

    def __str__(self):
        top_bottom_border = "=" * 100
        middle_border = "-" * 100
        return (f"\n{top_bottom_border}\n"
                f"{'RESPONSE:'.center(100)}\n"
                f"{middle_border}\n"
                f"{self.response.center(100)}\n"
                f"{middle_border}\n\n"
                f"Completion tokens: {self.completion_tokens}\n"
                f"Prompt tokens: {self.prompt_tokens}\n"
                f"Total tokens: {self.total_tokens}\n"
                f"{top_bottom_border}\n")

def get_llm_client_config() -> LLMClientConfig:
    """
    Load the LLM client configuration from a YAML file.
    """
    config_path = os.path.join(ROOT_PROJECT_FOLDER_PATH, "lab_common","llm", "llm_client_config.yaml")

    cfg = OmegaConf.load(config_path)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    config = LLMClientConfig(**cfg[LLMClientConfig.__name__])
    return config


def num_tokens_from_string(string: str) -> int:

    model_name = get_llm_client_config().model
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens
