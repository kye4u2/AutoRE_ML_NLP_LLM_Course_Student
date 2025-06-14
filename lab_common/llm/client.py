import logging
import os
from typing import Union, List, Dict, Optional


from openai import OpenAI

from lab_common.common import ROOT_PROJECT_FOLDER_PATH
from lab_common.llm.llm_common import LLMClientConfig, get_llm_client_config, LLMContext

logging.getLogger("httpx").setLevel(logging.WARNING)


class LLMClient:
    _instance = None  # Class-level variable for singleton instance

    def __new__(cls, *args, **kwargs):
        """
        Ensure only one instance of LLMClient exists (singleton).
        """
        if cls._instance is None:
            cls._instance = super(LLMClient, cls).__new__(cls)
        return cls._instance

    def __init__(self, api_key: Optional[str] = None, config: Optional[LLMClientConfig] = None):
        """
        Initialize the LLMClient.
        """
        if hasattr(self, "_initialized") and self._initialized:
            return  # Avoid re-initialization on repeated calls

        self.config : LLMClientConfig = config if config else get_llm_client_config()

        pass

        api_key = self.config.api_key

        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key is None:
                raise ValueError("OPENAI_API_KEY environment variable not set and no api_key provided.")

        self.client = OpenAI(api_key=api_key)
        self.config = config if config else get_llm_client_config()

        self._initialized = True

    def complete(
            self,
            prompt: Union[str, List[Dict[str, str]]],
            system_prompt: Optional[str] = None,
    ) -> LLMContext:
        """
        Generate a response from OpenAI's chat model using the loaded configuration.
        """
        if isinstance(prompt, str):

            if system_prompt is not None:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            else:
                messages = [
                    {"role": "system", "content": self.config.system_prompt},
                    {"role": "user", "content": prompt}
                ]
        elif isinstance(prompt, list):
            messages = prompt
        else:
            raise TypeError("Prompt must be a string or a list of message dictionaries.")

        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            frequency_penalty=self.config.frequency_penalty,
            presence_penalty=self.config.presence_penalty
        )

        choice = response.choices[0].message.content.strip()
        usage = response.usage  # usage object is available from OpenAI responses

        return LLMContext(
            response=choice,
            completion_tokens=usage.completion_tokens,
            prompt_tokens=usage.prompt_tokens,
            total_tokens=usage.total_tokens
        )



# Create a global client instance once
_llm_client_instance: Optional[LLMClient] = None


def llm_completion(
        prompt: Union[str, List[Dict[str, str]]],
        system_prompt: Optional[str] = None,
) -> LLMContext:
    """
    External function to perform a completion using a global LLMClient instance.
    """
    global _llm_client_instance
    if _llm_client_instance is None:
        _llm_client_instance = LLMClient()

    return _llm_client_instance.complete(prompt, system_prompt)


import argparse

def main():
    parser = argparse.ArgumentParser(description="LLM Completion Script")
    parser.add_argument("--prompt", type=str, help="Prompt string to send to the LLM")
    args = parser.parse_args()

    if args.prompt:
        # If a prompt is provided, use it directly
        response = llm_completion(args.prompt)
        print("Assistant:", response)
    else:
        # Example 1: Using a simple string prompt
        response = llm_completion("What is the capital of France?")
        print("Assistant:", response)

        # Example 2: Using a list of message dictionaries
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Can you explain the theory of relativity?"}
        ]
        response = llm_completion(messages)
        print("Assistant:", response)



if __name__ == "__main__":
    main()
