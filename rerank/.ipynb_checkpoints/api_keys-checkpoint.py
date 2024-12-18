import os
from typing import Dict

from dotenv import load_dotenv


def get_openai_api_key(resource='baidu', model_name='gpt-4o-2024-08-06') -> str:
    load_dotenv(dotenv_path=f".env.local")
    if resource == 'baidu':
        if model_name == 'gpt-4o-mini':
            key = os.getenv("BAIDU_API_KEY_GPT4omini")
        elif model_name == 'gpt-4o-2024-08-06':
            key = os.getenv("BAIDU_API_KEY_GPT4o")
    else:
        key = os.getenv("OPEN_AI_API_KEY")
    return key 

def get_azure_openai_args() -> Dict[str, str]:
    load_dotenv(dotenv_path=f".env.local")
    azure_args = {
        "api_type": "azure",
        "api_version": os.getenv("AZURE_OPENAI_API_VERSION"),
        "api_base": os.getenv("AZURE_OPENAI_API_BASE"),
    }

    # Sanity check
    assert all(
        list(azure_args.values())
    ), "Ensure that `AZURE_OPENAI_API_BASE`, `AZURE_OPENAI_API_VERSION` are set"
    return azure_args
