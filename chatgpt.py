import time
import openai
from colorama import Fore
from config import cfg

# Set OpenRouter API key and base URL
openai.api_key = cfg.openai_api_key  # Store your OpenRouter API key in config.py
openai.api_base = "https://openrouter.ai/api/v1"  # Change base URL to OpenRouter

def create_chat_completion(messages, model=cfg.fast_llm_model, temperature=cfg.temperature, max_tokens=None) -> str:
    """Create a chat completion using OpenRouter API (compatible with OpenAI)"""
    response = None
    num_retries = 5

    for attempt in range(num_retries):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            break
        except openai.error.RateLimitError:
            if cfg.debug_mode:
                print(Fore.RED + "Error:", "API Rate Limit Reached. Waiting 20 seconds..." + Fore.RESET)
            time.sleep(20)
        except openai.error.APIError as e:
            if e.http_status == 502:
                if cfg.debug_mode:
                    print(Fore.RED + "Error:", "API Bad gateway. Waiting 20 seconds..." + Fore.RESET)
                time.sleep(20)
            else:
                raise
        except openai.error.InvalidRequestError as e:
            print(Fore.RED + "Invalid Request Error:", e + Fore.RESET)
            raise

    if response is None:
        raise RuntimeError("Failed to get response after 5 retries")

    return response["choices"][0]["message"]["content"]
