import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN,
)

# Test different Mixtral provider suffixes
providers = [
    "mistralai/Mixtral-8x7B-Instruct-v0.1:featherless-ai",
    "mistralai/Mixtral-8x7B-Instruct-v0.1:vllm",
    "mistralai/Mixtral-8x7B-Instruct-v0.1:hyperbolic",
    "mistralai/Mixtral-8x7B-Instruct-v0.1:together",
]

for model in providers:
    print(f"Testing: {model}")
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hi"},
            ],
            max_tokens=50,
            temperature=0.7,
        )
        print(f"  SUCCESS")
    except Exception as e:
        error_msg = str(e)
        if "400" in error_msg:
            print(f"  FAILED: {error_msg.split('message')[1][:100] if 'message' in error_msg else error_msg[:100]}")
        else:
            print(f"  ERROR: {error_msg[:100]}")
    print()
