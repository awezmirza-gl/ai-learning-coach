import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN,
)

# Test various open models that should be available
models = [
    "meta-llama/Llama-2-7b-chat:featherless-ai",
    "meta-llama/Llama-3-8B-Instruct:featherless-ai",
    "mistralai/Mistral-7B-Instruct-v0.2:featherless-ai",
    "meta-llama/Llama-2-70b-chat:featherless-ai",
    "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO:featherless-ai",
    "NousResearch/Nous-Hermes-2-Mistral-7B-DPO:featherless-ai",
]

for model in models:
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
        print(f"  SUCCESS - Response OK")
    except Exception as e:
        error_msg = str(e)
        if "not supported" in error_msg.lower():
            print(f"  NOT AVAILABLE")
        elif "400" in error_msg:
            print(f"  BAD REQUEST")
        else:
            print(f"  ERROR: {error_msg[:80]}")
    print()
