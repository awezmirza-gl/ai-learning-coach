import os
import json
from openai import OpenAI
from dotenv import load_dotenv
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

HF_TOKEN = os.getenv("HF_TOKEN")
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN,
)

# Test different models
models_to_test = [
    "mistralai/Mistral-7B-Instruct-v0.2:featherless-ai",  # Current analysis/guidance model
    "mistralai/Mixtral-8x7B-Instruct-v0.1:together-ai",   # Judge model
    "meta-llama/Llama-2-7b-chat-hf",
    "HuggingFaceH4/zephyr-7b-beta",
    "mistralai/Mistral-7B-Instruct-v0.2",  # Without provider suffix
]

for model in models_to_test:
    print("\n" + "="*60)
    print(f"Testing: {model}")
    print("="*60)
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say hello briefly."},
            ],
            max_tokens=100,
            temperature=0.7,
        )
        response = completion.choices[0].message.content.strip()
        print(f"SUCCESS: {response[:100]}")
    except Exception as e:
        print(f"FAILED: {str(e)[:200]}")
