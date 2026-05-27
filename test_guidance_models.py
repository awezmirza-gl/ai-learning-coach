import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN,
)

# Test models for the guidance role
models = [
    "NousResearch/Nous-Hermes-2-Mistral-7B-DPO:featherless-ai",
    "mistralai/Mistral-7B-Instruct-v0.2:featherless-ai",
    "HuggingFaceH4/zephyr-7b-beta:featherless-ai",
]

for model in models:
    print(f"Testing: {model}")
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a friendly programming tutor."},
                {"role": "user", "content": "Explain variables and loops."},
            ],
            max_tokens=300,
            temperature=0.7,
        )
        response = completion.choices[0].message.content.strip()
        print(f"  SUCCESS - {len(response)} chars")
        if len(response) < 200:
            print(f"  Response: {response[:150]}")
    except Exception as e:
        error_msg = str(e)
        print(f"  FAILED: {error_msg[:150]}")
    print()
