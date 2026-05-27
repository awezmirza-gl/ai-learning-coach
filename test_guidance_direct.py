import os
import sys
sys.path.insert(0, '/c/Awez/FDE/brownfield-challenge/ai-learning-coach')

from app import call_model, generate_guidance

# Test the call_model function directly
result = call_model(
    "mistralai/Mistral-7B-Instruct-v0.2:featherless-ai",
    "You are a friendly tutor.",
    "Explain variables and loops.",
    max_tokens=300
)

print(f"Result length: {len(result)}")
print(f"First 100 chars: {result[:100]}")
print(f"Contains [Model Error]: {'[Model Error]' in result}")
print(f"Contains <!DOCTYPE: {'<!DOCTYPE' in result}")
print()

# Now test generate_guidance
guidance = generate_guidance("beginner", "I understand variables and loops")
print(f"\nGuidance length: {len(guidance)}")
print(f"First 100 chars: {guidance[:100]}")
print(f"Contains [Model Error]: {'[Model Error]' in guidance}")
print(f"Contains <!DOCTYPE: {'<!DOCTYPE' in guidance}")
