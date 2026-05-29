"""Test environment variable loading."""
import os
from dotenv import load_dotenv

load_dotenv()

print("Environment variables:")
print(f"AZURE_OPENAI_KEY: {os.getenv('AZURE_OPENAI_KEY')}")
print(f"AZURE_OPENAI_ENDPOINT: {os.getenv('AZURE_OPENAI_ENDPOINT')}")
print(f"AZURE_OPENAI_MODEL_ID: {os.getenv('AZURE_OPENAI_MODEL_ID')}")
print(f"AZURE_OPENAI_API_VERSION: {os.getenv('AZURE_OPENAI_API_VERSION', '2024-12-01-preview')}")

# Now test OpenAI import
from openai import AzureOpenAI

client = AzureOpenAI(
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
)

print("\nTesting API call...")
response = client.chat.completions.create(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello!"},
    ],
    max_completion_tokens=50,
    model=os.getenv("AZURE_OPENAI_MODEL_ID")
)

print(f"Response: {response.choices[0].message.content}")
print("\n✓ All tests passed!")
