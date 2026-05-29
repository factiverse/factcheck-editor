"""Test OpenAIUtils class."""
import sys
sys.path.insert(0, '/home/vsetty/repos/factcheck-editor')

from src.llm_utils.openai_utils import OpenAIUtils

print("Testing OpenAIUtils class...")
utils = OpenAIUtils()
print(f"Engine/Model: {utils._engine}")
print(f"Client: {utils._client}")

# Test a simple completion
print("\nTesting completion...")
response = utils._client.chat.completions.create(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello!"},
    ],
    max_completion_tokens=50,
    model=utils._engine
)

print(f"Response: {response.choices[0].message.content}")
print("\n✓ OpenAIUtils works!")
