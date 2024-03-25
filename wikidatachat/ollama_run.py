import os
import ollama

OLLAMA_MODEL_NAME = os.environ.get("OLLAMA_MODEL_NAME")
OLLAMA_URL = os.environ.get("OLLAMA_URL")
OLLAMA_GENERATE_URL = f"{OLLAMA_URL}/api/generate"

print(f"Enter prompt to {OLLAMA_MODEL_NAME}:\n")
print("""Examples:
- Why is the sky blue?
- Build a VueJS search frontend.
"""
)
prompt = input('Prompt: ')

# Basic Usage
response = ollama.chat(
    model=OLLAMA_MODEL_NAME,
    messages=[{
        'role': 'user',
        'content': prompt,
    }]
)

print(response['message']['content'])

# Streaming responses
stream = ollama.chat(
    model=OLLAMA_MODEL_NAME,
    messages=[{'role': 'user', 'content': prompt}],
    stream=True,
)

for chunk in stream:
  print(chunk['message']['content'], end='', flush=True)


# Ollama Generator API
result_gen = ollama.generate(
    model=OLLAMA_MODEL_NAME,
    prompt=prompt
)

print(result_gen['response'])