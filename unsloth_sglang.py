# Start the server with:
# vrun -P gpu-l4-1 -s svc python3 -m sglang.launch_server     --model-path unsloth/Llama-3.2-1B-Instruct     --host 0.0.0.0 --port 30000

# Install openai via pip install openai
from openai import OpenAI
import json
openai_client = OpenAI(
    base_url = "http://svc:30000/v1",
    api_key = "sk-no-key-required",
)
completion = openai_client.chat.completions.create(
    model = "unsloth/Llama-3.2-1B-Instruct",
    messages = [{"role": "user", "content": "What is 2+2?"},],
)
print(completion.choices[0].message.content)