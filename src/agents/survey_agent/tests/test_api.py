from openai import OpenAI
 
api_key = "sk-UCfwgl63Xg27JF8W33D746F3B80d4862979c82A51951485f"
api_base = "https://api.xi-ai.cn/v1"
client = OpenAI(api_key=api_key, base_url=api_base)
 
completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ]
)
 
print(completion.choices[0].message)