import requests
import os

COHERE_API_KEY = "4nSkz9GlH60YqLv0NiRwV5F0yYF0TOfxKlDdBiBA"  # 🔁 Replace this with your real Cohere API key

def cohere_chat(prompt):
    url = "https://api.cohere.ai/v1/chat"
    headers = {
        "Authorization": f"Bearer {COHERE_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "message": prompt,
        "model": "command-nightly",  # This model works with chat
        "temperature": 0.7,
        "stream": False
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        return response.json()["text"]
    else:
        return f"⚠️ Error: {response.status_code} — {response.text}"