from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import re

app = FastAPI()

class ArticleInput(BaseModel):
    title: str
    description: str

def call_ollama(prompt: str, model: str = "mistral"):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            }
        )
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Ollama generation failed")
        return response.json()["response"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/suggested_titles")
def suggest_titles(article: ArticleInput):
    prompt = f"""You are a headline editor.

Given the following news title and description:

Title: {article.title}
Description: {article.description}

Suggest 3 improved and engaging titles. Respond only as:
1. Title One
2. Title Two
3. Title Three
"""

    response_text = call_ollama(prompt)
    titles = re.findall(r"\d+\.\s(.+)", response_text)
    return {"suggested_titles": titles}

@app.post("/rewrite_description")
def rewrite_description(article: ArticleInput):
    prompt = f"""You are a professional news editor.

Given the following article:

Title: {article.title}
Description: {article.description}

Rewrite the description to improve clarity, fix grammar, and enhance readability.
Respond only with the improved description text.
"""

    response_text = call_ollama(prompt)
    return {"rewritten_description": response_text.strip()}
