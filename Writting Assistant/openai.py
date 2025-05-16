import os
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# FastAPI app setup
app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request body models
class ContentInput(BaseModel):
    title: str
    description: str

@app.post("/generate_title")
async def generate_title(data: ContentInput):
    prompt = f"""
You are an assistant for writing short news headlines.
Given the following news article title and description, suggest the top 3 catchy, concise, and relevant titles for a news article.

Title: {data.title}
Description: {data.description}

Return the 3 titles as a numbered list.
"""
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=100
        )

        content = response.choices[0].message.content.strip()
        titles = re.findall(r'\d+\.\s*"?(.+?)"?\s*(?=\n|$)', content)
        return {"suggested_titles": titles}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate titles: {e}")


@app.post("/generate_description")
async def generate_description(data: ContentInput):
    prompt = f"""
You are a journalist assistant.
Given the following title and short description, rephrase and enrich the description in a clear, engaging way, suitable for a news article summary.

Title: {data.title}
Description: {data.description}

Return:
Rephrased Description:
"""
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=150
        )

        content = response.choices[0].message.content.strip()
        rephrased_match = re.search(r"Rephrased Description:\s*(.*)", content, re.IGNORECASE | re.DOTALL)
        rephrased = rephrased_match.group(1).strip() if rephrased_match else content

        return {"rephrased_description": rephrased}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to rephrase description: {e}")


