# ✍️ Writing Assistant for LiveBreak

This microservice provides **AI-powered writing assistance** tailored for journalists, content creators, and independent news contributors. It supports two powerful backend models:

1. **OpenAI GPT-3.5 API** — for scalable cloud-based generation  
2. **Locally hosted Ollama (Mistral model)** — for offline, private inference

---

## 🚀 Features

- **🎯 Suggested Titles Generator**  
  Generates 3 concise, engaging, and relevant headline options based on the news content provided.

- **📝 Description Rewriter**  
  Rewrites brief news descriptions into professionally styled journalistic summaries.

---

## 🧠 Approaches

### 🔹 1. OpenAI API (`openai.py`)

Uses `gpt-3.5-turbo` for:
- Rewriting news descriptions in a formal news tone.
- Suggesting creative and relevant headlines.

### 🔹 2. Ollama Mistral Backend (`ollama.py`)

Uses **Mistral 7B model** running via [Ollama](https://ollama.com/) on your local machine. Offers:
- Privacy-preserving, low-latency inference.
- No API cost or dependency on internet.

---

## 🧪 Endpoints

| Endpoint | Method | Description |
|---------|--------|-------------|
| `/generate_title` | `POST` | Returns 3 suggested titles. |
| `/generate_description` | `POST` | Returns rewritten version of the original description. |

### 📥 Request Format (both endpoints)

```json
{
  "title": "Original title of the news article",
  "description": "Short user-written description"
}
```

### 📤 Example Response: `/generate_title`

```json
{
  "suggested_titles": [
    "City Council Approves New Development Plan",
    "Green Light Given to Downtown Expansion",
    "Officials Back New Infrastructure Project"
  ]
}
```

### 📤 Example Response: `/generate_description`

```json
{
  "rewritten_description": "The city council has approved a new infrastructure development plan aimed at modernizing the downtown area. The initiative is expected to boost local business and improve public services."
}
```

---

## ⚙️ Setup and Run Instructions

### 1. 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. 🔐 Setup Environment Variables

Create a `.env` file and add your OpenAI API key:

```env
OPENAI_API_KEY=your_openai_key_here
```

### 3. ▶️ Run the FastAPI Server

```bash
uvicorn main:app --reload --port 8000
```

- Access Swagger UI: `http://localhost:8000/docs`
- Test endpoints directly from browser or Postman

---

## 🧠 Using Locally Hosted Ollama (Mistral)

### 🔧 Prerequisites

- [Install Ollama](https://ollama.com/download) on your machine
- Pull the model:

```bash
ollama pull mistral
```

- Start Ollama service and ensure it's running

Update `.env` if using a different base URL or model:
```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral
```

---

## 📂 Project Structure

```
.
├── main.py               # FastAPI entry point
├── openai.py             # Logic using OpenAI API
├── ollama.py             # Logic using local Mistral model
├── utils.py              # Helper methods
├── requirements.txt
└── .env
```

---

## ✅ Testing Locally (Example using `curl`)

### Generate Suggested Titles

```bash
curl -X POST http://localhost:8000/generate_title \
-H "Content-Type: application/json" \
-d '{"title":"City Plans New Park", "description":"A new green space is planned in the downtown area to serve local communities."}'
```

### Rewrite Description

```bash
curl -X POST http://localhost:8000/generate_description \
-H "Content-Type: application/json" \
-d '{"title":"City Plans New Park", "description":"A new green space is planned in the downtown area to serve local communities."}'
```

---

## 👥 Contributing

We welcome contributions from the community!  
Feel free to open an issue or submit a pull request.

---

## 📄 License

MIT License — Free to use, modify, and distribute.
