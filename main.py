from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware


import requests,os
from dotenv import load_dotenv
load_dotenv()


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


SENTIMENT_MODEL = "https://router.huggingface.co/hf-inference/models/j-hartmann/emotion-english-distilroberta-base"
TOPIC_MODEL = "https://router.huggingface.co/hf-inference/models/facebook/bart-large-mnli"

#
HF_TOKEN = os.getenv("HF_TOKEN")

@app.post("/analyze")
async def analyze(request: Request):
    data = await request.json()
    text = data.get("text", "")

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }

    # --- Emotion / Sentiment ---
    sent_payload = {"inputs": text}
    sent_response = requests.post(SENTIMENT_MODEL, headers=headers, json=sent_payload)

    try:
        sentiments = sent_response.json()
    except Exception:
        return {"error": "Invalid response from sentiment model"}

    # handle cases where huggingface returns error dict
    if isinstance(sentiments, dict) and "error" in sentiments:
        return {"error": f"Sentiment API error: {sentiments['error']}"}

    # handle nested list case
    if isinstance(sentiments, list) and len(sentiments) > 0:
        if isinstance(sentiments[0], list):
            sentiments = sentiments[0]
    else:
        return {"error": "Unexpected sentiment format", "raw_response": sentiments}

    # ensure dict structure
    if not isinstance(sentiments[0], dict) or "score" not in sentiments[0]:
        return {"error": "Malformed sentiment response", "raw_response": sentiments}

    top_sent = max(sentiments, key=lambda x: x["score"])
    emotion = top_sent["label"].lower()

    # --- Topic Classification ---
    candidate_labels = [
        "politics", "technology", "sports", "finance", "entertainment",
        "education", "science", "relationships", "crime", "mental health",
        "insult", "racism", "humor", "motivation", "violence", "social issues",
        "religion", "personal life", "news"
    ]

    topic_payload = {
        "inputs": text,
        "parameters": {"candidate_labels": candidate_labels}
    }

    topic_response = requests.post(TOPIC_MODEL, headers=headers, json=topic_payload)

    try:
        topic_data = topic_response.json()
    except Exception:
        return {"error": "Invalid response from topic model"}

    if isinstance(topic_data, dict) and "error" in topic_data:
        return {"error": f"Topic API error: {topic_data['error']}"}

    topics = []
    if "labels" in topic_data and "scores" in topic_data:
        sorted_topics = sorted(
            zip(topic_data["labels"], topic_data["scores"]),
            key=lambda x: x[1],
            reverse=True
        )
        topics = [label for label, _ in sorted_topics[:3]]

    return {
        "emotion": {"sentiment": emotion},
        "topic": {"labels": topics},
        "raw_sentiment": sentiments,
        "raw_topic": topic_data
    }
