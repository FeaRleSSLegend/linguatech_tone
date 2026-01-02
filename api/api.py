"""
Digital Empathy Assistant API with Groq LLM
Analyzes sentiment/toxicity and suggests better phrasing using AI
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# DeBERTa model definition
class MultiTaskDeBERTa(nn.Module):
    def __init__(self):
        super().__init__()
        self.deberta = AutoModel.from_pretrained("microsoft/deberta-v3-base")
        self.dropout = nn.Dropout(0.1)
        self.sentiment_classifier = nn.Linear(768, 3)
        self.toxicity_classifier = nn.Linear(768, 4)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        return self.sentiment_classifier(pooled), self.toxicity_classifier(pooled)

# Load DeBERTa model - path relative to this file
# Assumes folder structure: project/api/api.py and project/models/final_deberta_multitask
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "../models/final_deberta_multitask")

print("Loading DeBERTa model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = MultiTaskDeBERTa()

from safetensors.torch import load_file
state_dict = load_file(f"{MODEL_PATH}/model.safetensors")
model.load_state_dict(state_dict)

model.to(device)
model.eval()

print(f"✅ DeBERTa model loaded from: {MODEL_PATH}")
print(f"✅ Running on: {device}")

# Initialize Groq LLM - API key loaded from .env file
# Your .env file should contain: GROQ_API_KEY=your-key-here
try:
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.7,  # Controls creativity (0=deterministic, 1=creative)
        max_tokens=150
    )
    print("✅ Groq LLM initialized")
except Exception as e:
    print(f"⚠️ Groq LLM initialization failed: {e}")
    llm = None

# Prompts for different toxicity levels
REPHRASE_PROMPTS = {
    "toxic": ChatPromptTemplate.from_messages([
        ("system", """You are a helpful communication assistant. Your job is to rephrase toxic or hurtful messages into kind, constructive alternatives while preserving the core intent.

Rules:
- Keep it brief (1-2 sentences max)
- Maintain the original meaning but make it respectful
- Use "I feel" statements when appropriate
- Remove insults, harsh words, and aggressive language
- Be empathetic and constructive"""),
        ("user", "Original message: {text}\n\nRephrase this to be kind and constructive:")
    ]),
    
    "negative": ChatPromptTemplate.from_messages([
        ("system", """You are a helpful communication assistant. Your job is reframe negative messages into more positive or solution-focused alternatives.

Rules:
- Keep it brief (1-2 sentences max)
- Turn complaints into requests for help
- Focus on solutions rather than problems
- Keep the same general topic
- Make it encouraging"""),
        ("user", "Original message: {text}\n\nReframe this more positively:")
    ]),
    
    "mildly_toxic": ChatPromptTemplate.from_messages([
        ("system", """You are a helpful communication assistant. Your job is to soften mildly hostile or passive-aggressive messages.

Rules:
- Keep it brief (1-2 sentences max)
- Remove sarcasm and passive-aggressiveness
- Make the tone more direct and friendly
- Keep the core message intact"""),
        ("user", "Original message: {text}\n\nSoften the tone:")
    ])
}

# Create FastAPI app
app = FastAPI(
    title="Digital Empathy Assistant API",
    version="2.0.0",
    description="AI-powered sentiment and toxicity analysis with intelligent rephrasing"
)

# Enable CORS for all origins (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use Groq LLM to generate smart rephrase suggestions
async def generate_llm_rephrase(text: str, sentiment_idx: int, toxicity_idx: int) -> dict:
    """
    Generates AI-powered rephrase suggestions based on toxicity/sentiment
    
    Returns dict with 'suggestion' and 'reason' (or None if no rephrase needed)
    """
    
    # Fallback if LLM not available
    if llm is None:
        return {"suggestion": None, "reason": "LLM not available"}
    
    # Don't rephrase positive/neutral non-toxic messages
    if toxicity_idx == 0 and sentiment_idx != 2:
        return {"suggestion": None, "reason": None}
    
    try:
        # Select appropriate prompt based on severity
        if toxicity_idx >= 2:  # Toxic or very toxic
            prompt = REPHRASE_PROMPTS["toxic"]
            reason = "This message contains hurtful language"
        elif toxicity_idx == 1:  # Mildly toxic
            prompt = REPHRASE_PROMPTS["mildly_toxic"]
            reason = "This message could be perceived as hostile"
        elif sentiment_idx == 2:  # Negative but not toxic
            prompt = REPHRASE_PROMPTS["negative"]
            reason = "This message is quite negative"
        else:
            return {"suggestion": None, "reason": None}
        
        # Generate rephrase using Groq
        chain = prompt | llm
        response = await chain.ainvoke({"text": text})
        
        suggestion = response.content.strip()
        
        # Clean up quotes that LLM might add
        if suggestion.startswith('"') and suggestion.endswith('"'):
            suggestion = suggestion[1:-1]
        
        return {"suggestion": suggestion, "reason": reason}
        
    except Exception as e:
        print(f"Error generating rephrase: {e}")
        # Fallback suggestion
        return {
            "suggestion": "Consider rephrasing this more constructively.",
            "reason": "This message could be improved"
        }

# Generate user-friendly feedback messages
def generate_feedback(sentiment_idx, toxicity_idx):
    if toxicity_idx >= 2:
        return "⚠️ This message may be hurtful. Consider rephrasing."
    elif toxicity_idx == 1:
        return "💭 This could come across as mildly hostile. Try a gentler tone?"
    elif sentiment_idx == 0:
        return "✨ Great message! Very positive and encouraging tone."
    elif sentiment_idx == 2:
        return "📝 This seems negative. Is everything okay?"
    else:
        return "✅ Message looks good!"

# Root endpoint - health check
@app.get("/")
def home():
    return {
        "status": "running",
        "version": "2.0.0",
        "features": ["sentiment analysis", "toxicity detection", "AI-powered rephrasing"],
        "llm_available": llm is not None,
        "endpoints": {
            "analyze": "POST /analyze",
            "health": "GET /health"
        }
    }

# Detailed health check
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "deberta_loaded": True,
        "llm_available": llm is not None,
        "device": str(device)
    }

# Main analysis endpoint
@app.post("/analyze")
async def analyze(data: dict):
    """
    Analyzes text for sentiment/toxicity and provides AI rephrasing suggestions
    
    Request: {"text": "your message here"}
    Response: sentiment, toxicity, feedback, and rephrase suggestion
    """
    
    # Validate input
    text = data.get("text", "").strip()
    
    if not text:
        return {"error": "Text cannot be empty"}
    
    if len(text) > 500:
        return {"error": "Text too long (max 500 characters)"}
    
    # Tokenize text for model
    inputs = tokenizer(
        text, 
        return_tensors='pt', 
        padding=True, 
        truncation=True, 
        max_length=128
    ).to(device)
    
    # Run prediction
    with torch.no_grad():
        sentiment_logits, toxicity_logits = model(
            inputs['input_ids'], 
            inputs['attention_mask']
        )
    
    # Convert logits to probabilities
    sentiment_probs = torch.softmax(sentiment_logits, dim=1)[0].cpu().numpy()
    toxicity_probs = torch.softmax(toxicity_logits, dim=1)[0].cpu().numpy()
    
    # Get sentiment prediction
    sentiment_idx = int(sentiment_probs.argmax())
    
    # Get toxicity prediction with confidence thresholds
    # Higher thresholds reduce false positives
    if toxicity_probs[3] > 0.3:  # Very toxic threshold
        toxicity_idx = 3
    elif toxicity_probs[2] > 0.4:  # Toxic threshold
        toxicity_idx = 2
    elif toxicity_probs[1] > 0.35:  # Mildly toxic threshold
        toxicity_idx = 1
    else:
        toxicity_idx = 0  # Non-toxic
    
    # Map indices to labels
    sentiment_labels = ['positive', 'neutral', 'negative']
    toxicity_labels = ['non-toxic', 'mildly toxic', 'toxic', 'very toxic']
    
    sentiment_label = sentiment_labels[sentiment_idx]
    toxicity_label = toxicity_labels[toxicity_idx]
    
    # Generate feedback message
    feedback = generate_feedback(sentiment_idx, toxicity_idx)
    
    # Generate AI rephrase suggestion (uses Groq API)
    rephrase_result = await generate_llm_rephrase(text, sentiment_idx, toxicity_idx)
    
    # Build response
    response = {
        "text": text,
        "sentiment": {
            "label": sentiment_label,
            "confidence": float(sentiment_probs[sentiment_idx]),
            "scores": {
                sentiment_labels[i]: float(sentiment_probs[i]) 
                for i in range(3)
            }
        },
        "toxicity": {
            "label": toxicity_label,
            "confidence": float(toxicity_probs[toxicity_idx]),
            "scores": {
                toxicity_labels[i]: float(toxicity_probs[i]) 
                for i in range(4)
            }
        },
        "feedback": feedback,
        "should_warn": toxicity_idx >= 2,
        "rephrase": rephrase_result
    }
    
    return response

# Batch analysis endpoint
@app.post("/batch-analyze")
async def batch_analyze(data: dict):
    """
    Analyzes multiple texts at once
    Request: {"texts": ["text1", "text2", "text3"]}
    """
    
    texts = data.get("texts", [])
    
    if not texts:
        return {"error": "No texts provided"}
    
    if len(texts) > 50:
        return {"error": "Maximum 50 texts per batch"}
    
    # Process each text
    results = []
    for text in texts:
        try:
            result = await analyze({"text": text})
            results.append(result)
        except Exception as e:
            results.append({"error": str(e), "text": text})
    
    return {
        "results": results,
        "total": len(results)
    }

# Run server
if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting Digital Empathy Assistant API v2.0")
    print("📝 Features: Sentiment Analysis + Toxicity Detection + AI Rephrasing")
    
    # Get port from environment (for production) or use 8000 (for local)
    port = int(os.getenv("PORT", 8000))
    
    print(f"🔗 Server running on port {port}")
    print(f"🔗 Docs: http://localhost:{port}/docs\n")
    
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)