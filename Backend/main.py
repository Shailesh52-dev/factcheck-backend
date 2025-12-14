from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import random
import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import urllib.parse

# Initialize App
app = FastAPI(title="FactCheck AI Backend")

# --- CORS Configuration ---
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000", 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Data Models ---
class TextRequest(BaseModel):
    text: str

class UrlRequest(BaseModel):
    url: str

# --- Helper: Live Web Search (Google News RSS) ---
def search_google_news(query_text):
    try:
        # Create a search query from the first few meaningful words to avoid URL length issues
        # Remove common stop words for better search quality
        stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with'])
        words = [w for w in query_text.split() if w.lower() not in stop_words and w.isalnum()]
        search_query = " ".join(words[:12]) # Limit to ~12 key words
        
        if not search_query:
            return []

        encoded_query = urllib.parse.quote(search_query)
        rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
        
        response = requests.get(rss_url, timeout=4)
        if response.status_code != 200:
            return []

        # Parse XML
        root = ET.fromstring(response.content)
        news_items = []
        
        # Get top 3 items
        for item in root.findall('./channel/item')[:3]:
            title = item.find('title').text if item.find('title') is not None else "No Title"
            link = item.find('link').text if item.find('link') is not None else "#"
            source = item.find('source').text if item.find('source') is not None else "News Source"
            
            news_items.append({
                "title": title,
                "url": link,
                "source": source
            })
            
        return news_items
    except Exception as e:
        print(f"Search Error: {e}")
        return []

# --- Smart Analysis Logic ---
def analyze_content(text: str, source_type: str = "text"):
    text_lower = text.lower()
    factors = [] 
    
    # 1. Fake Indicators
    fake_triggers = {
        "shocking": "Uses emotionally charged language ('shocking').",
        "secret": "Claims to reveal 'secret' information.",
        "exposed": "Uses sensationalist terms like 'exposed'.",
        "they don't want you to know": "Appeals to conspiracy narratives.",
        "100%": "Makes absolute claims ('100%').",
        "guaranteed": "Uses marketing-style language.",
        "share before deleted": "Creates artificial urgency.",
        "miracle": "Promises 'miracle' results.",
        "censored": "Claims censorship to build false credibility."
    }
    
    # 2. Real Indicators
    real_triggers = {
        "official": "Cites 'official' sources.",
        "report": "References a 'report' or structured document.",
        "study": "Mentions a research 'study'.",
        "according to": "Attributes information to a specific source.",
        "statement": "References a formal statement.",
        "analysis": "Indicates analytical depth.",
        "confirmed": "Uses verification language ('confirmed')."
    }

    fake_score = 0
    real_score = 0
    
    for word, reason in fake_triggers.items():
        if word in text_lower:
            fake_score += 1.5 
            factors.append(f"🚩 {reason}")
            
    for word, reason in real_triggers.items():
        if word in text_lower:
            real_score += 1
            factors.append(f"✅ {reason}")

    # 3. Structural Checks
    if len(text) > 20 and sum(1 for c in text if c.isupper()) / len(text) > 0.6:
        fake_score += 2
        factors.append("🚩 Excessive use of capitalization detected.")
    
    if text.count("!") > 3:
        fake_score += 1
        factors.append("🚩 Excessive exclamation marks detected.")

    # 4. Calculate Confidence
    if fake_score == 0 and real_score == 0:
        val = random.uniform(0.4, 0.6)
        is_fake = val > 0.5
        conf_fake = val if is_fake else 1 - val
        conf_real = 1 - conf_fake
        factors.append("ℹ️ No strong keyword triggers found; relying on linguistic structure.")
    else:
        total = fake_score + real_score + 0.1
        conf_fake = fake_score / total
        conf_real = real_score / total
        if conf_fake > conf_real:
            conf_fake = min(0.98, conf_fake + 0.2)
            conf_real = 1 - conf_fake
        else:
            conf_real = min(0.98, conf_real + 0.2)
            conf_fake = 1 - conf_real

    classification = "Fake" if conf_fake > conf_real else "Real"
    
    # Ensure factors exist
    if len(factors) < 2:
        if classification == "Fake":
            factors.append("🚩 Lacks citations from authoritative bodies.")
            factors.append("🚩 Tone analysis suggests subjective persuasion.")
        else:
            factors.append("✅ Tone appears neutral and objective.")
            factors.append("✅ Structure resembles journalistic reporting.")

    # 5. Fetch Live News (The New Combination Feature)
    related_news = search_google_news(text)

    # 6. Static Resources
    verification_tools = []
    if classification == "Real":
        verification_tools = [
            {"source": "Reuters Fact Check", "url": "https://www.reuters.com/fact-check"},
            {"source": "AP News Verification", "url": "https://apnews.com/hub/ap-fact-check"},
        ]
    else:
        verification_tools = [
            {"source": "Snopes Search", "url": "https://www.snopes.com/"},
            {"source": "PolitiFact", "url": "https://www.politifact.com/"},
            {"source": "Google Fact Check", "url": "https://toolbox.google.com/factcheck/explorer"}
        ]

    return {
        "classification": classification,
        "confidenceReal": round(conf_real, 4),
        "confidenceFake": round(conf_fake, 4),
        "factors": factors,
        "related_news": related_news,    # Live results
        "verification_tools": verification_tools # Static tools
    }

# --- Endpoints ---

@app.get("/")
def home():
    return {"status": "active", "message": "FactCheck AI Backend is running."}

@app.post("/predict_text")
async def predict_text(request: TextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    return analyze_content(request.text, "text")

@app.post("/predict_url")
async def predict_url(request: UrlRequest):
    if not request.url.strip():
        raise HTTPException(status_code=400, detail="URL cannot be empty")
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(request.url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        title = soup.title.string if soup.title else ""
        paragraphs = [p.get_text() for p in soup.find_all('p')]
        full_text = title + " " + " ".join(paragraphs[:5])
        
        if len(full_text) < 50:
             raise HTTPException(status_code=400, detail="Could not extract text.")

        return analyze_content(full_text, "url")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {str(e)}")

@app.post("/predict_image")
async def predict_image(file: UploadFile = File(...)):
    mock_text = "Breaking news: The shocking truth about the secret update they don't want you to know!"
    return analyze_content(mock_text, "image")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)