from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import random
import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import urllib.parse
from textblob import TextBlob # Import for Sentiment Analysis

# Initialize App
app = FastAPI(title="FactCheck AI Backend")

# --- CORS Configuration ---
# TEMPORARY FIX: We are setting allow_origins=["*"] to ensure the Vercel connection works.
# Once we confirm the connection, we should replace "*" with the specific Vercel URL.
origins = ["*"] 

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
    
    # --- 0. Sentiment Analysis (New Feature) ---
    blob = TextBlob(text)
    sentiment = blob.sentiment
    subjectivity = sentiment.subjectivity # 0.0 (Objective) to 1.0 (Subjective)
    polarity = sentiment.polarity         # -1.0 (Negative) to 1.0 (Positive)

    fake_score = 0
    real_score = 0

    # Subjectivity Check: Real news is usually objective (low subjectivity)
    if subjectivity > 0.5:
        fake_score += 2.0
        factors.append(f"🚩 High subjectivity detected ({round(subjectivity*100)}%). Content appears opinionated rather than factual.")
    else:
        real_score += 1.0
        factors.append("✅ Tone is objective and neutral (Low subjectivity).")

    # Polarity Check: Real news is rarely extremely emotional
    if abs(polarity) > 0.8:
        fake_score += 1.5
        factors.append("🚩 Extremely emotional language detected.")

    # 1. Fake Indicators (Expanded with Health Scams & Conspiracies)
    fake_triggers = {
        "shocking": "Uses emotionally charged language ('shocking').",
        "secret": "Claims to reveal 'secret' information.",
        "exposed": "Uses sensationalist terms like 'exposed'.",
        "they don't want you to know": "Appeals to conspiracy narratives.",
        "100%": "Makes absolute claims ('100%').",
        "guaranteed": "Uses marketing-style language.",
        "share before deleted": "Creates artificial urgency.",
        "miracle": "Promises 'miracle' results.",
        "censored": "Claims censorship to build false credibility.",
        "banned": "Claims item is 'banned' to create intrigue.",
        "leaked": "Uses 'leaked' to imply forbidden knowledge.",
        "viral": "Focuses on virality rather than facts.",
        "you won't believe": "Clickbait phrasing detected.",
        "end of the world": "Fear-mongering detected.",
        "cure": "Promises a 'cure' (medical news usually says 'treatment' or 'results').",
        "permanently": "Makes absolute promises about permanence.",
        "within days": "Promises unrealistic timelines.",
        "simple trick": "Uses clickbait 'trick' language.",
        "doctors hate": "Appeals to anti-medical conspiracy.",
        "big pharma": "Uses conspiracy theorist terminology.",
        "mainstream media": "Attempts to discredit standard journalism.",
        "wake up": "Uses cult-like awakening language.",
        "truth about": "Implies a hidden truth vs public lie.",
        "government plot": "Directly alleges conspiracy without evidence.",
        "hidden agenda": "Implies malicious intent without proof."
    }
    
    # 2. Real Indicators (Significantly Expanded)
    real_triggers = {
        "official": "Cites 'official' sources.",
        "report": "References a 'report' or structured document.",
        "study": "Mentions a research 'study'.",
        "according to": "Attributes information to a specific source.",
        "statement": "References a formal statement.",
        "analysis": "Indicates analytical depth.",
        "confirmed": "Uses verification language ('confirmed').",
        "government": "References government authority.",
        "court": "References judicial proceedings.",
        "police": "References law enforcement.",
        "announced": "Uses standard reporting verb 'announced'.",
        "said": "Uses neutral attribution verb 'said'.",
        "deal": "References a business or diplomatic 'deal'.",
        "signed": "References a formal agreement being 'signed'.",
        "sources": "Attributes info to 'sources'.",
        "minister": "References a government official.",
        "department": "References an official department.",
        "university": "Cites an academic institution.",
        "research": "References scientific research.",
        "published in": "Cites a publication venue.",
        "journal": "References academic or professional journals.",
        "spokesperson": "Attributes quote to an official representative.",
        "evidence suggests": "Uses cautious, scientific language."
    }

    # 3. Trusted Source Mentions (New Category)
    trusted_sources = [
        "who", "cdc", "nasa", "fda", "un", "nato", "reuters", "ap", "afp", "bbc", 
        "cnn", "nytimes", "washington post", "guardian", "isro", "rbi", "sebi", "iit", "aiims"
    ]
    
    # Check Fake Triggers
    for word, reason in fake_triggers.items():
        if word in text_lower:
            fake_score += 1.5 
            factors.append(f"🚩 {reason}")
            
    # Check Real Triggers
    for word, reason in real_triggers.items():
        if word in text_lower:
            real_score += 1
            factors.append(f"✅ {reason}")

    # Check Trusted Sources (Boost Real Score)
    for source in trusted_sources:
        if f" {source} " in f" {text_lower} " or f"{source}." in text_lower: # basic word boundary check
            real_score += 2.0
            factors.append(f"✅ Cites reputable entity ('{source.upper()}').")

    # 4. Structural Checks
    if len(text) > 20 and sum(1 for c in text if c.isupper()) / len(text) > 0.6:
        fake_score += 2
        factors.append("🚩 Excessive use of capitalization detected.")
    
    if text.count("!") > 3:
        fake_score += 1
        factors.append("🚩 Excessive exclamation marks detected.")

    # 5. Contextual Logic (The "Robustness" Layer)
    # If text makes medical claims ("cure", "doctor", "health") but lacks scientific backing triggers ("study", "journal"), penalize it.
    medical_keywords = ["cure", "medicine", "health", "doctor", "treatment", "virus", "disease", "diabetes", "cancer"]
    has_medical_context = any(word in text_lower for word in medical_keywords)
    has_scientific_backing = any(word in text_lower for word in ["study", "research", "journal", "clinical", "trial", "published", "report"])
    
    if has_medical_context and not has_scientific_backing:
        fake_score += 2.0
        factors.append("🚩 Makes medical claims without citing studies, trials, or reports.")

    # 6. Calculate Confidence (Weighted Logic)
    # If no triggers found, heavily rely on sentiment and structure
    if fake_score == 0 and real_score == 0:
        if subjectivity < 0.4 and len(text.split()) > 6:
             # Neutral, grammatical sentences are likely real
             conf_fake = 0.3 
             conf_real = 0.7
             factors.append("✅ No sensationalism found and tone is objective.")
        elif subjectivity > 0.6:
             conf_fake = 0.65
             conf_real = 0.35
             factors.append("ℹ️ High subjectivity suggests opinion, but no specific fake keywords found.")
        else:
             # Too ambiguous
             conf_fake = 0.5
             conf_real = 0.5
             factors.append("ℹ️ Text is neutral but lacks verifiable citations.")
    else:
        total = fake_score + real_score + 0.1
        conf_fake = fake_score / total
        conf_real = real_score / total
        
        # Boost the winner
        if conf_fake > conf_real:
            conf_fake = min(0.99, conf_fake + 0.15)
            conf_real = 1 - conf_fake
        else:
            conf_real = min(0.99, conf_real + 0.15)
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

    # 7. Fetch Live News (The New Combination Feature)
    related_news = search_google_news(text)

    # 8. Static Resources
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
        "related_news": related_news,      # Live results
        "verification_tools": verification_tools # Static tools
    }

# --- Endpoints ---

@app.get("/")
def home():
    return {"status": "active", "message": "FactCheck AI Backend is running."}

@app.post("/predict_text")
@app.post("/predict")
@app.post("/analyze")
@app.post("/check")
@app.post("/analyze-text")
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