from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import random
import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import urllib.parse
from textblob import TextBlob 
import os # Import os to read environment variables

# Initialize App
app = FastAPI(title="FactCheck AI Backend")

# --- CORS Configuration ---
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

# --- Helper: Remote NLI Assistant (Hugging Face API) ---
# This uses a powerful Transformer model remotely to check if text is Fact, Opinion, or Conspiracy.
def check_nli_remote(text):
    hf_token = os.getenv("HF_API_KEY") # Get key from Render Environment Variables
    if not hf_token:
        return None # Skip if no key is set (graceful fallback)

    api_url = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
    headers = {"Authorization": f"Bearer {hf_token}"}
    
    # We ask the model to classify the text into these categories (Zero-Shot Classification)
    payload = {
        "inputs": text,
        "parameters": {
            "candidate_labels": ["factual reporting", "personal opinion", "conspiracy theory", "satire"],
            "multi_label": False
        }
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=2.5)
        if response.status_code != 200:
            return None
        return response.json()
    except:
        return None

# --- Smart Analysis Logic ---
def analyze_content(text: str, source_type: str = "text"):
    text_lower = text.lower()
    factors = [] 
    
    # --- 0. Sentiment Analysis ---
    blob = TextBlob(text)
    sentiment = blob.sentiment
    subjectivity = sentiment.subjectivity # 0.0 (Objective) to 1.0 (Subjective)
    polarity = sentiment.polarity         # -1.0 (Negative) to 1.0 (Positive)

    fake_score = 0
    real_score = 0

    # Subjectivity Check
    if subjectivity > 0.5:
        fake_score += 2.0
        factors.append(f"🚩 High subjectivity detected ({round(subjectivity*100)}%). Content appears opinionated rather than factual.")
    else:
        real_score += 1.0
        factors.append("✅ Tone is objective and neutral (Low subjectivity).")

    # Polarity Check
    if abs(polarity) > 0.8:
        fake_score += 1.5
        factors.append("🚩 Extremely emotional language detected.")

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

    # 2. Professional Misinformation Phrase Patterns
    vague_patterns = {
        "experts familiar with": "Uses vague 'experts' to imply insider knowledge without naming sources.",
        "sources close to": "Relying on unnamed 'close sources' is a common trope in unverified leaks.",
        "according to internal": "Cites 'internal' sources without providing documents.",
        "it is believed that": "Uses passive voice to state opinions as facts without attribution.",
        "questions are being asked": "Uses passive framing to imply controversy where none may exist.",
        "anonymous sources confirm": "Uses anonymity to shield the source from verification.",
        "it has been revealed": "Passive voice implies a revelation without stating who revealed it.",
        "growing body of evidence": "Vague appeal to 'evidence' without citing specific studies.",
        "many people are saying": "Classic 'Bandwagon' propaganda technique.",
        "up to us to decide": "Appeals to emotion rather than fact."
    }
    
    # 3. Real Indicators
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
        "evidence suggests": "Uses cautious, scientific language.",
        "experts": "Attributes info to domain experts.",
        "analysts": "Attributes info to professional analysts."
    }

    # 4. Trusted Source Mentions
    trusted_sources = [
        "who", "cdc", "nasa", "fda", "un", "nato", "reuters", "ap", "afp", "bbc", 
        "cnn", "nytimes", "washington post", "guardian", "isro", "rbi", "sebi", "iit", "aiims"
    ]
    
    valid_citation_context = [
        "report", "study", "journal", "published", "publication", "statement", "released", 
        "official", "data", "link", "website", "202", "201", "according to", "cited"
    ]
    
    # Check Fake Triggers
    for word, reason in fake_triggers.items():
        if word in text_lower:
            fake_score += 1.5 
            factors.append(f"🚩 {reason}")

    # Check Vague Phrase Patterns
    for phrase, reason in vague_patterns.items():
        if phrase in text_lower:
            fake_score += 2.0
            factors.append(f"🚩 Phrase Pattern Detected: {reason}")
            
    # Check Real Triggers
    for word, reason in real_triggers.items():
        if word in text_lower:
            real_score += 1
            factors.append(f"✅ {reason}")

    # Check Trusted Sources
    for source in trusted_sources:
        if f" {source} " in f" {text_lower} " or f"{source}." in text_lower:
            has_context = any(ctx in text_lower for ctx in valid_citation_context)
            if has_context:
                real_score += 2.0
                factors.append(f"✅ Cites reputable entity ('{source.upper()}') with verifiable context.")

    # 5. Structural Checks
    if len(text) > 20 and sum(1 for c in text if c.isupper()) / len(text) > 0.6:
        fake_score += 2
        factors.append("🚩 Excessive use of capitalization detected.")
    
    if text.count("!") > 3:
        fake_score += 1
        factors.append("🚩 Excessive exclamation marks detected.")

    # 6. Contextual Logic
    medical_keywords = ["cure", "medicine", "health", "doctor", "treatment", "virus", "disease", "diabetes", "cancer", "brain", "blood", "human body"]
    has_medical_context = any(word in text_lower for word in medical_keywords)
    has_scientific_backing = any(word in text_lower for word in ["study", "research", "journal", "clinical", "trial", "published", "report"])
    
    if has_medical_context and not has_scientific_backing:
        fake_score += 2.0
        factors.append("🚩 Makes medical claims without citing studies, trials, or reports.")

    # --- Contradiction Detector ---
    if "secret" in text_lower and ("study" in text_lower or "report" in text_lower):
        fake_score += 3.0
        factors.append("🚩 Contradictory sourcing detected: legitimate scientific studies are rarely 'secret'.")

    # --- 7. Claim-Evidence Alignment Check ---
    strong_claim_words = ["cure", "proven", "guaranteed", "100%", "miracle", "permanently", "undoubtedly", "definitely"]
    weak_evidence_words = ["suggests", "might", "could", "linked to", "associated with", "survey", "observation", "potential", "preliminary"]
    strong_evidence_words = ["clinical trial", "randomized", "peer-reviewed", "meta-analysis", "systematic review", "conclusive", "consensus"]

    has_strong_claim = any(word in text_lower for word in strong_claim_words)
    has_weak_evidence = any(word in text_lower for word in weak_evidence_words)
    has_strong_evidence = any(word in text_lower for word in strong_evidence_words)

    if has_strong_claim and has_weak_evidence and not has_strong_evidence:
        fake_score += 2.5
        factors.append("🚩 Claim-Evidence Mismatch: Absolute claims supported only by weak/observational language.")

    if has_strong_claim and has_strong_evidence:
        real_score += 2.5
        factors.append("✅ Claim-Evidence Alignment: Strong claims backed by robust scientific terminology.")

    # --- 8. High-Risk Domain Multiplier ---
    high_risk_domains = {
        "Health & Medicine": ["medicine", "doctor", "health", "virus", "disease", "cancer", "diabetes", "vaccine", "cure", "treatment", "patient", "hospital", "weight loss", "diet", "nutrition"],
        "Finance": ["stock", "market", "crypto", "bitcoin", "invest", "profit", "bank", "economy", "dollar", "rupee", "crash", "wealth"],
        "Politics": ["election", "vote", "poll", "government", "senate", "parliament", "congress", "president", "minister", "law", "ballot", "voting"]
    }
    
    detected_domain = None
    for domain, keywords in high_risk_domains.items():
        if any(k in text_lower for k in keywords):
            detected_domain = domain
            break
            
    if detected_domain:
        if real_score < 2.0:
            fake_score += 2.0
            factors.append(f"⚠️ High-Risk Topic ({detected_domain}): Requires strict verifiable sourcing (studies, official reports) which is missing.")
            
        anecdotal_phrases = ["worked for me", "i tried", "simple trick", "just one", "my friend", "grandmother", "ancestors", "believe me"]
        if any(p in text_lower for p in anecdotal_phrases):
            fake_score += 2.0
            factors.append(f"🚩 High-Risk Topic: Anecdotal evidence ('worked for me') is not scientific proof.")

    # --- 9. Remote NLI Check (NEW) ---
    # Ask the Transformer model: "Is this fact, opinion, or conspiracy?"
    nli_result = check_nli_remote(text)
    if nli_result and 'labels' in nli_result:
        top_label = nli_result['labels'][0]
        top_score = nli_result['scores'][0]
        
        if top_score > 0.6: # Only act if model is confident
            if top_label == "conspiracy theory":
                fake_score += 3.0
                factors.append(f"🚩 AI Deep Analysis: Transformer model classified content as '{top_label}'.")
            elif top_label == "personal opinion":
                fake_score += 1.5
                factors.append(f"🚩 AI Deep Analysis: Classified as subjective '{top_label}' rather than fact.")
            elif top_label == "factual reporting":
                real_score += 2.5
                factors.append(f"✅ AI Deep Analysis: Transformer model classified content as '{top_label}'.")

    # 10. Calculate Confidence
    CONFIDENCE_THRESHOLD = 0.60 

    if fake_score == 0 and real_score == 0:
        classification = "Unverified"
        conf_fake = 0.5
        conf_real = 0.5
        if subjectivity > 0.6:
             factors.append("ℹ️ High subjectivity suggests opinion, but lacks verifiable evidence.")
        else:
             factors.append("ℹ️ Content is neutral but lacks sufficient data or citations to verify.")
    else:
        total = fake_score + real_score + 0.1 
        conf_fake = fake_score / total
        conf_real = real_score / total
        
        if conf_fake > CONFIDENCE_THRESHOLD:
            classification = "Fake"
            conf_fake = min(0.99, conf_fake + 0.1)
            conf_real = 1 - conf_fake
        elif conf_real > CONFIDENCE_THRESHOLD:
            classification = "Real"
            conf_real = min(0.99, conf_real + 0.1)
            conf_fake = 1 - conf_real
        else:
            classification = "Unverified"
            if conf_fake > conf_real:
                conf_fake = 0.55
                conf_real = 0.45
            else:
                conf_fake = 0.45
                conf_real = 0.55
            factors.append("⚠️ Ambiguous Evidence: Signals are mixed or too weak for a definitive rating.")

    # 11. Natural Language Explanation
    explanation = ""
    if classification == "Fake":
        reasons = []
        if any("medical claims" in f for f in factors): reasons.append("makes unverified medical claims")
        if any("subjectivity" in f for f in factors): reasons.append("uses highly subjective language")
        if any("sensationalist" in f or "emotionally" in f or "urgency" in f for f in factors): reasons.append("relies on sensationalism")
        if any("Contradictory" in f for f in factors): reasons.append("references sources contradictorily")
        if any("AI Deep Analysis" in f for f in factors): reasons.append("deep learning model detected conspiracy patterns")
        
        if reasons:
            explanation = f"Flagged as Misinformation because it {', and '.join(reasons[:2])}."
        else:
            explanation = "Flagged as likely misinformation due to a lack of verifiable sources and presence of suspicious patterns."
            
    elif classification == "Real":
        reasons = []
        if any("reputable entity" in f for f in factors): reasons.append("cites reputable sources with context")
        if any("Alignment" in f for f in factors): reasons.append("claims are supported by strong evidence")
        if any("AI Deep Analysis" in f for f in factors): reasons.append("deep learning model validated the reporting style")
        
        if reasons:
            explanation = f"Rated Credible because it {', and '.join(reasons)}."
        else:
            explanation = "Rated Credible as it follows standard journalistic structures without sensationalism."
            
    else: # Unverified
        explanation = "Rated Unverified. The content does not contain obvious misinformation triggers, but also lacks the strong citations or authoritative references required for a 'Credible' rating."

    # 12. Fetch Live News
    related_news = search_google_news(text)

    # 13. Static Resources
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
        "explanation": explanation,
        "related_news": related_news,
        "verification_tools": verification_tools
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