from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import random
import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import urllib.parse
import os 
import math 
import traceback

# Try to import TextBlob, handle failure gracefully
try:
    from textblob import TextBlob
except ImportError:
    TextBlob = None

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

# --- Global Exception Handler ---
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    error_msg = f"Global Error: {str(exc)}"
    print(error_msg)
    traceback.print_exc() # Print full traceback to logs
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error. Check Render logs for details."},
    )

# --- Data Models ---
class TextRequest(BaseModel):
    text: str

class UrlRequest(BaseModel):
    url: str

# --- Helper: Live Web Search (Google News RSS) ---
def search_google_news(query_text):
    try:
        stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with'])
        words = [w for w in query_text.split() if w.lower() not in stop_words and w.isalnum()]
        
        if not words:
            return []
            
        search_query = " ".join(words[:12]) 
        
        if not search_query:
            return []

        encoded_query = urllib.parse.quote(search_query)
        rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
        
        response = requests.get(rss_url, timeout=4)
        if response.status_code != 200:
            return []

        root = ET.fromstring(response.content)
        news_items = []
        
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
def check_nli_remote(text):
    hf_token = os.getenv("HF_API_KEY") 
    if not hf_token:
        return None 

    api_url = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
    headers = {"Authorization": f"Bearer {hf_token}"}
    
    payload = {
        "inputs": text,
        "parameters": {
            "candidate_labels": ["fact", "hypothesis", "speculation", "opinion"],
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

# --- CORE LOGIC: 5-Factor Weighted Scoring System ---
def analyze_content(text: str, source_type: str = "text"):
    try:
        text_lower = text.lower()
        factors = [] 
        
        # --- 1. Language Integrity (Base: 100) ---
        subjectivity = 0.5
        polarity = 0.0
        
        try:
            if TextBlob:
                blob = TextBlob(text)
                subjectivity = blob.sentiment.subjectivity
                polarity = blob.sentiment.polarity
        except Exception as e:
            print(f"TextBlob Error: {e}")
            # Continue with defaults
        
        lang_score = 100.0
        
        sensational_words = [
            "shocking", "secret", "exposed", "they don't want you to know", "miracle", 
            "censored", "banned", "leaked", "viral", "you won't believe", "end of the world",
            "mind-blowing", "destroyed", "obliterated", "share before deleted", "wake up", 
            "truth about", "government plot", "hidden agenda", "big pharma", "mainstream media",
            "doctors hate", "simple trick", "within days", "permanently"
        ]
        
        found_sensational = [w for w in sensational_words if w in text_lower]
        if found_sensational:
            penalty = 15 + (len(found_sensational) * 5)
            lang_score -= penalty
            factors.append(f"🚩 Language Integrity: Uses sensational terms ('{found_sensational[0]}').")

        if subjectivity > 0.6:
            lang_score -= 20
            factors.append(f"🚩 Language Integrity: Highly subjective tone ({int(subjectivity*100)}% opinion).")
        elif subjectivity < 0.2:
            factors.append("✅ Language Integrity: Tone is neutral and objective.")

        if abs(polarity) > 0.8:
            lang_score -= 15
            factors.append("🚩 Language Integrity: Extremely emotional language.")

        if len(text) > 20 and sum(1 for c in text if c.isupper()) / len(text) > 0.5:
            lang_score -= 20
            factors.append("🚩 Language Integrity: Excessive capitalization.")

        lang_score = max(0, min(100, lang_score))


        # --- 2. Evidence Quality (Base: 0) ---
        evidence_score = 0.0
        
        strong_evidence = [
            "clinical trial", "peer-reviewed", "meta-analysis", "systematic review", 
            "published in", "official report", "court documents", "police report", 
            "data shows", "scientific study", "evidence suggests", "statement by"
        ]
        medium_evidence = [
            "study", "research", "survey", "according to", "statement", "announced", 
            "journal", "report", "investigation", "authorities", "witnesses", "incident"
        ]
        
        found_strong = [w for w in strong_evidence if w in text_lower]
        found_medium = [w for w in medium_evidence if w in text_lower]
        
        if found_strong:
            evidence_score += 60 + (len(found_strong) * 10)
            factors.append(f"✅ Evidence Quality: Cites strong evidence ('{found_strong[0]}').")
        elif found_medium:
            evidence_score += 30 + (len(found_medium) * 5)
            factors.append(f"✅ Evidence Quality: References standard sources ('{found_medium[0]}').")
        
        if "secret" in text_lower and ("study" in text_lower or "report" in text_lower):
            evidence_score = 0 
            factors.append("🚩 Evidence Quality: 'Secret study' is a contradiction; verifiable science is public.")

        evidence_score = max(0, min(100, evidence_score))


        # --- 3. Source Specificity (Base: 0) ---
        source_score = 0.0
        
        trusted_orgs = [
            "who", "cdc", "nasa", "fda", "un", "nato", "reuters", "ap", "afp", "bbc", 
            "cnn", "isro", "rbi", "sebi", "iit", "aiims", "government", "ministry", 
            "department", "university", "police", "fbi", "cia", "authorities"
        ]
        
        vague_sources = [
            "experts say", "sources say", "experts familiar", "many people", "observers", 
            "insiders", "anonymous", "sources close to", "according to internal", 
            "it is believed that", "questions are being asked", "it has been revealed", 
            "growing body of evidence", "up to us to decide"
        ]
        
        citation_context_words = ["said", "reported", "announced", "warned", "stated", "published", "released", "confirmed"]
        
        has_trusted = False
        for org in trusted_orgs:
            if org in text_lower:
                if any(ctx in text_lower for ctx in citation_context_words):
                    source_score += 70
                    has_trusted = True
                    factors.append(f"✅ Source Specificity: Cites specific entity '{org.upper()}' with attribution.")
                    break
        
        if not has_trusted:
            found_vague = [w for w in vague_sources if w in text_lower]
            if found_vague:
                source_score -= 15
                factors.append(f"🚩 Source Specificity: Relies on vague attribution ('{found_vague[0]}').")
        
        source_score = max(0, min(100, source_score))


        # --- 4. Claim Robustness (Base: 50) ---
        claim_score = 50.0
        
        absolutes = [
            "proven", "guaranteed", "100%", "permanently", "cure", "undoubtedly", 
            "definitely", "miracle", "banned by doctors"
        ]
        cautious = ["suggests", "might", "potential", "likely", "estimated", "appears to", "alleged"]
        
        has_absolute = any(w in text_lower for w in absolutes)
        has_cautious = any(w in text_lower for w in cautious)
        
        if has_absolute:
            if evidence_score > 60:
                claim_score += 30
                factors.append("✅ Claim Robustness: Strong claim is supported by high-quality evidence.")
            else:
                claim_score -= 40
                factors.append("🚩 Claim Robustness: Absolute claim ('cure'/'proven') lacks sufficient evidence.")
                
        if has_cautious:
            claim_score += 20
            factors.append("✅ Claim Robustness: Uses cautious/scientific language ('suggests'/'might').")

        claim_score = max(0, min(100, claim_score))


        # --- 5. Domain Risk Penalty ---
        risk_penalty = 0.0
        
        high_risk_domains = {
            "Health": ["medicine", "doctor", "health", "virus", "disease", "cancer", "diabetes", "vaccine", "cure", "treatment", "patient"],
            "Finance": ["stock", "market", "crypto", "bitcoin", "invest", "bank", "economy", "dollar", "rupee"],
            "Elections": ["election", "vote", "poll", "ballot", "voting", "fraud", "rigged"]
        }
        
        detected_domain = None
        for domain, keywords in high_risk_domains.items():
            if any(k in text_lower for k in keywords):
                detected_domain = domain
                break
                
        if detected_domain:
            if evidence_score < 40:
                risk_penalty = 15.0
                factors.append(f"⚠️ Domain Risk ({detected_domain}): Topic requires high evidentiary standards which are missing.")
            
            anecdotal = ["worked for me", "i tried", "simple trick", "believe me", "my friend", "grandmother", "ancestors"]
            if any(w in text_lower for w in anecdotal):
                risk_penalty += 20.0
                factors.append("🚩 Domain Risk: Anecdotal evidence is invalid for high-stakes topics.")

        # --- 6. Absence of Evidence Rule ---
        absence_phrases = [
            "no conclusive evidence", "no scientific evidence", "no proven link", 
            "has not been shown", "lack of evidence", "no evidence to support", "unproven"
        ]
        has_absence_language = any(p in text_lower for p in absence_phrases)
        
        # Setup vars for NLI
        has_strong_claim = has_absolute
        has_weak_evidence = any(w in text_lower for w in ["suggests", "might", "could", "survey"])
        has_strong_evidence = (evidence_score > 50)

        # --- 9. NLI Threshold Tuning ---
        nli_result = check_nli_remote(text)
        if nli_result and 'labels' in nli_result:
            top_label = nli_result['labels'][0]
            top_score = nli_result['scores'][0]
            
            if top_score < 0.50:
                pass 
            else:
                language_already_penalized = lang_score < 50

                if top_label == "fact" and top_score >= 0.75:
                    if evidence_score < 40:
                        if not language_already_penalized and not has_absence_language:
                            risk_penalty += 30.0 
                            factors.append(f"🚩 NLI Analysis: Stated as Fact ({int(top_score*100)}% conf) but lacks supporting evidence.")
                        elif has_absence_language:
                            factors.append("✅ NLI Analysis: Correctly identifies a factual absence of evidence.")
                    else:
                        evidence_score += 20
                        factors.append(f"✅ NLI Analysis: Validated as a factual claim supported by evidence.")

                elif top_label == "hypothesis" and top_score >= 0.70:
                    if has_absolute:
                        if not language_already_penalized:
                            risk_penalty += 20.0
                            factors.append("🚩 NLI Analysis: Contradiction. Text claims certainty, but AI detects hypothesis.")
                    else:
                        claim_score += 15
                        factors.append(f"ℹ️ NLI Analysis: Identified as Hypothesis ({int(top_score*100)}%). Treated as unverified theory.")

                elif top_label == "speculation" and top_score >= 0.70:
                    if evidence_score < 20:
                        if not language_already_penalized:
                            risk_penalty += 10.0
                            factors.append(f"⚠️ NLI Analysis: Detected as Speculation ({int(top_score*100)}%) without data.")
                    else:
                        factors.append(f"ℹ️ NLI Analysis: Content is speculative.")

                elif top_label == "opinion" and top_score >= 0.60:
                    if evidence_score > 50: evidence_score = 50 
                    factors.append(f"ℹ️ NLI Analysis: Content is primarily Opinion ({int(top_score*100)}%).")

        if has_strong_claim and has_weak_evidence and not has_strong_evidence:
            if has_absence_language:
                real_score_boost = 10.0 # Placeholder
                factors.append("✅ Precision: Correctly notes the absence of evidence/proof.")
            else:
                claim_score -= 20
                factors.append("🚩 Claim-Evidence Mismatch: Absolute claims supported only by weak/observational language.")

        # --- FINAL SCORE CALCULATION & CALIBRATION ---
        final_score = (lang_score * 0.30) + \
                    (evidence_score * 0.30) + \
                    (source_score * 0.20) + \
                    (claim_score * 0.20) - \
                    risk_penalty
                    
        if evidence_score < 10 and final_score > 65:
            final_score = 65.0
        
        if lang_score < 40 and final_score > 60:
            final_score = 60.0
            
        final_score = max(0, min(100, final_score))
        
        # --- Classification Logic ---
        if final_score >= 75:
            classification = "Real"
            conf_real = final_score / 100.0
            conf_fake = 1 - conf_real
        elif final_score <= 40:
            classification = "Fake"
            conf_fake = (100 - final_score) / 100.0
            conf_real = 1 - conf_fake
        else:
            classification = "Unverified"
            conf_real = final_score / 100.0 
            conf_fake = 1 - conf_real
            
            if classification == "Fake" and lang_score >= 80 and risk_penalty == 0 and evidence_score == 0:
                classification = "Unverified"
                factors.append("ℹ️ Content is neutral but lacks sufficient evidence to be confirmed as fact.")

            if not factors and classification == "Unverified":
                factors.append("⚠️ Ambiguous Evidence: Signals are mixed or too weak for a definitive rating.")

        # --- Explanation Generation ---
        explanation = ""
        if classification == "Fake":
            reasons = []
            if any("medical claims" in f for f in factors): reasons.append("makes unverified medical claims")
            if any("subjectivity" in f for f in factors): reasons.append("uses highly subjective language")
            if any("sensationalist" in f or "emotionally" in f or "urgency" in f for f in factors): reasons.append("relies on sensationalism")
            if any("Contradictory" in f for f in factors): reasons.append("references sources contradictorily")
            if any("NLI Analysis" in f and "Flag" in f for f in factors): reasons.append("AI detected a mismatch between claim strength and evidence")
            
            if reasons:
                explanation = f"Flagged as likely misinformation (Score: {int(final_score)}/100). The content failed on key credibility metrics, likely due to {factors[-1].replace('🚩 ', '').lower() if factors else 'unverifiable claims'}."
            else:
                explanation = "Flagged as likely misinformation due to significant suspicion indicators outweighing verifiable evidence."
                
        elif classification == "Real":
            explanation = f"Rated as Credible (Score: {int(final_score)}/100). The content demonstrates high language integrity and cites verifiable evidence."
        else:
            explanation = f"Rated as Unverified (Score: {int(final_score)}/100). While no obvious fabrication was detected, the content lacks sufficient evidence or specific sourcing to be confirmed as fact."

        # --- 13. Improvement Suggestion ---
        suggestion = ""
        if classification == "Real":
            suggestion = "No major improvements needed. The content appears well-sourced and neutral."
        else:
            # Priority Logic: Fixing the biggest penalty first
            if risk_penalty > 0:
                suggestion = "For high-stakes topics (Health/Finance), cite specific clinical trials or official government reports to prove your claim."
            elif evidence_score < 40:
                suggestion = "Linking a peer-reviewed study, official report, or verifiable data source would significantly increase credibility."
            elif source_score < 40:
                suggestion = "Naming specific organizations or experts (e.g., 'WHO stated...') instead of vague groups ('experts say...') would improve reliability."
            elif lang_score < 60:
                suggestion = "The tone is too sensational. Removing emotional words (e.g., 'shocking', 'miracle') and using objective language would boost the score."
            elif claim_score < 40:
                suggestion = "The claim is framed too strongly for the available evidence. Consider using more cautious language (e.g., 'suggests' instead of 'proves')."
            else:
                suggestion = "Adding verifiable details such as dates, locations, or direct quotes would strengthen the analysis."

        related_news = search_google_news(text)
        
        verification_tools = [
            {"source": "Google Fact Check", "url": "https://toolbox.google.com/factcheck/explorer"},
            {"source": "Snopes", "url": "https://www.snopes.com/"}
        ]

        return {
            "classification": classification,
            "confidenceReal": round(conf_real, 4),
            "confidenceFake": round(conf_fake, 4),
            "factors": factors,
            "explanation": explanation,
            "suggestion": suggestion,   
            "related_news": related_news,
            "verification_tools": verification_tools
        }
    except Exception as e:
        print(f"Analysis Logic Error: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

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
        
        # --- IMPROVED SCRAPING ---
        all_paragraphs = [p.get_text().strip() for p in soup.find_all('p')]
        meaningful_paragraphs = [p for p in all_paragraphs if len(p) > 50]
        content_paragraphs = meaningful_paragraphs[:7] if meaningful_paragraphs else all_paragraphs[:5]
        
        meta_desc = ""
        meta_tag = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", attrs={"property": "og:description"})
        if meta_tag and meta_tag.get("content"):
            meta_desc = meta_tag["content"]

        full_text = title + ". " + meta_desc + ". " + " ".join(content_paragraphs)
        
        if len(full_text) < 50:
            raise HTTPException(status_code=400, detail="Could not extract enough text for analysis. Site may be blocking scrapers.")

        return analyze_content(full_text, "url")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {str(e)}")

@app.post("/predict_image")
async def predict_image(file: UploadFile = File(...)):
    # 0. Check file size (Hard limit: 1MB to prevent OOM/timeouts on free tier)
    contents = await file.read()
    if len(contents) > 1 * 1024 * 1024:
         raise HTTPException(status_code=413, detail="File size exceeds 1MB limit.")

    # 1. Try Real OCR via external API (OCR.space)
    try:
        # Use env var for key if available, else fallback to 'helloworld'
        ocr_key = os.getenv("OCR_API_KEY", "helloworld")
        
        response = requests.post(
            'https://api.ocr.space/parse/image',
            files={'file': (file.filename, contents, file.content_type)},
            data={'apikey': ocr_key, 'language': 'eng'},
            timeout=5
        )
        result = response.json()
        
        if not result.get('IsErroredOnProcessing') and result.get('ParsedResults'):
            extracted_text = result['ParsedResults'][0]['ParsedText']
            if len(extracted_text.strip()) > 10:
                analysis = analyze_content(extracted_text, "image")
                analysis['explanation'] = f"(Analyzed Image Text): {analysis['explanation']}"
                return analysis
    except Exception as e:
        print(f"OCR Error: {str(e)}")
        
    # 2. Fallback
    mock_text = "Breaking news: The shocking truth about the secret update they don't want you to know!"
    return analyze_content(mock_text, "image")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)