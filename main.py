from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from apify_client import ApifyClient
from urllib.parse import urlparse
from groq import Groq
import re
from collections import Counter
import json

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Apify Client (Groq is initialized inside the endpoint to keep it fresh)
apify_client = ApifyClient(os.getenv("APIFY_API_TOKEN"))

class ProductRequest(BaseModel):
    url: str

# --- HELPER FUNCTIONS ---

def get_product_details(amazon_url: str):
    print("Starting Product Scraper...")
    run_input = {
        "categoryOrProductUrls": [{"url": amazon_url}],
        "maxItemsPerStartUrl": 1,
        "useCaptchaSolver": True
    }
    try:
        run = apify_client.actor("junglee/amazon-crawler").call(run_input=run_input)
        dataset_items = list(apify_client.dataset(run["defaultDatasetId"]).iterate_items())
        
        if not dataset_items:
            return None
            
        product = dataset_items[0]
        return {
            "title": product.get("title", "Unknown Product"),
            "price": product.get("price", 0),
            "asin": product.get("asin", "Unknown ASIN"),
            "competitors": product.get("similarItems", [])[:9]
        }
    except Exception as e:
        print(f"Error scraping product details: {e}")
        return None

def get_product_reviews(asin: str, domain_code: str):
    print(f"Starting Reviews Scraper for ASIN: {asin} on amazon.{domain_code}...")
    
    product_url = f"https://www.amazon.{domain_code}/dp/{asin}"
    
    run_input = {
        "productUrls": [{"url": product_url}],
        "maxReviews": 50,
        "sort": "recent"
    }
    
    try:
        run = apify_client.actor("junglee/amazon-reviews-scraper").call(run_input=run_input)
        dataset_items = list(apify_client.dataset(run["defaultDatasetId"]).iterate_items())
        
        reviews_text = [item.get("reviewDescription", "") for item in dataset_items]
        reviews_text = [text for text in reviews_text if text] 
        return reviews_text
    except Exception as e:
        print(f"Error scraping reviews: {e}")
        return []

# --- MAIN ENDPOINT ---

@app.post("/analyze")
async def analyze_product(request: ProductRequest):
    print(f"\n--- New Request Received ---")
    print(f"URL: {request.url}")
    
    domain_code = urlparse(request.url).netloc.split('amazon.')[-1]
    
    # 1. Scrape listing details
    product_data = get_product_details(request.url)
    if not product_data or not product_data.get("asin"):
        raise HTTPException(status_code=400, detail="Could not extract product data.")
    
    # 2. Scrape reviews 
    reviews = get_product_reviews(product_data["asin"], domain_code)
    if not reviews:
        raise HTTPException(status_code=400, detail="Could not extract reviews. Amazon might be blocking the scraper.")

    # 3. Calculate Revenue Estimate (Bulletproof parsing)
    price_val = product_data.get("price", 0)
    price = 0.0
    
    if isinstance(price_val, dict):
        extracted_val = price_val.get("value", 0)
        try:
            price = float(extracted_val)
        except (ValueError, TypeError):
            price = 0.0
    elif isinstance(price_val, str):
        price_str = price_val.replace("$", "").replace("₹", "").replace("Rs.", "").replace(",", "").strip()
        try:
            price = float(price_str)
        except ValueError:
            price = 0.0
    elif isinstance(price_val, (int, float)):
        price = float(price_val)
        
    estimated_revenue = price * len(reviews) * 75
    
    # 4. Feed reviews to Groq (Llama-3)
    print("Sending data to Groq AI for analysis...")
    
    try:
        groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        # Get Purchase Criteria
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an expert e-commerce analyst. Return ONLY a raw JSON array of 3 strings summarizing the top purchase criteria. No markdown."},
                {"role": "user", "content": f"Analyze these Amazon reviews: {reviews}"}
            ],
            model="llama-3.1-8b-instant",
        )
        purchase_criteria = chat_completion.choices[0].message.content.strip()

        # SMART FALLBACK: Infer market competitors
        competitors = product_data.get("competitors", [])
        if not competitors or len(competitors) == 0:
            print("Scraper missed competitors. Generating AI competitors fallback...")
            comp_completion = groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are an AI market researcher. The user will give you a product name. Return ONLY a raw JSON array of exactly 9 likely competitor product titles or brand alternatives in that specific niche. Keep titles under 40 characters. No markdown."},
                    {"role": "user", "content": f"Generate 9 generic competitors for this product: {product_data.get('title')}"}
                ],
                model="llama-3.1-8b-instant",
            )
            try:
                ai_comps = json.loads(comp_completion.choices[0].message.content.strip())
                competitors = [{"title": comp} for comp in ai_comps]
            except Exception:
                competitors = [{"title": f"Competitor Brand {i+1} Equivalent"} for i in range(9)]
        else:
            competitors = competitors[:9]
            
    except Exception as e:
        print(f"Groq API failed: {e}. Pivoting to local NLP fallback...")
        
        # ACTUAL FALLBACK: Pure Python text processing!
        if reviews:
            all_text = " ".join(reviews).lower()
            words = re.findall(r'\b[a-z]{4,}\b', all_text)
            stopwords = {'this', 'that', 'with', 'have', 'from', 'very', 'just', 'they', 'what', 'good', 'product', 'backpack', 'bag', 'bags', 'quality', 'colour', 'color', 'capacity', 'overall', 'like', 'really'}
            filtered = [w for w in words if w not in stopwords]
            
            top_words = [word for word, count in Counter(filtered).most_common(3)]
            
            if len(top_words) >= 3:
                purchase_criteria = f'["Mentions of: {top_words[0].title()}", "Mentions of: {top_words[1].title()}", "Mentions of: {top_words[2].title()}"]'
            else:
                purchase_criteria = '["Not enough review data"]'
        else:
            purchase_criteria = '["No reviews scraped"]'

        competitors = [{"title": f"Market Alternative {i+1}"} for i in range(9)]

    print("Analysis complete! Sending to frontend.")
    return {
        "status": "success",
        "product_title": product_data.get("title", "Unknown Product"),
        "estimated_revenue": estimated_revenue,
        "competitors": competitors, 
        "purchase_criteria": purchase_criteria
    }