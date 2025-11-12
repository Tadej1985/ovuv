import os
import json
import requests
import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta

from google import genai
from google.genai import types
from google.genai.errors import APIError

# --- Configuration (No Changes) ---
# ... (Configuration block remains the same) ...

GEMINI_CLIENT = None
try:
    GEMINI_CLIENT = genai.Client()
except Exception as e:
    print(f"Warning: Could not initialize Gemini Client. Check GEMINI_API_KEY environment variable. Error: {e}")

COINCAP_API_BASE = "https://rest.coincap.io/V3" 
COINCAP_KEY = os.environ.get("COINCAPSECRET") 

if COINCAP_KEY:
    HEADERS = {
        'Authorization': f'Bearer {COINCAP_KEY}' 
    }
else:
    HEADERS = {}
    print("WARNING: COINCAPSECRET environment variable not set. V3 API requests will likely fail.") 

OUTPUT_FILE = "docs/data.json"
TOP_N = 200 

# --- Data Retrieval (CoinCap V3 - No Changes) ---

def fetch_coincap_assets(limit=TOP_N):
    """Fetches the top N assets from CoinCap API using the Authorization header."""
    print(f"Fetching top {limit} assets from CoinCap V3...")
    try:
        response = requests.get(
            f"{COINCAP_API_BASE}/assets?limit={limit}", 
            headers=HEADERS
        )
        response.raise_for_status()
        data = response.json().get('data', [])
        return pd.DataFrame(data)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching CoinCap V3 assets. Check your API key and rate limits: {e}")
        return pd.DataFrame()


# --- Gemini Batch Function 1: Triage and Initial Scoring (NO Search Tool) ---

def triage_coins(all_coins_df: pd.DataFrame) -> dict:
    """
    Triage 200 coins and assign initial scores based on model's knowledge only.
    This avoids the 'TOO_MANY_TOOL_CALLS' error.
    """
    if not GEMINI_CLIENT:
        return {'undervalued': [], 'overvalued': []}
    
    all_coins_list = all_coins_df[['id', 'name', 'symbol', 'marketCapUsd', 'priceUsd']].copy()
    all_coins_list['marketCapUsd'] = pd.to_numeric(all_coins_list['marketCapUsd'], errors='coerce').round(0).fillna(0)
    all_coins_list['priceUsd'] = pd.to_numeric(all_coins_list['priceUsd'], errors='coerce').round(4).fillna(0.0)
    
    coin_data_text = "\n".join([
        f"- {r['name']} ({r['symbol']}, ID: {r['id']}) - MarketCap: ${r['marketCapUsd']:,} - Price: ${r['priceUsd']}" 
        for _, r in all_coins_list.iterrows()
    ])

    print(f"\n--- Starting Gemini Triage and Initial Scoring for {len(all_coins_df)} coins (1st Call) ---")

    prompt = (
        f"You are an expert crypto analyst. Analyze the following {len(all_coins_df)} projects. "
        f"**Do not use any external tools or perform web searches.** Your task is to select 25 Undervalued and 25 Overvalued candidates and assign a fundamental score (1.0 to 5.0) based purely on your knowledge of the project's long-term potential.\n\n"
        f"**Selection Rules:**\n"
        f"1. **Select 25 Undervalued Candidates:** Coins with strong long-term narratives and stable market positions (outside the top 5 is usually a good bet).\n"
        f"2. **Select 25 Overvalued Candidates:** Coins with weak long-term narratives or high risk based on past events/known issues.\n\n"
        f"**CRITICAL OUTPUT INSTRUCTIONS:**\n"
        f"1. Output ONLY a single JSON object with keys `undervalued` and `overvalued`.\n"
        f"2. Each key must contain a list of objects.\n"
        f"3. Each object must contain two keys: `id` (string) and `fundamental_score` (float).\n\n"
        f"Coin Data:\n{coin_data_text}"
    )
    
    try:
        response = GEMINI_CLIENT.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt],
            # NO tools=... to avoid the TOO_MANY_TOOL_CALLS error
        )
        
        raw_text = response.text.strip()
        json_match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", raw_text, re.DOTALL)
        
        json_str = json_match.group(1).strip() if json_match else raw_text
        if not json_str:
             raise ValueError("Gemini returned an empty or unparsable response.")

        results = json.loads(json_str)
        print("--- Gemini Triage Complete ---")
        return results

    except (APIError, json.JSONDecodeError, Exception) as e:
        print(f"   [FATAL GEMINI TRIAGE ERROR] Failed to perform triage. Error: {e!r}. Returning empty list.")
        return {'undervalued': [], 'overvalued': []}


# --- Gemini Batch Function 2: Research and Summary (WITH Search Tool) ---

def research_candidates(candidates: list) -> dict:
    """
    Performs in-depth research and generates a summary for the selected candidates.
    """
    if not GEMINI_CLIENT or not candidates:
        return {}
    
    print(f"\n--- Starting Gemini In-Depth Research for {len(candidates)} candidates (2nd Call) ---")
    
    candidate_list_text = "\n".join([f"- {c['id']} (Current Score: {c['fundamental_score']})" for c in candidates])
    
    prompt = (
        f"You are an expert crypto market analyst. Your task is to perform in-depth Google Search on the following {len(candidates)} selected crypto projects.\n\n"
        f"For each project, you must find and summarize the latest news (partnerships, team activity, regulatory status) from the **last 48 hours**. \n\n"
        f"**CRITICAL OUTPUT INSTRUCTIONS:**\n"
        f"1. Output ONLY a single JSON object. Do NOT include any other text.\n"
        f"2. The JSON object must map the lowercase Coin ID (e.g., 'bitcoin') to its summary string.\n"
        f"3. The summary must be a brief, 1-2 sentence description of the research findings.\n\n"
        f"Candidates to Research:\n{candidate_list_text}\n\n"
        f"EXAMPLE OUTPUT: {{\"solana\": \"New cross-chain bridge launched to Polygon, boosting total locked value.\", \"dogecoin\": \"Elon Musk tweet about new payment integration caused a temporary price spike.\"]}}"
    )
    
    try:
        response = GEMINI_CLIENT.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt],
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())], # Tool use is fine here, as the list is small
            ),
        )
        
        raw_text = response.text.strip()
        json_match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", raw_text, re.DOTALL)
        
        json_str = json_match.group(1).strip() if json_match else raw_text
        if not json_str:
             raise ValueError("Gemini returned an empty or unparsable response.")

        results = json.loads(json_str)
        print("--- Gemini Research Complete ---")
        return results

    except (APIError, json.JSONDecodeError, Exception) as e:
        print(f"   [FATAL GEMINI RESEARCH ERROR] Failed to perform research. Error: {e!r}. Skipping summaries.")
        return {}


# --- Main Scoring Logic ---

def compute_all_scores():
    """Main function to fetch data, perform AI analysis, and save results."""
    # 1. Fetch 200 coins
    df = fetch_coincap_assets()
    if df.empty:
        print("Exiting: Could not fetch data.")
        return

    # 2. Gemini Call 1: Triage and Score (No Search)
    ai_triage_results = triage_coins(df)
    
    # Combine the undervalued and overvalued lists from the triage
    all_candidates = ai_triage_results.get('undervalued', []) + ai_triage_results.get('overvalued', [])

    if not all_candidates:
        print("Exiting: AI triage returned no candidates.")
        return
    
    # 3. Gemini Call 2: Research and Summarize (With Search)
    summary_map = research_candidates(all_candidates)

    # 4. Merge Results and Prepare Final Output
    final_data_list = []
    
    for candidate in all_candidates:
        coin_id = candidate['id']
        
        # Find the matching CoinCap data row
        coin_row = df[df['id'] == coin_id]
        if coin_row.empty:
            continue
            
        coin_data = coin_row.iloc[0].to_dict()
        
        # Get the summary from the research map, default to N/A
        summary = summary_map.get(coin_id, "N/A (Could not generate research summary.)")
        
        final_data_list.append({
            'rank': coin_data.get('rank', 'N/A'),
            'symbol': coin_data.get('symbol', 'N/A'),
            'name': coin_data.get('name', 'N/A'),
            'price': round(float(coin_data.get('priceUsd', 0)), 4),
            'market_cap': int(round(float(coin_data.get('marketCapUsd', 0)))),
            'price_change_24h': round(float(coin_data.get('changePercent24Hr', 0)), 2),
            'fundamental_score': candidate['fundamental_score'],
            'summary': summary,
            'category': 'Undervalued' if candidate in ai_triage_results.get('undervalued', []) else 'Overvalued'
        })

    # 5. Final Sorting and JSON Output
    final_df = pd.DataFrame(final_data_list)
    
    # Sort logic to show Undervalued first, then by score
    final_df['sort_key'] = final_df['category'].apply(lambda x: 0 if x == 'Undervalued' else 1)
    final_df = final_df.sort_values(by=['sort_key', 'fundamental_score'], 
                                    ascending=[True, False]).drop(columns=['sort_key'])

    output_dict = {
        "generated_at": datetime.now().isoformat(),
        "data": final_df.to_dict('records')
    }
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output_dict, f, indent=4)
        
    print(f"\n✅ Successfully generated {len(final_df)} highly-vetted scores and saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    compute_all_scores()
