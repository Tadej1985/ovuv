import os
import json
import requests
import pandas as pd
import numpy as np
import re # <-- Added Regular Expression import for robust JSON parsing
from datetime import datetime, timedelta

# Import Gemini SDK components
from google import genai
from google.genai import types
from google.genai.errors import APIError

# --- Configuration ---

GEMINI_CLIENT = None
try:
    # Client initialization relies on the GEMINI_API_KEY environment variable
    GEMINI_CLIENT = genai.Client()
except Exception as e:
    print(f"Warning: Could not initialize Gemini Client. Check GEMINI_API_KEY environment variable. Error: {e}")

COINCAP_API_BASE = "https://rest.coincap.io/V3" 
# Key must be named COINCAPSECRET in the environment/GitHub Secrets
COINCAP_KEY = os.environ.get("COINCAPSECRET") 

if COINCAP_KEY:
    HEADERS = {
        'Authorization': f'Bearer {COINCAP_KEY}' 
    }
else:
    HEADERS = {}
    print("WARNING: COINCAPSECRET environment variable not set. V3 API requests will likely fail.") 

OUTPUT_FILE = "docs/data.json"
TOP_N = 200 # <-- PULL 200 COINS FROM COINCAP

# --- Data Retrieval (CoinCap V3) ---

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

# --- Gemini Batch Processing Function (Triage, Research, and Ranking) ---

def get_final_ranked_list(all_coins_df: pd.DataFrame) -> dict:
    """
    Sends all coin data to Gemini in one prompt, asking the model to perform 
    the triage, research, and final ranking, returning a structured dict.
    """
    if not GEMINI_CLIENT:
        print("\nGemini Client is not initialized. Cannot perform analysis.")
        return {'undervalued': [], 'overvalued': []}
    
    # Prepare the list of all 200 coins for the prompt
    all_coins_list = all_coins_df[['id', 'name', 'symbol', 'marketCapUsd', 'priceUsd']].copy()
    
    # Convert numerical columns to float and format them nicely for the prompt
    all_coins_list['marketCapUsd'] = pd.to_numeric(all_coins_list['marketCapUsd'], errors='coerce').round(0).fillna(0)
    all_coins_list['priceUsd'] = pd.to_numeric(all_coins_list['priceUsd'], errors='coerce').round(4).fillna(0.0)
    
    # Create the text input string for Gemini
    coin_data_text = "\n".join([
        f"- {r['name']} ({r['symbol']}, ID: {r['id']}) - MarketCap: ${r['marketCapUsd']:,} - Price: ${r['priceUsd']}" 
        for _, r in all_coins_list.iterrows()
    ])

    print(f"\n--- Starting Gemini Triage, Research, and Ranking Batch Analysis for {len(all_coins_df)} coins (1 Request) ---")

    prompt = (
        f"You are an expert quantitative and fundamental cryptocurrency analyst. Your task is to perform a comprehensive "
        f"screening and analysis on the following {len(all_coins_df)} projects, which are the top 200 by market capitalization.\n\n"
        f"**PHASE 1: Triage and Selection**\n"
        f"1. **Select 25 Undervalued Candidates:** Identify the 25 coins that show the highest potential for growth. Use the initial data (Market Cap, Price) to favor coins outside the top 10 that have low volatility and a strong narrative (you must research this).\n"
        f"2. **Select 25 Overvalued Candidates:** Identify the 25 coins that show the highest risk or downward pressure. Favor projects with high market cap but recent negative news or a weak narrative (you must research this).\n"
        f"3. **Total Candidates:** Your final analyzed list must contain exactly **50** unique coins (25 undervalued, 25 overvalued).\n\n"
        f"**PHASE 2: In-Depth Research and Scoring**\n"
        f"For each of the 50 selected candidates, you must use Google Search to perform an in-depth fundamental analysis, focusing on: latest developments, team activity, new partnerships, and regulatory status from the **last 48 hours**.\n\n"
        f"**CRITICAL OUTPUT INSTRUCTIONS:**\n"
        f"1. You must output ONLY a single JSON object. Do NOT include any other text, preambles, or markdown outside the final JSON. **If you must wrap it, use only the ```json and ``` block.**\n"
        f"2. The JSON object must contain two keys: `undervalued` and `overvalued`.\n"
        f"3. Each key must contain a list of objects (25 items each).\n"
        f"4. Each item must contain the following keys:\n"
        f"   - `id` (string): The lowercase Coin ID (e.g., 'bitcoin').\n"
        f"   - `fundamental_score` (float): Your rating from 1.0 (Worst Fundamentals) to 5.0 (Best Fundamentals).\n"
        f"   - `summary` (string): A brief, 1-2 sentence summary of your research findings.\n\n"
        f"**Coin Data to Triage (Top 200):**\n{coin_data_text}\n\n"
        f"**EXAMPLE OUTPUT STRUCTURE:**\n"
        f"{{\n"
        f"  \"undervalued\": [\n"
        f"    {{\"id\": \"solana\", \"fundamental_score\": 4.5, \"summary\": \"New strategic partnership with Google Cloud announced, driving adoption.\"]}},\n"
        f"    /* ... 24 more undervalued items ... */\n"
        f"  ],\n"
        f"  \"overvalued\": [\n"
        f"    {{\"id\": \"dogecoin\", \"fundamental_score\": 1.2, \"summary\": \"Large whale wallets began liquidating holdings following a major security breach.\"]}},\n"
        f"    /* ... 24 more overvalued items ... */\n"
        f"  ]\n"
        f"}}"
    )
    
    try:
        response = GEMINI_CLIENT.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt],
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
            ),
        )
        
        # --- START HARDENED JSON PARSING FIX ---
        raw_text = response.text.strip()
        
        # Use regex to find the content inside the markdown block (most reliable)
        # Regex explanation: Find ```json followed by optional whitespace, capture everything 
        # up to the closing ```. The use of [\s\S]*? makes it non-greedy across newlines.
        json_match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", raw_text, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            # If no markdown block is found, assume the entire response is the JSON string
            json_str = raw_text
            
        # Handle a completely empty response
        if not json_str:
             raise ValueError("Gemini returned an empty or unparsable response.")

        results = json.loads(json_str)
        # --- END HARDENED JSON PARSING FIX ---
                
        print("--- Gemini Analysis Complete ---")
        return results

    except (APIError, json.JSONDecodeError, Exception) as e:
        print(f"   [FATAL GEMINI BATCH ERROR] Failed to perform triage and scoring. Error: {e!r}. Returning empty list.")
        return {'undervalued': [], 'overvalued': []}


# --- Main Scoring Logic ---

def compute_all_scores():
    """Main function to fetch data, perform AI analysis, and save results."""
    # 1. Fetch 200 coins
    df = fetch_coincap_assets()
    if df.empty:
        print("Exiting: Could not fetch data.")
        return

    # Convert numeric fields
    df['marketCapUsd'] = pd.to_numeric(df['marketCapUsd'], errors='coerce')
    df['priceUsd'] = pd.to_numeric(df['priceUsd'], errors='coerce')
    df['changePercent24Hr'] = pd.to_numeric(df['changePercent24Hr'], errors='coerce')

    # 2. Get AI Triage and Scoring
    ai_results = get_final_ranked_list(df)
    
    # Combine the undervalued and overvalued lists into a single, flat list
    all_final_candidates = ai_results.get('undervalued', []) + ai_results.get('overvalued', [])

    if not all_final_candidates:
        print("Exiting: AI analysis returned no candidates.")
        return

    # 3. Merge AI Results with CoinCap Data and Prepare Final Output
    final_data_list = []
    
    for candidate in all_final_candidates:
        coin_id = candidate['id']
        
        # Find the matching CoinCap data row
        # Use a more robust check for missing IDs
        coin_row = df[df['id'] == coin_id]
        if coin_row.empty:
            print(f"Warning: AI suggested coin ID '{coin_id}' not found in the CoinCap data.")
            continue
            
        coin_data = coin_row.iloc[0].to_dict()
        
        # Create the final output structure
        final_data_list.append({
            'rank': coin_data.get('rank', 'N/A'),
            'symbol': coin_data.get('symbol', 'N/A'),
            'name': coin_data.get('name', 'N/A'),
            'price': round(float(coin_data.get('priceUsd', 0)), 4),
            'market_cap': int(round(float(coin_data.get('marketCapUsd', 0)))),
            'price_change_24h': round(float(coin_data.get('changePercent24Hr', 0)), 2),
            'fundamental_score': candidate['fundamental_score'],
            'summary': candidate['summary'],
            'category': 'Undervalued' if candidate in ai_results.get('undervalued', []) else 'Overvalued'
        })

    # 4. Final Sorting and JSON Output
    final_df = pd.DataFrame(final_data_list)
    
    # Sort by category (Undervalued first) and then by fundamental score
    # Create a sort key: 0 for Undervalued, 1 for Overvalued
    final_df['sort_key'] = final_df['category'].apply(lambda x: 0 if x == 'Undervalued' else 1)
    
    # Sort by key (Undervalued first), then by score (highest score first)
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
