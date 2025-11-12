import os
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import Gemini SDK components
from google import genai
from google.genai import types
from google.genai.errors import APIError

# --- Configuration ---

# 1. Gemini API Key: Set this environment variable securely: 
#    export GEMINI_API_KEY="YOUR_GEMINI_SECRET_KEY"
GEMINI_CLIENT = None
try:
    GEMINI_CLIENT = genai.Client()
except Exception as e:
    print(f"Warning: Could not initialize Gemini Client. Check GEMINI_API_KEY environment variable. Error: {e}")

# 2. CoinCap V3 API Key and Base URL:
#    Set this environment variable: export COINCAP_API_KEY="YOUR_COINCAP_SECRET_KEY"
COINCAP_API_BASE = "https://rest.coincap.io/v2" 
COINCAPSECRET = os.environ.get("COINCAPSECRET")

# Set the Authorization header for CoinCap V3 requests
if COINCAPSECRET:
    HEADERS = {
        'Authorization': f'Bearer {COINCAPSECRET}' 
    }
else:
    HEADERS = {}
    print("WARNING: COINCAPSECRET environment variable not set. V3 API requests will likely fail.")

OUTPUT_FILE = "docs/data.json"
TOP_N = 50  # Number of coins to process from CoinCap's top list

# --- Helper Functions for Data Retrieval (Using Headers) ---

def fetch_coincap_assets(limit=TOP_N):
    """Fetches the top N assets from CoinCap API using the Authorization header."""
    print(f"Fetching top {limit} assets from CoinCap V3...")
    try:
        response = requests.get(
            f"{COINCAP_API_BASE}/assets?limit={limit}",
            headers=HEADERS  # Pass the Authorization header here
        )
        response.raise_for_status()
        data = response.json().get('data', [])
        return pd.DataFrame(data)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching CoinCap V3 assets. Check your API key and rate limits: {e}")
        return pd.DataFrame()

def fetch_historical_data(coin_id, days):
    """
    Fetches daily historical prices for a coin over a specified number of days.
    Returns: (Historical Price from N days ago, list of daily returns)
    """
    # Calculate timestamps for the last 'days' interval
    end_time_ms = int(datetime.now().timestamp() * 1000)
    start_time_ms = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    
    # Use daily interval (d1) for smooth data
    url = f"{COINCAP_API_BASE}/assets/{coin_id}/history?interval=d1&start={start_time_ms}&end={end_time_ms}"
    
    try:
        response = requests.get(url, headers=HEADERS) # Pass the Authorization header here
        response.raise_for_status()
        history = response.json().get('data', [])
        
        if len(history) < 2:
            return None, [] # Not enough data
            
        # Historical price is the first one fetched (oldest)
        historical_price = float(history[0]['priceUsd'])
        
        # Calculate daily returns for volatility: (P_i / P_{i-1}) - 1
        prices = [float(h['priceUsd']) for h in history if h.get('priceUsd')]
        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        
        return historical_price, returns
        
    except requests.exceptions.RequestException as e:
        # Note: CoinCap's /history endpoint may have stricter limits.
        print(f"Error fetching historical data for {coin_id}: {e}")
        return None, []


# --- Gemini Batch Processing Function (Rate Limit Optimization) ---

def get_fundamental_scores_batch(candidates: list) -> dict:
    """
    Uses a single Gemini API call to score all candidates, dramatically
    reducing the number of API calls and avoiding the 10 RPM limit.
    """
    if not GEMINI_CLIENT:
        return {c['id']: 3.0 for c in candidates}
    
    print(f"\n--- Starting Gemini Fundamental Batch Analysis for {len(candidates)} coins (1 Request) ---")

    coin_list_text = "\n".join([f"- {c['name']} ({c['symbol']}, ID: {c['id']})" for c in candidates])

    prompt = (
        f"You are a rational, expert cryptocurrency analyst. Your task is to perform a fundamental "
        f"analysis on the following {len(candidates)} crypto projects. For each project, you must "
        f"search for and summarize the latest development, partnership, and core team news "
        f"from the **last 8 hours**. "
        f"Then, rate the project's **long-term fundamental strength** "
        f"on a scale from 1.0 (Very Weak/Negative News) to 5.0 (Very Strong/Positive News) "
        f"based ONLY on the fresh news.\n\n"
        f"The projects to analyze are:\n{coin_list_text}\n\n"
        f"**CRITICAL OUTPUT INSTRUCTIONS:**\n"
        f"1. You must output ONLY a single JSON object."
        f"2. The JSON object must map the lowercase Coin ID (e.g., 'bitcoin') to its numerical rating (1.0 to 5.0)."
        f"3. Do NOT include any other text, description, or markdown outside the final JSON object.\n\n"
        f"EXAMPLE OUTPUT: {{\"bitcoin\": 4.5, \"ethereum\": 3.8}}"
    )
    
    try:
        response = GEMINI_CLIENT.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                tools=[types.Tool(google_search=types.GoogleSearch())],
            ),
        )
        
        results = json.loads(response.text.strip())
        
        final_scores = {}
        for candidate in candidates:
            coin_id = candidate['id']
            score = results.get(coin_id, 3.0) 
            try:
                final_scores[coin_id] = max(1.0, min(5.0, float(score)))
            except (ValueError, TypeError):
                final_scores[coin_id] = 3.0 
                
        print("--- Gemini Batch Analysis Complete ---")
        return final_scores

    except (APIError, json.JSONDecodeError, Exception) as e:
        print(f"   [FATAL GEMINI BATCH ERROR] Failed to score batch. Error: {e!r}. Using neutral scores (3.0).")
        return {c['id']: 3.0 for c in candidates}

# --- Main Scoring Logic ---

def compute_all_scores():
    """Main function to fetch data, compute scores, and save results."""
    df = fetch_coincap_assets()
    if df.empty:
        print("Exiting: Could not fetch data.")
        return

    # Convert numeric fields and prepare new columns
    df['marketCapUsd'] = pd.to_numeric(df['marketCapUsd'], errors='coerce')
    df['volumeUsd24Hr'] = pd.to_numeric(df['volumeUsd24Hr'], errors='coerce')
    df['priceUsd'] = pd.to_numeric(df['priceUsd'], errors='coerce')
    df['changePercent24Hr'] = pd.to_numeric(df['changePercent24Hr'], errors='coerce')
    
    # Initialize new columns
    df['vol_mcap_ratio'] = 0.0
    df['change_7d'] = 0.0
    df['volatility_30d'] = 0.0
    df['fundamental_score'] = 3.0 
    df['undervalue_score'] = 0.0

    # 1. Calculate Core Financial Metrics & Historical Data
    print("Calculating financial and historical metrics...")
    
    # Volume/Market Cap Ratio (Value Component)
    df['vol_mcap_ratio'] = (df['volumeUsd24Hr'] / df['marketCapUsd']) * 100
    
    # Calculate 7d/30d change and 30d Volatility
    for index, row in df.iterrows():
        
        # --- Fetch Historical Data ---
        price_7d_ago, _ = fetch_historical_data(row['id'], 7)
        price_30d_ago, returns_30d = fetch_historical_data(row['id'], 30)

        # 7-Day Change
        if row['priceUsd'] is not None and price_7d_ago is not None and price_7d_ago != 0:
            df.loc[index, 'change_7d'] = ((row['priceUsd'] - price_7d_ago) / price_7d_ago) * 100
        
        # 30-Day Volatility (Standard Deviation of Daily Returns)
        if len(returns_30d) > 1:
            df.loc[index, 'volatility_30d'] = np.std(returns_30d) * 100
        else:
            df.loc[index, 'volatility_30d'] = 0.0 # Default to 0 if not enough data


    # 2. Identify Gemini Candidates (Top 25 Undervalued + Top 25 Overvalued)
    
    # Standardize the Volume/Market Cap Ratio
    df['z_vol_mcap'] = (df['vol_mcap_ratio'] - df['vol_mcap_ratio'].mean()) / df['vol_mcap_ratio'].std()
    
    # Standardize Volatility-Adjusted Momentum (New Component)
    # Volatility-Adjusted Momentum: Lower price change relative to volatility is better for undervalue
    df['vol_adj_momentum'] = df['change_7d'] / (df['volatility_30d'] + 0.01) 
    df['z_vol_adj_momentum'] = (df['vol_adj_momentum'] - df['vol_adj_momentum'].mean()) / df['vol_adj_momentum'].std()

    # Create a composite score to select candidates for Gemini analysis
    df['candidate_score'] = df['z_vol_mcap'] - df['z_vol_adj_momentum'] 
    
    candidates_undervalue = df.sort_values(by='candidate_score', ascending=False).head(25)
    candidates_overvalue = df.sort_values(by='candidate_score', ascending=True).head(25)
    
    analysis_candidates = pd.concat([candidates_undervalue, candidates_overvalue]).drop_duplicates(subset=['id'])
    analysis_candidates_list = analysis_candidates[['id', 'name', 'symbol']].to_dict('records')
    

    # 3. Batch Call Gemini API
    fundamental_scores_map = get_fundamental_scores_batch(analysis_candidates_list)
    
    # Update DataFrame using the map (efficiently)
    df['fundamental_score'] = df['id'].map(fundamental_scores_map).fillna(df['fundamental_score'])


    # 4. Final Undervalue Score Calculation
    
    # Standardize the Final Fundamental Score
    df['z_fundamental'] = (df['fundamental_score'] - df['fundamental_score'].mean()) / df['fundamental_score'].std()

    # Undervalue Score (The Ranking Metric)
    # Weights can be adjusted 
    W_VOL_MCAP = 0.35
    W_VOL_MOM = 0.35
    W_FUNDAMENTAL = 0.30

    df['undervalue_score'] = (
        W_VOL_MCAP * df['z_vol_mcap'] 
        - W_VOL_MOM * df['z_vol_adj_momentum'] # Subtract momentum for the undervalue calculation
        + W_FUNDAMENTAL * df['z_fundamental']
    )

    # Sort and finalize the dataset
    final_df = df.sort_values(by='undervalue_score', ascending=False).head(100)
    
    # Select and rename final columns for the output JSON
    final_data = final_df[[
        'rank', 'symbol', 'name', 'priceUsd', 'marketCapUsd', 
        'volumeUsd24Hr', 'changePercent24Hr', 'vol_mcap_ratio', 
        'change_7d', 'volatility_30d', 'fundamental_score', 'undervalue_score'
    ]].copy()
    
    final_data.columns = [
        'rank', 'symbol', 'name', 'price', 'market_cap', 
        'volume_24h', 'price_change_24h', 'vol_mcap_ratio', 
        'price_change_7d', 'volatility_30d', 'fundamental_score', 'undervalue_score'
    ]
    
    # Round numerical data for clean JSON output
    for col in ['price', 'market_cap', 'volume_24h', 'price_change_24h', 
                'vol_mcap_ratio', 'price_change_7d', 'volatility_30d', 
                'fundamental_score', 'undervalue_score']:
        if col in ['market_cap', 'volume_24h']:
             final_data[col] = final_data[col].apply(lambda x: int(round(x)))
        else:
             final_data[col] = final_data[col].round(2)


    # 5. Save to JSON File
    output_dict = {
        "generated_at": datetime.now().isoformat(),
        "data": final_data.to_dict('records')
    }
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output_dict, f, indent=4)
        
    print(f"\n✅ Successfully generated {len(final_data)} scores and saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    compute_all_scores()
