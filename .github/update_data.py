import os
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Import Gemini Libraries
import google.genai as genai
from google.genai import types 
from google.genai.errors import APIError

# --- CONFIGURATION ---
# CoinCap API URL (Note: CoinCap v2 or v3 typically requires a Bearer token in headers, 
# but the provided URL is a common REST endpoint)
# NOTE: The provided CoinCap URL is for v3 and is used as-is.
COINCAP_URL = "https://rest.coincap.io/v3/assets"
COINCAP_API_KEY = os.getenv("COINCAPSECRET", "")

# Gemini Client
try:
    # The client automatically picks up the GEMINI_API_KEY environment variable.
    gemini_client = genai.Client()
except Exception as e:
    print(f"Warning: Could not initialize Gemini Client. Check GEMINI_API_KEY environment variable. Error: {e}")
    gemini_client = None

# Scoring Weights (Total must equal 1.0)
VALUE_WEIGHT = 0.4
MOMENTUM_WEIGHT = 0.3
FUNDAMENTAL_WEIGHT = 0.3
# Number of coins to analyze with Gemini (25 Undervalued + 25 Overvalued)
ANALYSIS_LIMIT = 25 
# --- END CONFIGURATION ---


def make_session() -> requests.Session:
    """Requests session with retries/backoff for transient network issues."""
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods={"GET"},
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s = requests.Session()
    s.headers.update({
        "User-Agent": "CryptoScoringBot/1.0",
        "Authorization": f"Bearer {COINCAP_API_KEY}" # CoinCap requires API Key in the Authorization header
    })
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def fetch_coins(limit: int = 200, session: requests.Session | None = None) -> pd.DataFrame:
    """Fetch top N assets from CoinCap v3."""
    if not COINCAP_API_KEY:
        raise RuntimeError("COINCAPSECRET is not set. Cannot fetch data.")

    s = session or make_session()
    params = {
        "limit": limit,
    }
    resp = s.get(COINCAP_URL, params=params, timeout=30)
    resp.raise_for_status()
    body = resp.json()
    data = body.get("data", [])
    if not data:
        raise RuntimeError(f"CoinCap returned no data: {body!r}")
    return pd.DataFrame(data)


def zscore(series: pd.Series) -> pd.Series:
    """Calculate the Z-Score for a pandas Series."""
    std = series.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series([0] * len(series), index=series.index)
    # Z-score formula: (X - mu) / sigma
    return (series - series.mean()) / std


def get_fundamental_score(coin_name: str, symbol: str) -> float:
    """
    Uses the Gemini API with Google Search Grounding to generate a fundamental score.
    Returns a score between 1.0 and 5.0.
    """
    if not gemini_client:
        return 3.0 # Return neutral if client not initialized

    prompt = (
        f"Search for and summarize the latest development, partnership, and core team news "
        f"for the crypto project: {coin_name} ({symbol}) from the **last 8 hours**. "
        f"Then, rate the project's **long-term fundamental strength** "
        f"on a scale from 1.0 (Very Weak/Negative News) to 5.0 (Very Strong/Positive News) "
        f"based ONLY on the fresh news. Output ONLY the numerical score."
    )
    
    # 🌟 FIX: The system instruction MUST be inside GenerateContentConfig
    system_prompt = "You are a rational, expert cryptocurrency analyst. Your task is to output ONLY a single numerical rating from 1.0 to 5.0."

    try:
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt],
            config=types.GenerateContentConfig(
                # System instruction is correctly placed here to resolve the error
                system_instruction=system_prompt,
                # CRITICAL: This enables Google Search Grounding for real-time data (1 RPD)
                tools=[types.Tool(google_search=types.GoogleSearch())],
            ),
        )
        
        # Attempt to clean and parse the response text into a float
        score_text = response.text.strip()
        # Find the first number (float) in the text
        import re
        match = re.search(r'\d+\.?\d*', score_text)
        if match:
            score = float(match.group(0))
        else:
            raise ValueError("Model output did not contain a recognizable number.")

        # Ensure the score is within the defined range
        return max(1.0, min(5.0, score))
        
    except (APIError, ValueError) as e:
        print(f"   [GEMINI ERROR] Failed to score {symbol}. Error: {e!r}. Using neutral score.")
        return 3.0 # Neutral score on API failure


def compute_scores(limit: int = 200) -> dict:
    """Compute all scores and lists."""
    session = make_session()
    df = fetch_coins(limit=limit, session=session)

    # Convert columns to numeric and drop NaNs
    for col in ["rank", "priceUsd", "marketCapUsd", "volumeUsd24Hr", "changePercent24Hr"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["rank", "priceUsd", "marketCapUsd", "volumeUsd24Hr", "changePercent24Hr"])
    df = df[df["marketCapUsd"] > 0]

    # Standardize column names
    df["current_price"] = df["priceUsd"].round(4)
    df["market_cap"] = df["marketCapUsd"].astype(float)
    df["volume_24h"] = df["volumeUsd24Hr"].astype(float)
    df["momentum_24h"] = df["changePercent24Hr"].astype(float).round(2)
    df["value_ratio"] = df["volume_24h"] / df["market_cap"]

    # 1. QUANTITATIVE COMPONENTS (Z-Scores)
    df["value_component"] = zscore(df["value_ratio"])
    # Negative change is a sign of undervalue, so we invert the Z-score for momentum
    df["momentum_component"] = zscore(-df["momentum_24h"]) 

    # 2. FUNDAMENTAL COMPONENT (Gemini Analysis)
    df["fundamental_score"] = 3.0 # Default neutral score (1.0 to 5.0 range)
    print("\n--- Starting Gemini Fundamental Analysis ---")
    
    # Calculate a simple initial score to identify top candidates for analysis
    df["initial_score"] = 0.5 * df["value_component"] + 0.5 * df["momentum_component"]
    
    # Select the top 25 for Undervalue and bottom 25 for Overvalue based on initial score
    df_temp = df.sort_values("initial_score", ascending=False)
    undervalue_candidates = df_temp.head(ANALYSIS_LIMIT)
    overvalue_candidates = df_temp.tail(ANALYSIS_LIMIT)

    # Combine candidates to run Gemini on (max 50 requests total)
    analysis_indices = pd.concat([undervalue_candidates, overvalue_candidates]).index.unique()
    
    for index in analysis_indices:
        row = df.loc[index]
        print(f"-> Analyzing {row['name']} ({row['symbol']})...")
        
        # Get score using Gemini
        gemini_score = get_fundamental_score(row["name"], row["symbol"])
        df.loc[index, "fundamental_score"] = gemini_score
        print(f"   [RESULT] Score: {gemini_score:.1f}")

    print("--- Gemini Analysis Complete ---\n")
    
    # 3. FINAL COMPOSITE SCORE
    # Normalize the 1.0-5.0 raw score into a Z-Score for combination
    df["fundamental_component"] = zscore(df["fundamental_score"])
    
    # Calculate the weighted final score
    df["undervalue_score"] = (
        VALUE_WEIGHT * df["value_component"] +
        MOMENTUM_WEIGHT * df["momentum_component"] +
        FUNDAMENTAL_WEIGHT * df["fundamental_component"]
    )

    # Sort results for final lists
    df_under = df.sort_values("undervalue_score", ascending=False).reset_index(drop=True)
    df_under["rank"] = df_under.index + 1

    df_over = df.sort_values("undervalue_score", ascending=True).reset_index(drop=True)
    df_over["rank"] = df_over.index + 1

    keep_cols = [
        "rank",
        "id",
        "name",
        "symbol",
        "current_price",
        "market_cap",
        "volume_24h",
        "value_ratio",
        "momentum_24h",
        "fundamental_score", # Raw 1-5 score from Gemini
        "undervalue_score",
    ]

    top_undervalued = (
        df_under.head(ANALYSIS_LIMIT)[keep_cols]
        .assign(symbol=lambda d: d["symbol"].str.upper())
        .to_dict(orient="records")
    )

    top_overvalued = (
        df_over.head(ANALYSIS_LIMIT)[keep_cols]
        .assign(symbol=lambda d: d["symbol"].str.upper())
        .to_dict(orient="records")
    )

    return {
        "updated": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "currency": "USD",
        "top_undervalued": top_undervalued,
        "top_overvalued": top_overvalued,
    }


def main():
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    out_path = docs_dir / "data.json"

    try:
        data = compute_scores(limit=200)
        out_path.write_text(json.dumps(data, indent=2))
        print(
            f"Successfully updated {len(data['top_undervalued'])} undervalued "
            f"and {len(data['top_overvalued'])} overvalued coins at {data['updated']}"
        )
    except Exception as e:
        print(f"ERROR while updating data: {e!r}")
        if out_path.exists():
            print("Keeping existing docs/data.json (no update).")
        else:
            placeholder = {
                "updated": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
                "currency": "USD",
                "top_undervalued": [],
                "top_overvalued": [],
                "error": "fetch_failed",
            }
            out_path.write_text(json.dumps(placeholder, indent=2))
            print("Wrote placeholder docs/data.json due to fetch failure.")


if __name__ == "__main__":
    main()
