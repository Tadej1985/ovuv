# .github/update_data.py
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

COINCAP_URL = "https://rest.coincap.io/v2/assets"


def make_session() -> requests.Session:
    """
    Requests session with robust retries/backoff for transient DNS/HTTP issues.
    """
    retry = Retry(
        total=5,                # total attempts
        connect=5,              # retry on connection errors (DNS, etc.)
        read=5,
        backoff_factor=1.5,     # 0s, 1.5s, 3s, 4.5s, 6s ...
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods={"GET"},
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)

    s = requests.Session()
    s.headers.update({"User-Agent": "OVUV/1.0 (https://ovuv.io)"})
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def fetch_coins(limit: int = 200, session: requests.Session | None = None) -> pd.DataFrame:
    """
    Fetch top N assets from CoinCap.
    """
    s = session or make_session()
    resp = s.get(COINCAP_URL, params={"limit": limit}, timeout=30)
    resp.raise_for_status()
    data = resp.json().get("data", [])
    if not data:
        raise RuntimeError("CoinCap returned no 'data'")
    return pd.DataFrame(data)


def zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series([0] * len(series), index=series.index)
    return (series - series.mean()) / std


def compute_scores(limit: int = 200) -> dict:
    """
    Returns a dict with:
      - updated: timestamp (UTC)
      - currency: "USD"
      - top_undervalued: list of 25 rows (most undervalued)
      - top_overvalued: list of 25 rows (most overvalued)
    """
    session = make_session()
    df = fetch_coins(limit=limit, session=session)

    # Convert numeric columns
    for col in ["rank", "priceUsd", "marketCapUsd", "volumeUsd24Hr", "changePercent24Hr"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["rank", "priceUsd", "marketCapUsd", "volumeUsd24Hr", "changePercent24Hr"])
    df = df[df["marketCapUsd"].astype(float) > 0]  # avoid division by zero

    # Core metrics
    df["current_price"] = df["priceUsd"].astype(float)
    df["market_cap"] = df["marketCapUsd"].astype(float)
    df["volume_24h"] = df["volumeUsd24Hr"].astype(float)
    df["momentum_24h"] = df["changePercent24Hr"].astype(float)

    # Value ratio: volume relative to market cap
    df["value_ratio"] = df["volume_24h"] / df["market_cap"]

    # Z-score components
    df["value_component"] = zscore(df["value_ratio"])
    df["momentum_component"] = zscore(-df["momentum_24h"])  # lower momentum => more undervalued

    # Composite score (tweak weights if you want)
    df["undervalue_score"] = 0.6 * df["value_component"] + 0.4 * df["momentum_component"]

    # Most undervalued: highest score
    df_under = df.sort_values("undervalue_score", ascending=False).reset_index(drop=True)
    df_under["rank"] = df_under.index + 1

    # Most overvalued: lowest score
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
        "undervalue_score",
    ]

    top_undervalued = (
        df_under.head(25)[keep_cols]
        .assign(symbol=lambda d: d["symbol"].str.upper())
        .to_dict(orient="records")
    )

    top_overvalued = (
        df_over.head(25)[keep_cols]
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
            f"Updated {len(data['top_undervalued'])} undervalued "
            f"and {len(data['top_overvalued'])} overvalued coins at {data['updated']}"
        )
    except Exception as e:
        # Graceful fallback: keep previous data if fetch fails (e.g., DNS hiccup)
        print(f"WARNING: fetch failed: {e!r}")
        if out_path.exists():
            print("Keeping existing docs/data.json (no update).")
            # Exit 0 so the workflow doesn't go red just because of a temporary network issue.
            return
        else:
            # First run and no data available: write a minimal placeholder so the site loads.
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
