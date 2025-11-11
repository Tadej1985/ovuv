import os
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ✅ v3 base URL
COINCAP_URL = "https://rest.coincap.io/v3/assets"
API_KEY = os.getenv("COINCAPSECRET", "")


def make_session() -> requests.Session:
    """
    Requests session with retries/backoff for transient network issues.
    """
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
    s.headers.update({"User-Agent": "OVUV/1.0 (https://ovuv.io)"})
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def fetch_coins(limit: int = 200, session: requests.Session | None = None) -> pd.DataFrame:
    """
    Fetch top N assets from CoinCap v3.
    Expects COINCAP_API_KEY to be set.
    """
    if not API_KEY:
        raise RuntimeError("COINCAP_API_KEY is not set")

    s = session or make_session()
    params = {
        "limit": limit,
        "apiKey": API_KEY,  # v3 expects an apiKey param
    }
    resp = s.get(COINCAP_URL, params=params, timeout=30)
    resp.raise_for_status()
    body = resp.json()
    data = body.get("data", [])
    if not data:
        raise RuntimeError(f"CoinCap returned no data: {body!r}")
    return pd.DataFrame(data)


def zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series([0] * len(series), index=series.index)
    return (series - series.mean()) / std


def compute_scores(limit: int = 200) -> dict:
    """
    Returns:
      - updated: timestamp (UTC)
      - currency: "USD"
      - top_undervalued: 25 rows
      - top_overvalued: 25 rows
    """
    session = make_session()
    df = fetch_coins(limit=limit, session=session)

    # v3 should still provide these fields (as strings)
    for col in ["rank", "priceUsd", "marketCapUsd", "volumeUsd24Hr", "changePercent24Hr"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["rank", "priceUsd", "marketCapUsd", "volumeUsd24Hr", "changePercent24Hr"])
    df = df[df["marketCapUsd"] > 0]

    df["current_price"] = df["priceUsd"]
    df["market_cap"] = df["marketCapUsd"]
    df["volume_24h"] = df["volumeUsd24Hr"]
    df["momentum_24h"] = df["changePercent24Hr"]

    df["value_ratio"] = df["volume_24h"] / df["market_cap"]

    df["value_component"] = zscore(df["value_ratio"])
    df["momentum_component"] = zscore(-df["momentum_24h"])

    df["undervalue_score"] = 0.6 * df["value_component"] + 0.4 * df["momentum_component"]

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
        print(f"ERROR while updating data: {e!r}")
        # Keep existing data.json if present, to avoid breaking the site
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
