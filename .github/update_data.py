import requests
import pandas as pd
import json
from datetime import datetime
from pathlib import Path

COINCAP_URL = "https://api.coincap.io/v2/assets"


def fetch_coins(limit: int = 200) -> pd.DataFrame:
    """
    Fetch top N assets from CoinCap.
    Docs: https://docs.coincap.io/
    """
    params = {"limit": limit}
    resp = requests.get(COINCAP_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()["data"]
    return pd.DataFrame(data)


def zscore(series: pd.Series) -> pd.Series:
    # Handle edge case: constant series (std=0)
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
    df = fetch_coins(limit=limit)

    # Convert numeric columns
    for col in ["rank", "priceUsd", "marketCapUsd", "volumeUsd24Hr", "changePercent24Hr"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["rank", "priceUsd", "marketCapUsd", "volumeUsd24Hr", "changePercent24Hr"])

    # Core metrics
    df["current_price"] = df["priceUsd"]
    df["market_cap"] = df["marketCapUsd"]
    df["volume_24h"] = df["volumeUsd24Hr"]
    df["momentum_24h"] = df["changePercent24Hr"]

    # Avoid division by zero
    df = df[df["market_cap"] > 0]

    # Value ratio: volume relative to market cap
    df["value_ratio"] = df["volume_24h"] / df["market_cap"]

    # Z-score components
    df["value_component"] = zscore(df["value_ratio"])
    # Lower or negative momentum_24h => potentially more undervalued
    df["momentum_component"] = zscore(-df["momentum_24h"])

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

    result = {
        "updated": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "currency": "USD",
        "top_undervalued": top_undervalued,
        "top_overvalued": top_overvalued,
    }
    return result


def main():
    data = compute_scores(limit=200)

    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)

    out_path = docs_dir / "data.json"
    out_path.write_text(json.dumps(data, indent=2))
    print(
        f"Updated {len(data['top_undervalued'])} undervalued "
        f"and {len(data['top_overvalued'])} overvalued coins at {data['updated']}"
    )


if __name__ == "__main__":
    main()
