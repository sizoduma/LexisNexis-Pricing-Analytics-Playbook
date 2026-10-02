from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_DATA_PATH = Path("data/sample_subscription_data.csv")
DEFAULT_OUTPUT_PATH = Path("outputs/pricing_recommendations.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Price optimization and recommendation engine for SaaS subscription products."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the subscription data file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to write pricing recommendations.",
    )
    return parser.parse_args()


def make_price_sensitivity_score(df: pd.DataFrame) -> pd.Series:
    """Build a simple proxy for price sensitivity using customer value indicators."""
    # Higher usage, higher renewal, and lower churn should allow room for pricing.
    score = (
        0.45 * df["usage_score"]
        + 0.35 * (df["renewal_probability"] * 100)
        + 0.15 * df["nps"]
        - 0.25 * (df["churn_risk"] * 100)
        - 0.10 * df["discount_pct"]
    )
    return score


def calculate_recommended_price(row: pd.Series) -> float:
    """Return recommended price based on a practical value-based adjustment."""
    # Base price change is driven by commercial desirability and risk.
    # Good customers with strong usage and renewal can justify a modest uplift.
    # Riskier or discount-heavy accounts should remain stable or receive a controlled discount.
    score = row["price_sensitivity_score"]

    if score >= 80:
        adjustment = 0.10
    elif score >= 70:
        adjustment = 0.06
    elif score >= 60:
        adjustment = 0.03
    elif score >= 50:
        adjustment = 0.00
    else:
        adjustment = -0.04

    # Adjust slightly for customer segment and region pricing differences.
    segment_adjustment = {
        "SMB": 0.01,
        "Corporate": 0.02,
        "Enterprise": 0.04,
    }.get(row["segment"], 0.0)

    region_adjustment = {
        "South Africa": 0.01,
        "Kenya": 0.00,
        "Nigeria": -0.01,
        "United Kingdom": 0.03,
        "Ghana": -0.02,
    }.get(row["region"], 0.0)

    final_adjustment = adjustment + segment_adjustment + region_adjustment
    recommended_price = row["monthly_price"] * (1 + final_adjustment)
    return round(float(recommended_price), 2)


def estimate_segment_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Compute segment-level recommended pricing strategy based on customer value signals."""
    summary = (
        df.groupby("segment", as_index=False)
        .agg(
            avg_usage=("usage_score", "mean"),
            avg_renewal=("renewal_probability", "mean"),
            avg_discount=("discount_pct", "mean"),
            avg_price=("monthly_price", "mean"),
            avg_churn=("churn_risk", "mean"),
            sample_size=("customer_id", "count"),
        )
        .sort_values("avg_usage", ascending=False)
        .copy()
    )

    summary["segment_price_strategy"] = np.where(
        summary["avg_renewal"] >= 0.80,
        "Premium with moderate uplift",
        np.where(
            summary["avg_usage"] >= 70,
            "Value-based pricing retention",
            "Price protection and offer optimization",
        ),
    )
    return summary


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.data)
    required_cols = {
        "customer_id",
        "region",
        "segment",
        "monthly_price",
        "usage_score",
        "renewal_probability",
        "discount_pct",
        "churn_risk",
        "nps",
    }
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["price_sensitivity_score"] = make_price_sensitivity_score(df)
    df["recommended_price"] = df.apply(calculate_recommended_price, axis=1)
    df["price_change_pct"] = ((df["recommended_price"] - df["monthly_price"]) / df["monthly_price"]) * 100
    df["pricing_action"] = np.where(
        df["price_change_pct"] > 5,
        "Increase price",
        np.where(df["price_change_pct"] < -3, "Reduce price / offer support", "Hold price"),
    )

    segment_summary = estimate_segment_baseline(df)
    output_dir = args.output.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    pricing_file = args.output
    pricing_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(pricing_file, index=False)

    summary_path = output_dir / "segment_pricing_summary.csv"
    segment_summary.to_csv(summary_path, index=False)

    print("Pricing optimization completed.")
    print(f"Customer level recommendations saved to: {pricing_file}")
    print(f"Segment summary saved to: {summary_path}")
    print("\nTop pricing opportunities by segment:")
    print(
        segment_summary[["segment", "avg_usage", "avg_renewal", "avg_discount", "segment_price_strategy"]]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
