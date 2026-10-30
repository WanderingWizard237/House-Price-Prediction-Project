"""
Generate a realistic synthetic Indian housing dataset.

Why synthetic? It keeps the project fully reproducible without downloading
anything. Prices follow realistic relationships (bigger area, better locality,
newer building => higher price) plus random noise, missing values, duplicate
rows and a few outliers, so the cleaning and modelling steps are meaningful.

To use a real dataset instead (e.g. from Kaggle), drop your CSV into data/ and
edit the CONFIG block at the top of src/train_model.py.

Usage:
    python src/generate_dataset.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_ROWS = 2000
OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "housing_data.csv"


def generate(n: int = N_ROWS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # --- Location -----------------------------------------------------------
    locality = rng.choice(["Prime", "Mid-range", "Suburban"], size=n, p=[0.2, 0.45, 0.35])
    dist_base = {"Prime": 4, "Mid-range": 10, "Suburban": 20}
    distance = np.array([max(0.5, rng.normal(dist_base[l], 2.5)) for l in locality])

    # --- Property features --------------------------------------------------
    bedrooms = rng.choice([1, 2, 3, 4, 5], size=n, p=[0.12, 0.35, 0.33, 0.15, 0.05])
    area = bedrooms * rng.normal(380, 60, n) + rng.normal(250, 80, n)
    area = np.clip(area, 350, 5000).round(0)
    bathrooms = np.clip(bedrooms + rng.choice([-1, 0, 0, 1], size=n), 1, 6)
    age = np.clip(rng.gamma(2.2, 4.5, n), 0, 40).round(0)
    total_floors = rng.integers(1, 21, size=n)
    floor = np.array([rng.integers(0, t + 1) for t in total_floors])
    parking = rng.choice([0, 1, 2], size=n, p=[0.2, 0.55, 0.25])
    furnishing = rng.choice(["Unfurnished", "Semi-furnished", "Furnished"], size=n, p=[0.35, 0.40, 0.25])
    lift = ((total_floors > 4) & (rng.random(n) < 0.9)).astype(int)
    security = rng.choice([0, 1], size=n, p=[0.35, 0.65])

    # --- Price (in Rs. lakhs) -----------------------------------------------
    loc_rate = {"Prime": 0.095, "Mid-range": 0.062, "Suburban": 0.042}  # lakhs per sq ft
    price = np.array([a * loc_rate[l] for a, l in zip(area, locality)])
    price += -0.9 * distance
    price += -0.55 * age
    price += 1.8 * bathrooms
    price += 3.5 * parking
    price += 4.0 * lift + 2.5 * security
    price += 0.25 * floor
    price += np.select([furnishing == "Furnished", furnishing == "Semi-furnished"], [8.0, 4.0], 0.0)
    price += rng.normal(0, 0.06 * np.abs(price) + 3, n)  # noise grows with price
    price = np.clip(price, 12, None).round(2)

    df = pd.DataFrame({
        "area_sqft": area.astype(int),
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "age_years": age.astype(int),
        "floor": floor,
        "total_floors": total_floors,
        "parking_spaces": parking,
        "furnishing": furnishing,
        "locality_type": locality,
        "distance_to_center_km": distance.round(1),
        "has_lift": lift,
        "has_security": security,
        "price_lakhs": price,
    })

    # --- Add realistic data problems ---------------------------------------
    for col, frac in [("age_years", 0.03), ("distance_to_center_km", 0.02), ("furnishing", 0.02)]:
        idx = rng.choice(n, int(frac * n), replace=False)
        df.loc[idx, col] = np.nan  # missing values
    out_idx = rng.choice(n, 12, replace=False)
    df.loc[out_idx, "price_lakhs"] = (df.loc[out_idx, "price_lakhs"] * rng.uniform(2.2, 3.0, 12)).round(2)  # outliers
    dup_idx = rng.choice(n, 10, replace=False)
    df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)  # duplicate rows
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    data = generate()
    OUT_PATH.parent.mkdir(exist_ok=True)
    data.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(data)} rows x {data.shape[1]} columns -> {OUT_PATH}")
