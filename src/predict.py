"""
Predict the price of a house using the trained model.

Usage:
    python src/predict.py                 # runs the built-in example
    python src/predict.py --area 1500 --bedrooms 3 --locality Prime
"""
import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_model import MODEL_PATH, add_features  # noqa: E402


def predict_price(house: dict) -> float:
    """Return the predicted price (in Rs. lakhs) for one house described by a dict."""
    model = joblib.load(MODEL_PATH)
    return float(model.predict(add_features(pd.DataFrame([house])))[0])


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict house price (Rs. lakhs)")
    ap.add_argument("--area", type=int, default=1200, help="Area in sq ft")
    ap.add_argument("--bedrooms", type=int, default=3)
    ap.add_argument("--bathrooms", type=int, default=2)
    ap.add_argument("--age", type=int, default=5, help="Age of building in years")
    ap.add_argument("--floor", type=int, default=3)
    ap.add_argument("--total-floors", type=int, default=10)
    ap.add_argument("--parking", type=int, default=1)
    ap.add_argument("--distance", type=float, default=8.0, help="Distance to city centre in km")
    ap.add_argument("--lift", type=int, default=1, choices=[0, 1])
    ap.add_argument("--security", type=int, default=1, choices=[0, 1])
    ap.add_argument("--furnishing", default="Semi-furnished",
                    choices=["Unfurnished", "Semi-furnished", "Furnished"])
    ap.add_argument("--locality", default="Mid-range", choices=["Prime", "Mid-range", "Suburban"])
    a = ap.parse_args()

    house = {
        "area_sqft": a.area, "bedrooms": a.bedrooms, "bathrooms": a.bathrooms,
        "age_years": a.age, "floor": a.floor, "total_floors": a.total_floors,
        "parking_spaces": a.parking, "distance_to_center_km": a.distance,
        "has_lift": a.lift, "has_security": a.security,
        "furnishing": a.furnishing, "locality_type": a.locality,
    }
    price = predict_price(house)
    print("House details:", house)
    print(f"\nPredicted price: Rs. {price:,.2f} lakhs  (~ Rs. {price / 100:,.2f} crore)")


if __name__ == "__main__":
    main()
