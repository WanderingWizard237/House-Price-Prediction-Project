"""
House Price Prediction using Multiple Linear Regression
=======================================================

End-to-end pipeline:
    1. Load data                    5. Train / test split
    2. Clean data                   6. Build + train the model
    3. Exploratory analysis         7. Evaluate (hold-out set + cross-validation)
    4. Feature engineering          8. Check regression assumptions + save outputs
       (area x locality interaction)

Usage:
    python src/train_model.py
"""
import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="deep")

# =============================================================================
# CONFIG  (edit this block to use your own dataset)
# =============================================================================
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "housing_data.csv"
FIG_DIR = ROOT / "outputs" / "figures"
MODEL_PATH = ROOT / "models" / "linear_regression_model.joblib"
METRICS_PATH = ROOT / "outputs" / "metrics.json"
COEF_PATH = ROOT / "outputs" / "coefficients.csv"

TARGET = "price_lakhs"
NUMERIC_FEATURES = [
    "area_sqft", "bedrooms", "bathrooms", "age_years", "floor", "total_floors",
    "parking_spaces", "distance_to_center_km", "has_lift", "has_security",
]
CATEGORICAL_FEATURES = ["furnishing", "locality_type"]
# Engineered features: price per sq ft differs by locality, so area's effect is
# allowed to change with locality (an interaction term).
ENGINEERED_FEATURES = ["area_x_prime", "area_x_midrange"]
TEST_SIZE = 0.20
RANDOM_STATE = 42
OUTLIER_IQR_MULTIPLIER = 3.0  # rows with price beyond Q3 + 3*IQR are removed
# =============================================================================


def savefig(name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=130, bbox_inches="tight")
    plt.close()


# ----------------------------------------------------------------------------
# 1-2. Load and clean
# ----------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {df.shape[0]} rows x {df.shape[1]} columns from {DATA_PATH.name}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)
    df = df.drop_duplicates().copy()
    print(f"  Removed {n0 - len(df)} duplicate rows")

    n1 = len(df)
    df = df.dropna(subset=[TARGET])
    print(f"  Removed {n1 - len(df)} rows with missing target")

    # Outlier removal on the target using the IQR rule
    q1, q3 = df[TARGET].quantile([0.25, 0.75])
    upper = q3 + OUTLIER_IQR_MULTIPLIER * (q3 - q1)
    n2 = len(df)
    df = df[df[TARGET] <= upper]
    print(f"  Removed {n2 - len(df)} price outliers (> {upper:.1f} lakhs)")

    print(f"  Missing feature values (imputed later in the pipeline):\n"
          f"{df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].isna().sum()[lambda s: s > 0].to_string()}")
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------------
# 3. Exploratory plots
# ----------------------------------------------------------------------------
def explore(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(df[TARGET], kde=True, ax=ax[0])
    ax[0].set(title="Distribution of house prices", xlabel="Price (lakhs)")
    sns.scatterplot(data=df, x="area_sqft", y=TARGET, hue="locality_type", alpha=0.6, ax=ax[1])
    ax[1].set(title="Area vs price by locality", xlabel="Area (sq ft)", ylabel="Price (lakhs)")
    savefig("01_price_distribution_and_area.png")

    plt.figure(figsize=(9, 7))
    sns.heatmap(df[NUMERIC_FEATURES + [TARGET]].corr(), annot=True, fmt=".2f",
                cmap="coolwarm", center=0, square=True, cbar_kws={"shrink": 0.8})
    plt.title("Correlation matrix")
    savefig("02_correlation_heatmap.png")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    sns.boxplot(data=df, x="locality_type", y=TARGET, order=["Suburban", "Mid-range", "Prime"], ax=ax[0])
    ax[0].set(title="Price by locality", xlabel="", ylabel="Price (lakhs)")
    sns.boxplot(data=df, x="furnishing", y=TARGET,
                order=["Unfurnished", "Semi-furnished", "Furnished"], ax=ax[1])
    ax[1].set(title="Price by furnishing", xlabel="", ylabel="Price (lakhs)")
    savefig("03_price_by_category.png")


# ----------------------------------------------------------------------------
# 4. Feature engineering
# ----------------------------------------------------------------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add area x locality interaction terms (Suburban is the reference group)."""
    df = df.copy()
    df["area_x_prime"] = df["area_sqft"] * (df["locality_type"] == "Prime")
    df["area_x_midrange"] = df["area_sqft"] * (df["locality_type"] == "Mid-range")
    return df


# ----------------------------------------------------------------------------
# 6. Model
# ----------------------------------------------------------------------------
def build_pipeline(numeric_features=None) -> Pipeline:
    """Preprocessing + LinearRegression in one object, so no data leakage."""
    numeric_features = numeric_features or NUMERIC_FEATURES + ENGINEERED_FEATURES
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
    ])
    pre = ColumnTransformer([
        ("num", numeric, numeric_features),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])
    return Pipeline([("preprocess", pre), ("model", LinearRegression())])


def regression_metrics(y_true, y_pred, n_features: int) -> dict:
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    return {
        "R2": round(r2, 4),
        "Adjusted_R2": round(1 - (1 - r2) * (n - 1) / (n - n_features - 1), 4),
        "MAE": round(mean_absolute_error(y_true, y_pred), 3),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
        "MAPE_percent": round(float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100), 2),
    }


# ----------------------------------------------------------------------------
# 7-8. Evaluation plots and assumption checks
# ----------------------------------------------------------------------------
def evaluation_plots(y_test, y_pred, residuals) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].scatter(y_test, y_pred, alpha=0.5, edgecolor="none")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax[0].plot(lims, lims, "r--", label="Perfect prediction")
    ax[0].set(title="Actual vs predicted price", xlabel="Actual (lakhs)", ylabel="Predicted (lakhs)")
    ax[0].legend()
    ax[1].scatter(y_pred, residuals, alpha=0.5, edgecolor="none")
    ax[1].axhline(0, color="r", linestyle="--")
    ax[1].set(title="Residuals vs predicted", xlabel="Predicted (lakhs)", ylabel="Residual (lakhs)")
    savefig("04_actual_vs_predicted_and_residuals.png")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(residuals, kde=True, ax=ax[0])
    ax[0].set(title="Residual distribution", xlabel="Residual (lakhs)")
    stats.probplot(residuals, dist="norm", plot=ax[1])
    ax[1].set_title("Q-Q plot of residuals")
    savefig("05_residual_normality.png")


def coefficient_table(pipe: Pipeline) -> pd.DataFrame:
    names = pipe.named_steps["preprocess"].get_feature_names_out()
    names = [n.split("__", 1)[1] for n in names]
    coefs = pipe.named_steps["model"].coef_
    table = (pd.DataFrame({"feature": names, "standardized_coefficient": coefs})
             .assign(abs_effect=lambda d: d.standardized_coefficient.abs())
             .sort_values("abs_effect", ascending=False).drop(columns="abs_effect")
             .reset_index(drop=True))
    plt.figure(figsize=(8, 6))
    colors = ["#2a9d8f" if c > 0 else "#e76f51" for c in table.standardized_coefficient[::-1]]
    plt.barh(table.feature[::-1], table.standardized_coefficient[::-1], color=colors)
    plt.axvline(0, color="black", lw=0.8)
    plt.title("Feature influence on price (standardized coefficients)")
    plt.xlabel("Change in price (lakhs) per 1 std-dev increase in feature")
    savefig("06_feature_coefficients.png")
    return table


def assumption_checks(pipe: Pipeline, X_train, y_train) -> dict:
    """OLS diagnostics on the training data: significance, multicollinearity, heteroscedasticity."""
    X_proc = pipe.named_steps["preprocess"].transform(X_train)
    names = [n.split("__", 1)[1] for n in pipe.named_steps["preprocess"].get_feature_names_out()]
    X_df = pd.DataFrame(X_proc, columns=names)
    ols = sm.OLS(y_train.values, sm.add_constant(X_df)).fit()

    vif = pd.DataFrame({
        "feature": names,
        "VIF": [variance_inflation_factor(X_df.values, i) for i in range(X_df.shape[1])],
    }).sort_values("VIF", ascending=False)

    from statsmodels.stats.diagnostic import het_breuschpagan
    bp_stat, bp_p, _, _ = het_breuschpagan(ols.resid, sm.add_constant(X_df))
    shapiro_p = stats.shapiro(ols.resid.sample(min(len(ols.resid), 5000), random_state=0) if hasattr(ols.resid, "sample")
                              else ols.resid[:5000])[1]

    summary_path = ROOT / "outputs" / "ols_summary.txt"
    summary_path.write_text(str(ols.summary()))
    vif.to_csv(ROOT / "outputs" / "vif.csv", index=False)

    return {
        "ols_r2_train": round(ols.rsquared, 4),
        "f_test_p_value": float(ols.f_pvalue),
        "durbin_watson": round(float(sm.stats.durbin_watson(ols.resid)), 3),
        "breusch_pagan_p_value": round(float(bp_p), 4),
        "shapiro_wilk_p_value": round(float(shapiro_p), 4),
        "max_vif": round(float(vif.VIF.max()), 2),
        "non_significant_features": [n for n, p in ols.pvalues.items() if p > 0.05 and n != "const"],
    }


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    print("\n[1/6] Loading data")
    df = load_data()

    print("\n[2/6] Cleaning data")
    df = clean_data(df)

    print("\n[3/6] Exploratory analysis (saving plots)")
    explore(df)

    print("\n[4/6] Feature engineering + train/test split")
    df = add_features(df)
    base_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    all_cols = NUMERIC_FEATURES + ENGINEERED_FEATURES + CATEGORICAL_FEATURES
    X, y = df[all_cols], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"  Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    print("\n[5/6] Training Linear Regression (baseline vs with interaction terms)")
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    comparison = {}
    for label, feats in [("baseline_no_interaction", NUMERIC_FEATURES),
                         ("final_with_interaction", NUMERIC_FEATURES + ENGINEERED_FEATURES)]:
        m = build_pipeline(feats).fit(X_train[feats + CATEGORICAL_FEATURES], y_train)
        pred = m.predict(X_test[feats + CATEGORICAL_FEATURES])
        cv_r2_ = cross_val_score(build_pipeline(feats), X[feats + CATEGORICAL_FEATURES], y, cv=cv, scoring="r2")
        comparison[label] = {
            "test_R2": round(r2_score(y_test, pred), 4),
            "test_RMSE": round(float(np.sqrt(mean_squared_error(y_test, pred))), 3),
            "cv_R2_mean": round(cv_r2_.mean(), 4),
        }
        print(f"  {label:26s} test R2={comparison[label]['test_R2']}  "
              f"RMSE={comparison[label]['test_RMSE']}  CV R2={comparison[label]['cv_R2_mean']}")

    pipe = build_pipeline().fit(X_train, y_train)  # final model
    n_features = pipe.named_steps["preprocess"].transform(X_train[:1]).shape[1]

    print("\n[6/6] Evaluating final model")
    pred_train, pred_test = pipe.predict(X_train), pipe.predict(X_test)
    train_m = regression_metrics(y_train, pred_train, n_features)
    test_m = regression_metrics(y_test, pred_test, n_features)

    cv_r2 = cross_val_score(build_pipeline(), X, y, cv=cv, scoring="r2")
    cv_rmse = -cross_val_score(build_pipeline(), X, y, cv=cv, scoring="neg_root_mean_squared_error")

    # Naive baseline: always predict the training mean
    baseline_rmse = float(np.sqrt(mean_squared_error(y_test, np.full(len(y_test), y_train.mean()))))

    evaluation_plots(y_test, pred_test, y_test - pred_test)
    coef = coefficient_table(pipe)
    coef.to_csv(COEF_PATH, index=False)
    diag = assumption_checks(pipe, X_train, y_train)

    results = {
        "train_metrics": train_m,
        "test_metrics": test_m,
        "cross_validation_5fold": {
            "R2_mean": round(cv_r2.mean(), 4), "R2_std": round(cv_r2.std(), 4),
            "RMSE_mean": round(cv_rmse.mean(), 3), "RMSE_std": round(cv_rmse.std(), 3),
        },
        "baseline_mean_predictor_RMSE": round(baseline_rmse, 3),
        "model_comparison": comparison,
        "assumption_diagnostics": diag,
        "dataset": {"rows_after_cleaning": len(df), "n_train": len(X_train), "n_test": len(X_test)},
    }
    METRICS_PATH.write_text(json.dumps(results, indent=2))
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Train  : {train_m}")
    print(f"Test   : {test_m}")
    print(f"5-fold CV R2   : {cv_r2.mean():.4f} (+/- {cv_r2.std():.4f})")
    print(f"5-fold CV RMSE : {cv_rmse.mean():.3f} (+/- {cv_rmse.std():.3f})")
    print(f"Baseline (predict mean) RMSE: {baseline_rmse:.3f}")
    print("\nTop 5 price drivers (standardized coefficients):")
    print(coef.head(5).to_string(index=False))
    print("\nAssumption diagnostics:")
    for k, v in diag.items():
        print(f"  {k}: {v}")
    print(f"\nModel saved   -> {MODEL_PATH.relative_to(ROOT)}")
    print(f"Metrics saved -> {METRICS_PATH.relative_to(ROOT)}")
    print(f"Figures saved -> {FIG_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
