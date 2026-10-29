import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

base_dir = Path(r"C:\Users\satis\OneDrive\Desktop\UEL\7010\Dissertation")
predictions_file = base_dir / "outputs" / "predictions.csv"
figure_dir = base_dir / "outputs" / "figures"

df = pd.read_csv(predictions_file)

df["date"] = pd.to_datetime(df["date"])

daily = (
    df.groupby("date", as_index=False)[
        [
            "actual",
            "predicted_RandomForest",
            "predicted_XGBoost",
            "predicted_LightGBM",
        ]
    ]
    .sum()
    .sort_values("date")
)

print("\nDaily aggregated forecast data:")
print(daily.head(10).to_string(index=False))
print("\nNumber of daily observations:", len(daily))
print(
    "Date range:",
    daily["date"].min().date(),
    "to",
    daily["date"].max().date()
)

models = {
    "RandomForest": "predicted_RandomForest",
    "XGBoost": "predicted_XGBoost",
    "LightGBM": "predicted_LightGBM",
}

for model_name, pred_col in models.items():
    plt.figure(figsize=(12, 5))
    plt.plot(daily["date"], daily["actual"], label="Actual")
    plt.plot(daily["date"], daily[pred_col], label="Predicted")

    plt.title(f"{model_name}: Daily Actual and Predicted Demand")
    plt.xlabel("Date")
    plt.ylabel("Total demand")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    output_file = figure_dir / f"corrected_daily_actual_predicted_{model_name}.png"
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"Saved: {output_file}")

print("\nFinished creating corrected daily forecast figures.")