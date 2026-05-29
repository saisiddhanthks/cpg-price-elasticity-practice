import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

try:
    from pygam import LinearGAM, s, f
    GAM_AVAILABLE = True
except:
    GAM_AVAILABLE = False


DATA_PATH = "huge_cpg_price_elasticity_dataset.csv"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


df = pd.read_csv(DATA_PATH)

print("\nDATA LOADED SUCCESSFULLY")
print(df.shape)
print(df.head())
print(df.info())


print("\nMISSING VALUES")
print(df.isnull().sum())


numeric_cols = ["own_price", "promo_price", "competitor_price", "quantity_sold"]

median_imputer = SimpleImputer(strategy="median")
df[numeric_cols] = median_imputer.fit_transform(df[numeric_cols])

print("\nMISSING VALUES AFTER IMPUTATION")
print(df.isnull().sum())


df["week_sin"] = np.sin(2 * np.pi * df["week"] / 52)
df["week_cos"] = np.cos(2 * np.pi * df["week"] / 52)
df["year_encoded"] = df["year"] - df["year"].min()
df["log_volume"] = np.log(df["quantity_sold"])
df["log_own_price"] = np.log(df["promo_price"])
df["log_comp_price"] = np.log(df["competitor_price"])
df["discount_pct"] = (df["edv_price"] - df["promo_price"]) / df["edv_price"]
df["discount_pct"] = df["discount_pct"].clip(lower=0)


print("\nFEATURE ENGINEERING COMPLETED")
print(df.head())


print("\nBASIC EDA")
print(df.describe())

product_sales = df.groupby("product_id")["quantity_sold"].sum().sort_values(ascending=False)
print("\nPRODUCT-WISE SALES")
print(product_sales)

banner_sales = df.groupby("banner")["quantity_sold"].sum().sort_values(ascending=False)
print("\nBANNER-WISE SALES")
print(banner_sales)


plt.figure(figsize=(10, 5))
product_sales.plot(kind="bar")
plt.title("Total Quantity Sold by Product")
plt.xlabel("Product")
plt.ylabel("Quantity Sold")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/product_sales.png")
plt.show()


plt.figure(figsize=(10, 5))
df.groupby("week")["quantity_sold"].mean().plot()
plt.title("Average Weekly Quantity Sold")
plt.xlabel("Week")
plt.ylabel("Average Quantity")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/weekly_quantity.png")
plt.show()


corr_cols = [
    "edv_price", "own_price", "promo_price", "competitor_price",
    "quantity_sold", "revenue", "discount_pct", "week_sin", "week_cos"
]

corr_matrix = df[corr_cols].corr()

print("\nCORRELATION MATRIX")
print(corr_matrix)

plt.figure(figsize=(10, 7))
plt.imshow(corr_matrix, aspect="auto")
plt.xticks(range(len(corr_cols)), corr_cols, rotation=90)
plt.yticks(range(len(corr_cols)), corr_cols)
plt.colorbar()
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/correlation_heatmap.png")
plt.show()


vif_features = [
    "log_own_price", "log_comp_price", "promo_flag",
    "discount_pct", "week_sin", "week_cos", "year_encoded"
]

X_vif = df[vif_features].copy()
X_vif = sm.add_constant(X_vif)

vif_data = pd.DataFrame()
vif_data["feature"] = X_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]

print("\nVIF VALUES")
print(vif_data)


ols_static_features = ["log_own_price", "log_comp_price", "promo_flag", "year_encoded"]
X_static = df[ols_static_features]
y = df["log_volume"]

X_static = sm.add_constant(X_static)
ols_static_model = sm.OLS(y, X_static).fit()

print("\nOLS STATIC MODEL SUMMARY")
print(ols_static_model.summary())

own_price_elasticity = ols_static_model.params["log_own_price"]
cross_price_elasticity = ols_static_model.params["log_comp_price"]

print("\nOLS STATIC ELASTICITY RESULTS")
print("Own Price Elasticity:", own_price_elasticity)
print("Cross Price Elasticity:", cross_price_elasticity)

print("\nBUSINESS INTERPRETATION")
print(f"1% increase in own price changes volume by {own_price_elasticity:.2f}%")
print(f"1% increase in competitor price changes your volume by {cross_price_elasticity:.2f}%")


ols_dynamic_features = [
    "log_own_price", "log_comp_price", "promo_flag",
    "week_sin", "week_cos", "year_encoded"
]

df["own_price_week_sin"] = df["log_own_price"] * df["week_sin"]
df["own_price_week_cos"] = df["log_own_price"] * df["week_cos"]

dynamic_features = [
    "log_own_price", "log_comp_price", "promo_flag",
    "week_sin", "week_cos",
    "own_price_week_sin", "own_price_week_cos",
    "year_encoded"
]

X_dynamic = df[dynamic_features]
X_dynamic = sm.add_constant(X_dynamic)

ols_dynamic_model = sm.OLS(y, X_dynamic).fit()

print("\nOLS DYNAMIC MODEL SUMMARY")
print(ols_dynamic_model.summary())


beta_base = ols_dynamic_model.params["log_own_price"]
beta_sin = ols_dynamic_model.params["own_price_week_sin"]
beta_cos = ols_dynamic_model.params["own_price_week_cos"]

weeks = np.arange(1, 53)
weekly_elasticity = (
    beta_base
    + beta_sin * np.sin(2 * np.pi * weeks / 52)
    + beta_cos * np.cos(2 * np.pi * weeks / 52)
)

weekly_elasticity_df = pd.DataFrame({
    "week": weeks,
    "elasticity": weekly_elasticity
})

print("\nWEEKLY DYNAMIC ELASTICITY")
print(weekly_elasticity_df)

plt.figure(figsize=(10, 5))
plt.plot(weekly_elasticity_df["week"], weekly_elasticity_df["elasticity"])
plt.title("Dynamic Weekly Price Elasticity")
plt.xlabel("Week")
plt.ylabel("Elasticity")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/dynamic_weekly_elasticity.png")
plt.show()


banner_elasticities = []

for banner in df["banner"].unique():
    temp = df[df["banner"] == banner].copy()
    X_banner = temp[["log_own_price", "log_comp_price", "promo_flag", "week_sin", "week_cos"]]
    y_banner = temp["log_volume"]
    X_banner = sm.add_constant(X_banner)
    model = sm.OLS(y_banner, X_banner).fit()
    banner_elasticities.append({
        "banner": banner,
        "own_price_elasticity": model.params["log_own_price"],
        "cross_price_elasticity": model.params["log_comp_price"],
        "r2": model.rsquared
    })

banner_elasticity_df = pd.DataFrame(banner_elasticities)

print("\nBANNER-WISE ELASTICITY")
print(banner_elasticity_df)

plt.figure(figsize=(10, 5))
plt.bar(banner_elasticity_df["banner"], banner_elasticity_df["own_price_elasticity"])
plt.title("Banner-wise Own Price Elasticity")
plt.xlabel("Banner")
plt.ylabel("Elasticity")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/banner_elasticity.png")
plt.show()


product_elasticities = []

for product in df["product_id"].unique():
    temp = df[df["product_id"] == product].copy()
    X_product = temp[["log_own_price", "log_comp_price", "promo_flag", "week_sin", "week_cos"]]
    y_product = temp["log_volume"]
    X_product = sm.add_constant(X_product)
    model = sm.OLS(y_product, X_product).fit()
    product_elasticities.append({
        "product_id": product,
        "own_price_elasticity": model.params["log_own_price"],
        "cross_price_elasticity": model.params["log_comp_price"],
        "r2": model.rsquared
    })

product_elasticity_df = pd.DataFrame(product_elasticities)

print("\nPRODUCT-WISE ELASTICITY")
print(product_elasticity_df)

plt.figure(figsize=(10, 5))
plt.bar(product_elasticity_df["product_id"], product_elasticity_df["own_price_elasticity"])
plt.title("Product-wise Own Price Elasticity")
plt.xlabel("Product")
plt.ylabel("Elasticity")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/product_elasticity.png")
plt.show()


ml_features = [
    "promo_price", "competitor_price", "promo_flag", "week",
    "week_sin", "week_cos", "year_encoded", "discount_pct"
]

df_ml = pd.get_dummies(
    df[ml_features + ["product_id", "banner", "location_type", "quantity_sold"]],
    columns=["product_id", "banner", "location_type"],
    drop_first=True
)

X_ml = df_ml.drop(columns=["quantity_sold"])
y_ml = df_ml["quantity_sold"]

X_train, X_test, y_train, y_test = train_test_split(
    X_ml, y_ml, test_size=0.2, random_state=42
)

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print("\nRANDOM FOREST MODEL")
print("RMSE:", np.sqrt(mean_squared_error(y_test, rf_pred)))
print("R2:", r2_score(y_test, rf_pred))


if GAM_AVAILABLE:
    gam_df = df.sample(30000, random_state=42).copy()

    gam_X = gam_df[["promo_price", "competitor_price", "promo_flag", "week"]].values
    gam_y = gam_df["quantity_sold"].values

    gam_model = LinearGAM(
        s(0) + s(1) + f(2) + s(3)
    ).fit(gam_X, gam_y)

    print("\nGAM MODEL TRAINED SUCCESSFULLY")

    price_grid = np.linspace(
        gam_df["promo_price"].quantile(0.05),
        gam_df["promo_price"].quantile(0.95),
        100
    )

    avg_comp_price = gam_df["competitor_price"].mean()
    avg_promo_flag = 0
    avg_week = 26

    gam_pred_data = np.column_stack([
        price_grid,
        np.repeat(avg_comp_price, 100),
        np.repeat(avg_promo_flag, 100),
        np.repeat(avg_week, 100)
    ])

    gam_predictions = gam_model.predict(gam_pred_data)

    gam_result = pd.DataFrame({
        "price": price_grid,
        "predicted_quantity": gam_predictions,
        "predicted_revenue": price_grid * gam_predictions
    })

    print("\nGAM PRICE OPTIMIZATION RESULT")
    print(gam_result.sort_values("predicted_revenue", ascending=False).head())

    plt.figure(figsize=(10, 5))
    plt.plot(gam_result["price"], gam_result["predicted_quantity"])
    plt.title("GAM: Price vs Predicted Quantity")
    plt.xlabel("Price")
    plt.ylabel("Predicted Quantity")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/gam_price_quantity_curve.png")
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(gam_result["price"], gam_result["predicted_revenue"])
    plt.title("GAM: Price vs Predicted Revenue")
    plt.xlabel("Price")
    plt.ylabel("Predicted Revenue")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/gam_price_revenue_curve.png")
    plt.show()

else:
    print("\nGAM SKIPPED")
    print("Install pygam using: pip install pygam")


scenario_df = df[
    (df["product_id"] == "Cola_500ml") &
    (df["banner"] == "UrbanMart")
].copy()

avg_row = scenario_df.mean(numeric_only=True)

base_price = avg_row["promo_price"]
base_quantity = avg_row["quantity_sold"]

price_changes = np.arange(-0.20, 0.21, 0.05)

simulation_results = []

for change in price_changes:
    new_price = base_price * (1 + change)
    expected_volume_change = own_price_elasticity * change
    new_quantity = base_quantity * (1 + expected_volume_change)
    new_revenue = new_price * new_quantity

    simulation_results.append({
        "price_change_pct": round(change * 100, 1),
        "new_price": round(new_price, 2),
        "expected_quantity": round(new_quantity, 2),
        "expected_revenue": round(new_revenue, 2)
    })

simulation_df = pd.DataFrame(simulation_results)

print("\nPRICE SCENARIO SIMULATION")
print(simulation_df)

plt.figure(figsize=(10, 5))
plt.plot(simulation_df["price_change_pct"], simulation_df["expected_revenue"], marker="o")
plt.title("Price Change vs Expected Revenue")
plt.xlabel("Price Change %")
plt.ylabel("Expected Revenue")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/price_scenario_revenue.png")
plt.show()


df.to_csv(f"{OUTPUT_DIR}/processed_cpg_dataset.csv", index=False)
weekly_elasticity_df.to_csv(f"{OUTPUT_DIR}/weekly_dynamic_elasticity.csv", index=False)
banner_elasticity_df.to_csv(f"{OUTPUT_DIR}/banner_elasticity.csv", index=False)
product_elasticity_df.to_csv(f"{OUTPUT_DIR}/product_elasticity.csv", index=False)
simulation_df.to_csv(f"{OUTPUT_DIR}/price_simulation.csv", index=False)

print("\nPIPELINE COMPLETED SUCCESSFULLY")
print("All outputs saved inside outputs folder")
