"""Validate elasticity estimators against the simulator's ground truth.

The synthetic dataset records the elasticity used to generate each row
(`elasticity_used`) and the cross-price elasticity, so we can measure how well
each estimator recovers the truth instead of only reporting R-squared.

Compared, per product:
  * pooled log-log OLS      ln Q ~ ln P + ln P_comp + promo
  * fixed-effects log-log   ... + store FE + week FE + year FE

Outputs: outputs/elasticity_recovery.csv and docs/figures/elasticity_recovery.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

DATA = "data/cpg_weekly_store_sku.csv"
os.makedirs("outputs", exist_ok=True)
os.makedirs("docs/figures", exist_ok=True)

df = pd.read_csv(DATA).dropna(subset=["own_price", "competitor_price", "quantity_sold"])
df = df[(df["quantity_sold"] > 0) & (df["own_price"] > 0) & (df["competitor_price"] > 0)].copy()
df["lq"] = np.log(df["quantity_sold"])
df["lp"] = np.log(df["own_price"])
df["lc"] = np.log(df["competitor_price"])

rows = []
for product, g in df.groupby("product_id"):
    pooled = smf.ols("lq ~ lp + lc + promo_flag", g).fit()
    fe = smf.ols("lq ~ lp + lc + promo_flag + C(store_id) + C(week) + C(year)", g).fit(cov_type="cluster", cov_kwds={"groups": g["store_id"]})
    lo, hi = fe.conf_int().loc["lp"]
    rows.append(dict(product=product, true_own=g["elasticity_used"].mean(), pooled_own=pooled.params["lp"],
                     fe_own=fe.params["lp"], fe_own_lo=lo, fe_own_hi=hi,
                     true_cross=g["cross_price_elasticity"].mean(), fe_cross=fe.params["lc"]))

res = pd.DataFrame(rows).sort_values("true_own").reset_index(drop=True)
res.round(4).to_csv("outputs/elasticity_recovery.csv", index=False)
mae_pooled = (res["pooled_own"] - res["true_own"]).abs().mean()
mae_fe = (res["fe_own"] - res["true_own"]).abs().mean()
print(res.round(3).to_string(index=False))
print(f"\nOwn-price elasticity MAE  pooled OLS: {mae_pooled:.3f}   fixed effects: {mae_fe:.3f}")
print(f"Cross-price elasticity    true mean: {res['true_cross'].mean():.3f}   FE mean: {res['fe_cross'].mean():.3f}")

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(8.5, 4.2))
y = np.arange(len(res))
ax.hlines(y, res["true_own"], res["pooled_own"], color="#d1d5db", lw=2, zorder=1)
ax.scatter(res["true_own"], y, color="#111827", s=46, label="True (simulator)", zorder=3, marker="D")
ax.scatter(res["pooled_own"], y, color="#f59e0b", s=40, label=f"Pooled OLS (MAE {mae_pooled:.2f})", zorder=2)
ax.errorbar(res["fe_own"], y, xerr=[res["fe_own"] - res["fe_own_lo"], res["fe_own_hi"] - res["fe_own"]],
            fmt="o", color="#2563eb", ms=5, capsize=2, label=f"Store + week + year FE (MAE {mae_fe:.2f})", zorder=4)
ax.set_yticks(y); ax.set_yticklabels(res["product"]); ax.set_xlabel("Own-price elasticity")
ax.set_title("Recovering known elasticities: fixed effects remove the pooled-OLS bias", loc="left", fontweight="bold")
ax.legend(frameon=False, loc="lower right", fontsize=8); ax.grid(axis="x", alpha=0.25)
plt.tight_layout(); plt.savefig("docs/figures/elasticity_recovery.png", dpi=160, facecolor="white")
print("saved docs/figures/elasticity_recovery.png")
