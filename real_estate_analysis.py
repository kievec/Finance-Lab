from pathlib import Path
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# -------------------------------------------------
# Paths
# -------------------------------------------------
BASE_DIR = Path(r"C:\Users\kieve\Downloads")

REAL_ESTATE_PATH = BASE_DIR / "real_estate_eur.csv"
INFLATION_PATH = BASE_DIR / "eu_inflation.csv"

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def load_real_estate_asset(path, asset_name="Real Estate (EUR)"):
    df = pd.read_csv(path)

    # Real estate file has metadata rows in Price, then monthly dates
    df["Date"] = pd.to_datetime(df["Price"], errors="coerce")
    df["Date"] = df["Date"].dt.to_period("M").dt.to_timestamp("M")  # month-end alignment
    df["asset_return"] = pd.to_numeric(df["monthly_return"], errors="coerce")
    df["asset_name"] = asset_name

    df = df.dropna(subset=["Date", "asset_return"]).copy()
    df = df.sort_values("Date").reset_index(drop=True)

    return df[["Date", "asset_return", "asset_name", "Price", "price_eur", "monthly_return"]]


def load_inflation(path):
    inf = pd.read_csv(path)
    inf["DATE"] = pd.to_datetime(inf["DATE"], errors="coerce")
    inf["Date"] = inf["DATE"].dt.to_period("M").dt.to_timestamp("M")
    inf = inf.rename(columns={"DATE": "OriginalDate"})
    return inf[["Date", "HICP", "HICP_YoY", "HICP_MoM", "HICP_12m_avg", "high_inflation"]]


def merge_asset_inflation(asset_df, inflation_df):
    merged = asset_df.merge(inflation_df, on="Date", how="inner")
    merged = merged.dropna(subset=["asset_return", "HICP_MoM", "HICP_YoY", "high_inflation"]).copy()
    merged = merged.sort_values("Date").reset_index(drop=True)
    return merged


def ols_beta(y, x):
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    return model


def bootstrap_beta(df, y_col="asset_return", x_col="HICP_MoM", n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    betas = []
    n = len(df)

    for _ in range(n_boot):
        sample_idx = rng.integers(0, n, size=n)
        sample = df.iloc[sample_idx]
        try:
            model = ols_beta(sample[y_col], sample[x_col])
            betas.append(model.params[x_col])
        except Exception:
            continue

    betas = np.array(betas)
    ci90 = np.percentile(betas, [5, 95])
    return betas, ci90


def rolling_corr(df, window=12, y_col="asset_return", x_col="HICP_YoY"):
    return df[y_col].rolling(window).corr(df[x_col])


def high_low_inflation_summary(df):
    grp = df.groupby("high_inflation")["asset_return"].agg(["count", "mean", "std"])
    grp = grp.rename(index={0: "Low inflation", 1: "High inflation"})
    return grp


def make_rolling_corr_plot(df, corr_series, outpath, title):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(df["Date"], corr_series, linewidth=1.5)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("12m rolling correlation")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def analyze_asset(asset_path, inflation_path, asset_name, out_prefix):
    asset = load_real_estate_asset(asset_path, asset_name=asset_name)
    inflation = load_inflation(inflation_path)
    df = merge_asset_inflation(asset, inflation)

    print("\nMerged rows:", len(df))

    if len(df) > 0:
        print(df[["Date", "asset_return", "HICP_MoM"]].head())

    if len(df) == 0:
        raise ValueError(
            "Merge produced zero rows. Check date alignment between "
            "real_estate_eur.csv and eu_inflation.csv."
        )

    hl_table = high_low_inflation_summary(df)

    model = ols_beta(df["asset_return"], df["HICP_MoM"])

    betas, ci90 = bootstrap_beta(df, n_boot=1000)

    df["roll_corr_12m"] = rolling_corr(df, window=12)
    plot_path = BASE_DIR / f"{out_prefix}_rolling_corr.png"
    make_rolling_corr_plot(
        df,
        df["roll_corr_12m"],
        plot_path,
        title=f"{asset_name}: 12M rolling correlation with HICP YoY"
    )

    merged_path = BASE_DIR / f"{out_prefix}_merged_analysis.csv"
    df.to_csv(merged_path, index=False)

    results = {
        "asset_name": asset_name,
        "n_obs": len(df),
        "high_low_table": hl_table,
        "ols_model": model,
        "beta": model.params["HICP_MoM"],
        "beta_pvalue": model.pvalues["HICP_MoM"],
        "beta_ci90": ci90,
        "bootstrap_betas": betas,
        "plot_path": str(plot_path),
        "merged_path": str(merged_path),
        "analysis_df": df
    }
    return results


def format_inflation_result(results):
    beta = results["beta"]
    pval = results["beta_pvalue"]
    ci_low, ci_high = results["beta_ci90"]

    print(f"\n=== {results['asset_name']} ===")
    print(f"Observations used: {results['n_obs']}")
    print("\nHigh vs low inflation months:")
    print(results["high_low_table"].to_string())
    print("\nOLS regression: asset_return = a + b * HICP_MoM")
    print(f"b = {beta:.6f}")
    print(f"p-value = {pval:.4f}")
    print(f"Bootstrap 90% CI for b = [{ci_low:.6f}, {ci_high:.6f}]")
    print(f"Zero inside CI? {'YES' if (ci_low <= 0 <= ci_high) else 'NO'}")
    print(f"Rolling-corr plot saved to: {results['plot_path']}")


# -------------------------------------------------
# Main
# -------------------------------------------------
if __name__ == "__main__":
    result = analyze_asset(
        asset_path=REAL_ESTATE_PATH,
        inflation_path=INFLATION_PATH,
        asset_name="Real Estate (EUR)",
        out_prefix="real_estate_eur"
    )

    format_inflation_result(result)

    summary_rows = []
    ci_low, ci_high = result["beta_ci90"]
    summary_rows.append({
        "asset": result["asset_name"],
        "n_obs": result["n_obs"],
        "beta_on_HICP_MoM": result["beta"],
        "p_value": result["beta_pvalue"],
        "ci90_low": ci_low,
        "ci90_high": ci_high,
        "zero_in_ci": (ci_low <= 0 <= ci_high)
    })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = BASE_DIR / "real_estate_inflation_hedge_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"\nSaved: {summary_path}")