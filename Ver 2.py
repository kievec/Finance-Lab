from pathlib import Path
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

BASE_DIR = Path(r"C:\Users\kieve\Downloads")
INFLATION_PATH = BASE_DIR / "eu_inflation.csv"


# -----------------------------
# Helpers
# -----------------------------
def load_asset(path, date_col="Date", return_col="monthly_return", asset_name=None):
    df = pd.read_csv(path)

    if date_col not in df.columns:
        raise KeyError(f"{path} is missing date column '{date_col}'")
    if return_col not in df.columns:
        raise KeyError(f"{path} is missing return column '{return_col}'")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.rename(columns={date_col: "Date", return_col: "asset_return"})

    if asset_name is None:
        asset_name = os.path.splitext(os.path.basename(path))[0]

    df["asset_name"] = asset_name

    return df[["Date", "asset_return", "asset_name"] + [
        c for c in df.columns if c not in ["Date", "asset_return", "asset_name"]
    ]]


def load_inflation(path):
    inf = pd.read_csv(path)

    if "DATE" not in inf.columns:
        raise KeyError(f"{path} is missing expected date column 'DATE'")

    inf["DATE"] = pd.to_datetime(inf["DATE"])
    inf = inf.rename(columns={"DATE": "Date"})

    needed = ["Date", "HICP", "HICP_YoY", "HICP_MoM", "HICP_12m_avg", "high_inflation"]
    return inf[needed]


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


def analyze_asset(asset_path, inflation_path, asset_name, out_prefix, date_col="Date", return_col="monthly_return"):
    asset = load_asset(asset_path, date_col=date_col, return_col=return_col, asset_name=asset_name)
    inflation = load_inflation(inflation_path)
    df = merge_asset_inflation(asset, inflation)

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


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    assets = [
        (BASE_DIR / "inflation_bonds.csv", "Inflation Linked Bonds", "inflation_bonds", "Date", "monthly_return"),
        (BASE_DIR / "eu_equities.csv", "Euro Stoxx 50", "eu_equities", "Date", "monthly_return"),
        (BASE_DIR / "vusa.csv", "VUSA", "vusa", "Date", "monthly_return"),
        (BASE_DIR / "gold_eur.csv", "Gold (EUR)", "gold_eur", "Date", "monthly_return"),
        (BASE_DIR / "euro_bonds.csv", "Euro Govt Bonds (IEGA.AS)", "euro_bonds", "Date", "monthly_return"),
        (BASE_DIR / "eur_cash.csv", "EUR Cash (ESTR)", "eur_cash", "DATE", "monthly_return"),
    ]

    all_results = []

    for asset_path, asset_name, prefix, date_col, return_col in assets:
        res = analyze_asset(
            asset_path=asset_path,
            inflation_path=INFLATION_PATH,
            asset_name=asset_name,
            out_prefix=prefix,
            date_col=date_col,
            return_col=return_col
        )
        format_inflation_result(res)
        all_results.append(res)

    summary_rows = []
    for res in all_results:
        ci_low, ci_high = res["beta_ci90"]
        summary_rows.append({
            "asset": res["asset_name"],
            "n_obs": res["n_obs"],
            "beta_on_HICP_MoM": res["beta"],
            "p_value": res["beta_pvalue"],
            "ci90_low": ci_low,
            "ci90_high": ci_high,
            "zero_in_ci": (ci_low <= 0 <= ci_high)
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = BASE_DIR / "inflation_hedge_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"\nSaved: {summary_path}")