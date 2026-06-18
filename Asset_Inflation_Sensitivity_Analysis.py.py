# Make sure to install this using this command in the terminal gitbash
# python -m pip install numpy pandas matplotlib statsmodels

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# -----------------------------
# Helpers
# -----------------------------

def load_asset(path, date_col="Date", return_col="monthly_return", asset_name=None):
    df = pd.read_csv(path)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.rename(columns={date_col: "Date", return_col: "asset_return"})
    if asset_name is None:
        asset_name = os.path.splitext(os.path.basename(path))[0]
    df["asset_name"] = asset_name
    return df[["Date", "asset_return", "asset_name"] + [c for c in df.columns if c not in ["Date", "asset_return", "asset_name"]]]

def load_inflation(path):
    inf = pd.read_csv(path)
    inf["DATE"] = pd.to_datetime(inf["DATE"])
    inf = inf.rename(columns={"DATE": "Date"})
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
    asset = load_asset(asset_path, asset_name=asset_name)
    inflation = load_inflation(inflation_path)
    df = merge_asset_inflation(asset, inflation)

    # 1) High vs low inflation months
    hl_table = high_low_inflation_summary(df)

    # 2) Simple OLS: asset_return = a + b * HICP_MoM
    model = ols_beta(df["asset_return"], df["HICP_MoM"])

    # 3) Bootstrap CI for beta
    betas, ci90 = bootstrap_beta(df, n_boot=1000)

    # 4) Rolling correlation vs HICP_YoY
    df["roll_corr_12m"] = rolling_corr(df, window=12)
    plot_path = f"{out_prefix}_rolling_corr.png"
    make_rolling_corr_plot(
        df,
        df["roll_corr_12m"],
        plot_path,
        title=f"{asset_name}: 12M rolling correlation with HICP YoY"
    )

    # Save merged analysis data
    df.to_csv(f"{out_prefix}_merged_analysis.csv", index=False)

    # Compact output objects
    results = {
        "asset_name": asset_name,
        "n_obs": len(df),
        "high_low_table": hl_table,
        "ols_model": model,
        "beta": model.params["HICP_MoM"],
        "beta_pvalue": model.pvalues["HICP_MoM"],
        "beta_ci90": ci90,
        "bootstrap_betas": betas,
        "plot_path": plot_path,
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
# Weather-regime deliverables
# -----------------------------
# This is the structure you need for the actual project deliverables.
# It assumes you already have a monthly "weather" column in a dataframe,
# where weather is one of your regimes (e.g. HH, HL, LH, LL).

def bootstrap_group_mean(df, group_col, value_col, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    out = []
    for group_name, g in df.groupby(group_col):
        vals = g[value_col].dropna().to_numpy()
        if len(vals) == 0:
            continue
        means = []
        n = len(vals)
        for _ in range(n_boot):
            sample = rng.choice(vals, size=n, replace=True)
            means.append(np.mean(sample))
        ci_low, ci_high = np.percentile(means, [5, 95])
        out.append({
            "weather": group_name,
            "n": n,
            "mean_return": np.mean(vals),
            "ci90_low": ci_low,
            "ci90_high": ci_high
        })
    return pd.DataFrame(out).sort_values("weather")

def build_weather_summary(weather_df, asset_cols, weather_col="weather", n_boot=1000):
    rows = []
    for asset in asset_cols:
        tmp = weather_df[[weather_col, asset]].dropna().copy()
        tmp = tmp.rename(columns={asset: "ret"})
        stats = bootstrap_group_mean(tmp, weather_col, "ret", n_boot=n_boot)
        stats.insert(1, "asset", asset)
        rows.append(stats)
    return pd.concat(rows, ignore_index=True)

def backtest_weather_portfolio(df, weather_col, asset_return_cols, weights_by_weather):
    """
    df must contain:
      - Date
      - weather_col
      - one return column per asset, with monthly returns for that month
    Backtest rule:
      weather at month t determines portfolio weights for month t+1 return.
    """
    df = df.sort_values("Date").copy()

    # Next month returns are the realized returns after today's weather is observed.
    for c in asset_return_cols:
        df[c + "_next"] = df[c].shift(-1)

    port_rets = []
    dates = []

    for i in range(len(df) - 1):
        weather = df.iloc[i][weather_col]
        if weather not in weights_by_weather:
            continue

        weights = weights_by_weather[weather]
        r_next = 0.0
        valid = True
        for asset in asset_return_cols:
            w = weights.get(asset, 0.0)
            x = df.iloc[i][asset + "_next"]
            if pd.isna(x):
                valid = False
                break
            r_next += w * x

        if valid:
            port_rets.append(r_next)
            dates.append(df.iloc[i + 1]["Date"])

    out = pd.DataFrame({"Date": dates, "portfolio_return": port_rets})
    out["wealth"] = (1 + out["portfolio_return"]).cumprod()
    return out

def portfolio_metrics(ret_series):
    ret = pd.Series(ret_series).dropna()
    ann_return = (1 + ret).prod() ** (12 / len(ret)) - 1
    ann_vol = ret.std(ddof=1) * np.sqrt(12)
    sharpe = ann_return / ann_vol if ann_vol != 0 else np.nan
    wealth = (1 + ret).cumprod()
    peak = wealth.cummax()
    drawdown = wealth / peak - 1
    max_dd = drawdown.min()
    return {
        "CAGR": ann_return,
        "Volatility": ann_vol,
        "Sharpe": sharpe,
        "MaxDrawdown": max_dd,
        "FinalWealth": wealth.iloc[-1]
    }

def compare_to_8020(weather_backtest_df, equity_col="equity_return_next", bond_col="bond_return_next"):
    # Static 80/20 benchmark
    bench = 0.8 * weather_backtest_df[equity_col] + 0.2 * weather_backtest_df[bond_col]
    adaptive = weather_backtest_df["portfolio_return"]

    bench_m = portfolio_metrics(bench)
    adap_m = portfolio_metrics(adaptive)

    summary = pd.DataFrame([adap_m, bench_m], index=["Adaptive", "80/20"])
    return summary, bench

# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":
    inflation_path = r"C:\Users\kieve\Downloads\eu_inflation.csv"

    assets = [
        (r"C:\Users\kieve\Downloads\inflation_bonds.csv",
         "Inflation Linked Bonds",
         "inflation_bonds"),
        (r"C:\Users\kieve\Downloads\eu_equities.csv", "Euro Stoxx 50", "eu_equities"),
        (r"C:\Users\kieve\Downloads\vusa.csv", "VUSA", "vusa"),
    ]

    all_results = []
    for asset_path, asset_name, prefix in assets:
        res = analyze_asset(asset_path, inflation_path, asset_name, prefix)
        format_inflation_result(res)
        all_results.append(res)

    # Optional export of the key regression/CI summary table
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
    summary_df.to_csv(r"C:\Users\kieve\Downloads\inflation_hedge_summary.csv", index=False)
    print("\nSaved: /mnt/data/inflation_hedge_summary.csv")

    # Example plotting the two rolling-correlation series together if needed:
    # load the saved merged files and compare on same axes later.

# Example input:
# weather_df columns:
# Date, weather, equity_return, bond_return

weights_by_weather = {
    "HH": {"equity_return": 0.70, "bond_return": 0.30},
    "HL": {"equity_return": 0.80, "bond_return": 0.20},
    "LH": {"equity_return": 0.40, "bond_return": 0.60},
    "LL": {"equity_return": 0.60, "bond_return": 0.40},
}
