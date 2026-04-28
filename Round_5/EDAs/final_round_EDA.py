import numpy as np
import pandas as pd
import plotly.express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
from scipy.optimize import brentq
from matplotlib import pyplot as plt

# ── Load & merge ──────────────────────────────────────────────────────────────
prices_1 = pd.read_csv('ROUND_5/prices_round_5_day_2.csv', delimiter=';')
prices_2 = pd.read_csv('ROUND_5/prices_round_5_day_3.csv', delimiter=';')
prices_3 = pd.read_csv('ROUND_5/prices_round_5_day_4.csv', delimiter=';')
trades_1 = pd.read_csv('ROUND_5/trades_round_5_day_2.csv', delimiter=';')
trades_2 = pd.read_csv('ROUND_5/trades_round_5_day_3.csv', delimiter=';')
trades_3 = pd.read_csv('ROUND_5/trades_round_5_day_4.csv', delimiter=';')

trades_1 = pd.merge(trades_1, prices_1['day'], left_index=True, right_index=True)
trades_2 = pd.merge(trades_2, prices_2['day'], left_index=True, right_index=True)
trades_3 = pd.merge(trades_3, prices_3['day'], left_index=True, right_index=True)

prices = pd.concat([prices_1, prices_2, prices_3]).reset_index(drop=True)
trades = pd.concat([trades_1, trades_2, trades_3]).reset_index(drop=True)

prices['global_ts'] = prices['day'] * 1_000_000 + prices['timestamp']
trades['global_ts'] = trades['day'] * 1_000_000 + trades['timestamp']

CATEGORIES = {
    "panels":      ["PANEL_1X2","PANEL_2X2","PANEL_1X4","PANEL_2X4","PANEL_4X4"],
    "pebbles":     ["PEBBLES_XS","PEBBLES_S","PEBBLES_M","PEBBLES_L","PEBBLES_XL"],
    "uv_visor":    ["UV_VISOR_YELLOW","UV_VISOR_AMBER","UV_VISOR_ORANGE","UV_VISOR_RED","UV_VISOR_MAGENTA"],
    "robots":      ["ROBOT_VACUUMING","ROBOT_MOPPING","ROBOT_DISHES","ROBOT_LAUNDRY","ROBOT_IRONING"],
    "galaxy":      ["GALAXY_SOUNDS_DARK_MATTER","GALAXY_SOUNDS_BLACK_HOLES","GALAXY_SOUNDS_PLANETARY_RINGS","GALAXY_SOUNDS_SOLAR_WINDS","GALAXY_SOUNDS_SOLAR_FLAMES"],
    "translators": ["TRANSLATOR_SPACE_GRAY","TRANSLATOR_ASTRO_BLACK","TRANSLATOR_ECLIPSE_CHARCOAL","TRANSLATOR_GRAPHITE_MIST","TRANSLATOR_VOID_BLUE"],
    "sleep_pods":  ["SLEEP_POD_SUEDE","SLEEP_POD_LAMB_WOOL","SLEEP_POD_POLYESTER","SLEEP_POD_NYLON","SLEEP_POD_COTTON"],
    "microchips":  ["MICROCHIP_CIRCLE","MICROCHIP_OVAL","MICROCHIP_SQUARE","MICROCHIP_RECTANGLE","MICROCHIP_TRIANGLE"],
    "oxygen":      ["OXYGEN_SHAKE_MORNING_BREATH","OXYGEN_SHAKE_EVENING_BREATH","OXYGEN_SHAKE_MINT","OXYGEN_SHAKE_CHOCOLATE","OXYGEN_SHAKE_GARLIC"],
    "snacks":      ["SNACKPACK_CHOCOLATE","SNACKPACK_VANILLA","SNACKPACK_PISTACHIO","SNACKPACK_STRAWBERRY","SNACKPACK_RASPBERRY"],
}


def get_mid(df, product):
    rows = df[df["product"] == product].copy().sort_values("timestamp")
    rows["mid"] = (rows["bid_price_1"] + rows["ask_price_1"]) / 2
    rows["label"] = product.split("_", 2)[-1]
    return rows[["timestamp", "mid", "label"]]


# --- Plot 1: Mid price lines, one subplot per category ---
fig = make_subplots(rows=10, cols=1, subplot_titles=list(CATEGORIES.keys()), shared_xaxes=False)

for i, (cat, products) in enumerate(CATEGORIES.items(), start=1):
    for p in products:
        try:
            s = get_mid(prices, p)
            fig.add_trace(
                go.Scatter(x=s["timestamp"], y=s["mid"], name=s["label"].iloc[0],
                           mode="lines", line=dict(width=1.5),
                           legendgroup=cat, showlegend=True),
                row=i, col=1
            )
        except Exception as e:
            print(f"Skipping {p}: {e}")

fig.update_layout(height=3000, title_text="Round 5 — mid prices by category", title_font_size=16)
fig.write_html("eda_all_categories.html")
fig.show()

# --- Plot 2: Normalised prices (all start at 1.0) — makes co-movement visible ---
fig2 = make_subplots(rows=10, cols=1, subplot_titles=[f"{k} (normalised)" for k in CATEGORIES], shared_xaxes=False)

for i, (cat, products) in enumerate(CATEGORIES.items(), start=1):
    for p in products:
        try:
            s = get_mid(prices, p)
            s["normed"] = s["mid"] / s["mid"].iloc[0]
            fig2.add_trace(
                go.Scatter(x=s["timestamp"], y=s["normed"], name=s["label"].iloc[0],
                           mode="lines", line=dict(width=1.5),
                           legendgroup=cat, showlegend=True),
                row=i, col=1
            )
        except Exception as e:
            print(f"Skipping {p}: {e}")

fig2.update_layout(height=3000, title_text="Round 5 — normalised mid prices (start=1.0)", title_font_size=16)
fig2.write_html("eda_normalised.html")
fig2.show()

# --- Ratio check: Panels and Pebbles ---
print("\n=== PANEL RATIO CHECK ===")
panel_mids = {}
for p in CATEGORIES["panels"]:
    try:
        panel_mids[p] = get_mid(prices, p)["mid"].mean()
    except:
        pass
base = panel_mids.get("PANEL_1X2", 1)
for p, v in panel_mids.items():
    print(f"  {p}: avg mid = {v:.2f}  |  ratio vs 1X2 = {v/base:.3f}")

print("\n=== PEBBLE RATIO CHECK ===")
pebble_mids = {}
for p in CATEGORIES["pebbles"]:
    try:
        pebble_mids[p] = get_mid(prices, p)["mid"].mean()
    except:
        pass
base_p = pebble_mids.get("PEBBLES_XS", 1)
for p, v in pebble_mids.items():
    print(f"  {p}: avg mid = {v:.2f}  |  ratio vs XS = {v/base_p:.3f}")

# --- Correlation matrix per category ---
print("\n=== WITHIN-CATEGORY CORRELATIONS ===")
for cat, products in CATEGORIES.items():
    try:
        mids = pd.DataFrame({
            p.split("_", 2)[-1]: get_mid(prices, p).set_index("timestamp")["mid"]
            for p in products
        }).dropna()
        corr = mids.corr()
        upper = corr.values[np.triu_indices_from(corr.values, k=1)]
        print(f"  {cat}: avg pairwise corr = {upper.mean():.3f}  |  min = {upper.min():.3f}  |  max = {upper.max():.3f}")

        # heatmap per category
        fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                             zmin=-1, zmax=1, title=f"Correlation — {cat}")
        fig_corr.write_html(f"corr_{cat}.html")
    except Exception as e:
        print(f"  {cat}: error — {e}")

from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import itertools


# ── Helper: aligned mid price matrix ─────────────────────────────────────────
def build_mid_matrix(price_df, products):
    frames = {}
    for p in products:
        s = price_df[price_df["product"] == p].copy().sort_values("timestamp")
        s["mid"] = (s["bid_price_1"] + s["ask_price_1"]) / 2
        frames[p] = s.set_index("timestamp")["mid"]
    return pd.DataFrame(frames).dropna()


ALL_PRODUCTS = [p for prods in CATEGORIES.values() for p in prods]
mid_matrix = build_mid_matrix(prices, ALL_PRODUCTS)


# ── Half-life fix ─────────────────────────────────────────────────────────────
# Bug was: spread.diff() and spread.shift(1) produce NaNs at different indices,
# and dropna() on each separately before concat causes misalignment.
# Fix: compute both from the same series, concat first, then dropna once.
def calc_half_life(spread):
    spread = spread.reset_index(drop=True)  # ensure clean integer index
    df = pd.DataFrame({
        "lag": spread.shift(1),
        "diff": spread.diff()
    }).dropna()  # single dropna on both columns together — alignment guaranteed

    if len(df) < 20:
        return np.nan

    res = OLS(df["diff"], add_constant(df["lag"])).fit()
    lam = res.params["lag"]

    # lam should be negative for mean reversion; if >= 0 it's trending/random walk
    if lam >= 0:
        return np.nan

    hl = -np.log(2) / lam
    return round(hl, 1)


# Quick sanity check on a synthetic mean-reverting series
_test = pd.Series(np.cumsum(np.random.randn(500)) * 0.1)  # near-random-walk
_test_mr = pd.Series([np.sin(i / 10) + np.random.randn() * 0.1 for i in range(500)])  # strong MR
print(f"Half-life sanity — random walk: {calc_half_life(_test)} (expect NaN or very large)")
print(f"Half-life sanity — mean-reverting: {calc_half_life(_test_mr)} (expect ~30)")

# ── All 1225 pairs ────────────────────────────────────────────────────────────
pairs = list(itertools.combinations(ALL_PRODUCTS, 2))
print(f"\nRunning {len(pairs)} pairs through cointegration + ADF...")

results = []

for i, (a, b) in enumerate(pairs):
    if i % 100 == 0:
        print(f"  {i}/{len(pairs)}...")

    try:
        s_a = mid_matrix[a]
        s_b = mid_matrix[b]

        # Engle-Granger cointegration
        coint_stat, coint_pval, _ = coint(s_a, s_b)

        # Hedge ratio via OLS (a ~ b)
        res = OLS(s_a, add_constant(s_b)).fit()
        hedge_ratio = res.params[b]

        # Spread
        spread = s_a - hedge_ratio * s_b

        # ADF on spread
        adf_stat, adf_pval, adf_lags, *_ = adfuller(spread, autolag="AIC")

        # Half-life
        hl = calc_half_life(spread)

        results.append({
            "product_a": a,
            "product_b": b,
            "cat_a": next(k for k, v in CATEGORIES.items() if a in v),
            "cat_b": next(k for k, v in CATEGORIES.items() if b in v),
            "coint_pval": round(coint_pval, 4),
            "adf_pval": round(adf_pval, 4),
            "hedge_ratio": round(hedge_ratio, 4),
            "spread_mean": round(spread.mean(), 2),
            "spread_std": round(spread.std(), 2),
            "half_life": hl,
        })

    except Exception as e:
        results.append({
            "product_a": a, "product_b": b,
            "cat_a": None, "cat_b": None,
            "coint_pval": np.nan, "adf_pval": np.nan,
            "hedge_ratio": np.nan, "spread_mean": np.nan,
            "spread_std": np.nan, "half_life": np.nan,
        })

results_df = pd.DataFrame(results)

# ── ADF p-val matrix (50×50) ──────────────────────────────────────────────────
adf_matrix = pd.DataFrame(np.nan, index=ALL_PRODUCTS, columns=ALL_PRODUCTS)
coint_matrix = pd.DataFrame(np.nan, index=ALL_PRODUCTS, columns=ALL_PRODUCTS)

for _, row in results_df.iterrows():
    adf_matrix.loc[row["product_a"], row["product_b"]] = row["adf_pval"]
    adf_matrix.loc[row["product_b"], row["product_a"]] = row["adf_pval"]  # symmetric
    coint_matrix.loc[row["product_a"], row["product_b"]] = row["coint_pval"]
    coint_matrix.loc[row["product_b"], row["product_a"]] = row["coint_pval"]

fig_adf = px.imshow(
    adf_matrix,
    color_continuous_scale="RdYlGn_r",
    zmin=0, zmax=0.1,
    title="ADF p-value Matrix — all 1225 pairs (green = low p-val = stationary spread)",
    width=1400, height=1200
)
fig_adf.write_html("adf_matrix.html")
fig_adf.show()

fig_coint = px.imshow(
    coint_matrix,
    color_continuous_scale="RdYlGn_r",
    zmin=0, zmax=0.1,
    title="Cointegration p-value Matrix — all 1225 pairs",
    width=1400, height=1200
)
fig_coint.write_html("coint_matrix.html")
fig_coint.show()

# ── Extract passing pairs ─────────────────────────────────────────────────────
COINT_THRESH = 0.05
ADF_THRESH = 0.05

passing = results_df[
    (results_df["coint_pval"] < COINT_THRESH) &
    (results_df["adf_pval"] < ADF_THRESH)
    ].copy().sort_values("adf_pval")

print(f"\n=== PAIRS PASSING BOTH TESTS (coint<{COINT_THRESH}, ADF<{ADF_THRESH}) ===")
print(f"  Found: {len(passing)}\n")
print(passing[[
    "product_a", "product_b", "cat_a", "cat_b",
    "coint_pval", "adf_pval", "hedge_ratio", "spread_std", "half_life"
]].to_string(index=False))

# If nothing passes, show top 20 by ADF alone as a fallback
if len(passing) == 0:
    print("\n  Nothing passed both. Top 20 by ADF p-val alone:")
    top20 = results_df.dropna(subset=["adf_pval"]).nsmallest(20, "adf_pval")
    print(top20[[
        "product_a", "product_b", "cat_a", "cat_b",
        "coint_pval", "adf_pval", "hedge_ratio", "spread_std", "half_life"
    ]].to_string(index=False))

results_df.to_csv("all_pairs_results.csv", index=False)
print("\nFull results saved to all_pairs_results.csv")
