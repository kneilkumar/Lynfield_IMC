import numpy as np
import pandas as pd
import plotly.express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from multiprocessing import Pool, cpu_count
import itertools

# ── Categories ────────────────────────────────────────────────────────────────
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

ALL_PRODUCTS = [p for prods in CATEGORIES.values() for p in prods]


# ── Helper functions — module level so multiprocessing can import them ─────────

def get_mid(df, product):
    rows = df[df["product"] == product].copy().sort_values("global_ts")
    rows["mid"] = (rows["bid_price_1"] + rows["ask_price_1"]) / 2
    rows["label"] = product.split("_", 2)[-1]
    return rows[["global_ts", "timestamp", "mid", "label"]]


def build_mid_matrix(price_df, products):
    """
    Build aligned mid price matrix indexed by global_ts.
    Using global_ts (not timestamp) so multi-day data doesn't fold back
    on itself — each day's timestamps restart at 0 but global_ts is unique.
    """
    frames = {}
    for p in products:
        s = price_df[price_df["product"] == p].copy().sort_values("global_ts")
        s["mid"] = (s["bid_price_1"] + s["ask_price_1"]) / 2
        frames[p] = s.set_index("global_ts")["mid"]   # ← was timestamp
    return pd.DataFrame(frames).dropna()


def calc_half_life(spread):
    """
    Fixed half-life calculation.
    Build lag and diff from the same series, concat first, dropna once —
    guarantees alignment and kills the 0.5-for-everything artifact.
    """
    spread = pd.Series(spread).reset_index(drop=True)
    df = pd.DataFrame({
        "lag":  spread.shift(1),
        "diff": spread.diff()
    }).dropna()
    if len(df) < 20:
        return np.nan
    res = OLS(df["diff"], add_constant(df["lag"])).fit()
    lam = res.params.iloc[1]
    if lam >= 0:
        return np.nan  # not mean-reverting — trending or random walk
    return round(-np.log(2) / lam, 1)


def test_pair(args):
    """
    Worker function for parallelised pair testing.
    Module-level so multiprocessing can pickle it.
    Takes raw numpy arrays to keep pickling fast.
    """
    a, b, arr_a, arr_b = args
    data_a = pd.Series(arr_a)
    data_b = pd.Series(arr_b)
    try:
        _, coint_pval, _ = coint(data_a, data_b)
        res         = OLS(data_a, add_constant(data_b)).fit()
        hedge_ratio = res.params.iloc[1]
        spread      = data_a - hedge_ratio * data_b
        _, adf_pval, *_ = adfuller(spread, autolag="AIC")
        hl          = calc_half_life(spread.values)
        cat_a       = next(k for k, v in CATEGORIES.items() if a in v)
        cat_b       = next(k for k, v in CATEGORIES.items() if b in v)
        return {
            "product_a":   a,
            "product_b":   b,
            "cat_a":       cat_a,
            "cat_b":       cat_b,
            "coint_pval":  round(coint_pval, 4),
            "adf_pval":    round(adf_pval, 4),
            "hedge_ratio": round(hedge_ratio, 4),
            "spread_mean": round(spread.mean(), 2),
            "spread_std":  round(spread.std(), 2),
            "half_life":   hl,
        }
    except Exception:
        return {
            "product_a": a, "product_b": b,
            "cat_a": None, "cat_b": None,
            "coint_pval": np.nan, "adf_pval": np.nan,
            "hedge_ratio": np.nan, "spread_mean": np.nan,
            "spread_std": np.nan, "half_life": np.nan,
        }


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # ── Load & merge ──────────────────────────────────────────────────────────
    prices_1 = pd.read_csv('ROUND_5/prices_round_5_day_2.csv', delimiter=';')
    prices_2 = pd.read_csv('ROUND_5/prices_round_5_day_3.csv', delimiter=';')
    prices_3 = pd.read_csv('ROUND_5/prices_round_5_day_4.csv', delimiter=';')
    trades_1 = pd.read_csv('ROUND_5/trades_round_5_day_2.csv', delimiter=';')
    trades_2 = pd.read_csv('ROUND_5/trades_round_5_day_3.csv', delimiter=';')  # ← was day_2
    trades_3 = pd.read_csv('ROUND_5/trades_round_5_day_4.csv', delimiter=';')

    # Attach day column to trades via index-aligned merge with the prices file
    trades_1 = pd.merge(trades_1, prices_1['day'], left_index=True, right_index=True)
    trades_2 = pd.merge(trades_2, prices_2['day'], left_index=True, right_index=True)
    trades_3 = pd.merge(trades_3, prices_3['day'], left_index=True, right_index=True)

    prices = pd.concat([prices_1, prices_2, prices_3]).reset_index(drop=True)
    trades = pd.concat([trades_1, trades_2, trades_3]).reset_index(drop=True)

    # global_ts is the single monotonic time axis used everywhere below
    prices['global_ts'] = prices['day'] * 1_000_000 + prices['timestamp']
    trades['global_ts'] = trades['day'] * 1_000_000 + trades['timestamp']

    # ── Plot 1: Raw mid prices per category ───────────────────────────────────
    fig = make_subplots(rows=10, cols=1, subplot_titles=list(CATEGORIES.keys()), shared_xaxes=False)
    for i, (cat, products) in enumerate(CATEGORIES.items(), start=1):
        for p in products:
            try:
                s = get_mid(prices, p)
                fig.add_trace(
                    go.Scatter(x=s["global_ts"], y=s["mid"], name=s["label"].iloc[0],
                               mode="lines", line=dict(width=1.5),
                               legendgroup=cat, showlegend=True),
                    row=i, col=1
                )
            except Exception as e:
                print(f"Skipping {p}: {e}")
    fig.update_layout(height=3000, title_text="Round 5 — mid prices by category", title_font_size=16)
    fig.write_html("eda_all_categories.html")
    fig.show()

    # ── Plot 2: Normalised prices (all start at 1.0) ──────────────────────────
    fig2 = make_subplots(rows=10, cols=1, subplot_titles=[f"{k} (normalised)" for k in CATEGORIES], shared_xaxes=False)
    for i, (cat, products) in enumerate(CATEGORIES.items(), start=1):
        for p in products:
            try:
                s = get_mid(prices, p)
                s = s.copy()
                s["normed"] = s["mid"] / s["mid"].iloc[0]
                fig2.add_trace(
                    go.Scatter(x=s["global_ts"], y=s["normed"], name=s["label"].iloc[0],
                               mode="lines", line=dict(width=1.5),
                               legendgroup=cat, showlegend=True),
                    row=i, col=1
                )
            except Exception as e:
                print(f"Skipping {p}: {e}")
    fig2.update_layout(height=3000, title_text="Round 5 — normalised mid prices (start=1.0)", title_font_size=16)
    fig2.write_html("eda_normalised.html")
    fig2.show()

    # ── Ratio checks ──────────────────────────────────────────────────────────
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

    # ── Within-category correlations + per-category heatmaps ──────────────────
    # Uses global_ts as index so multi-day data aligns correctly
    print("\n=== WITHIN-CATEGORY CORRELATIONS ===")
    for cat, products in CATEGORIES.items():
        try:
            mids = pd.DataFrame({
                p.split("_", 2)[-1]: get_mid(prices, p).set_index("global_ts")["mid"]  # ← was timestamp
                for p in products
            }).dropna()
            corr  = mids.corr()
            upper = corr.values[np.triu_indices_from(corr.values, k=1)]
            print(f"  {cat}: avg pairwise corr = {upper.mean():.3f}  |  min = {upper.min():.3f}  |  max = {upper.max():.3f}")
            fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                                 zmin=-1, zmax=1, title=f"Correlation — {cat}")
            fig_corr.write_html(f"corr_{cat}.html")
        except Exception as e:
            print(f"  {cat}: error — {e}")

    # ── Build full mid matrix ──────────────────────────────────────────────────
    print("\nBuilding mid price matrix for all 50 products...")
    mid_matrix = build_mid_matrix(prices, ALL_PRODUCTS)
    print(f"  Matrix shape: {mid_matrix.shape}  (timestamps × products)")

    # ── Half-life sanity check ─────────────────────────────────────────────────
    np.random.seed(42)
    _rw = pd.Series(np.cumsum(np.random.randn(500)) * 0.1)
    _mr = pd.Series([np.sin(i / 10) + np.random.randn() * 0.1 for i in range(500)])
    print(f"\nHalf-life sanity — random walk   : {calc_half_life(_rw.values)}  (expect large)")
    print(f"Half-life sanity — mean-reverting: {calc_half_life(_mr.values)}  (expect ~30)")

    # ── Full 50×50 correlation heatmap ────────────────────────────────────────
    full_corr = mid_matrix.corr()
    fig_full  = px.imshow(
        full_corr, text_auto=False,
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        title="Full 50×50 Cross-Category Correlation Heatmap",
        width=1400, height=1200
    )
    fig_full.write_html("corr_full_50x50.html")
    fig_full.show()

    # ── PCA across all 50 products ────────────────────────────────────────────
    scaler    = StandardScaler()
    scaled    = scaler.fit_transform(mid_matrix.dropna())
    pca       = PCA()
    pca.fit(scaled)
    explained  = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    fig_pca = make_subplots(rows=1, cols=2,
                             subplot_titles=["Variance per Component", "Cumulative Variance"])
    fig_pca.add_trace(go.Bar(y=explained[:20], name="Per component"), row=1, col=1)
    fig_pca.add_trace(go.Scatter(y=cumulative[:20], mode="lines+markers", name="Cumulative"), row=1, col=2)
    fig_pca.add_hline(y=0.8, line_dash="dash", line_color="red", row=1, col=2)
    fig_pca.update_layout(title_text="PCA — Variance Explained (first 20 components)", height=500)
    fig_pca.write_html("pca_scree.html")
    fig_pca.show()

    loadings = pd.DataFrame(
        pca.components_[:5],
        columns=mid_matrix.columns,
        index=[f"PC{i+1}" for i in range(5)]
    )
    fig_load = px.imshow(loadings, text_auto=".2f", color_continuous_scale="RdBu_r",
                         zmin=-0.3, zmax=0.3,
                         title="PCA Loadings — PC1 to PC5 across all 50 products",
                         height=500, width=1400)
    fig_load.write_html("pca_loadings.html")
    fig_load.show()

    print("\n=== PCA SUMMARY ===")
    for i in range(10):
        print(f"  PC{i+1}: {explained[i]*100:.1f}%  (cumulative: {cumulative[i]*100:.1f}%)")

    # ── Per-day average mid prices (spot regime changes) ──────────────────────
    print("\n=== PER-DAY AVG MID PRICE ===")
    for cat, products in CATEGORIES.items():
        print(f"\n  {cat}:")
        for p in products:
            day_avgs = []
            for day_df in [prices_1, prices_2, prices_3]:
                rows = day_df[day_df["product"] == p].copy()
                if len(rows):
                    mid = (rows["bid_price_1"] + rows["ask_price_1"]) / 2
                    day_avgs.append(f"{mid.mean():.0f}")
                else:
                    day_avgs.append("N/A")
            print(f"    {p:<45} day avgs: {' | '.join(day_avgs)}")

    # ── All 1225 pairs — parallelised ─────────────────────────────────────────
    # First run: uncomment the Pool block, comment out the read_csv line.
    # # Subsequent runs: comment out Pool block, uncomment read_csv to save time.
    # pairs_args = [
    #     (a, b, mid_matrix[a].values, mid_matrix[b].values)
    #     for a, b in itertools.combinations(ALL_PRODUCTS, 2)
    # ]
    # n_cores = cpu_count()
    # print(f"\nRunning {len(pairs_args)} pairs across {n_cores} cores...")
    # with Pool(processes=n_cores) as pool:
    #     results = pool.map(test_pair, pairs_args)
    # results_df = pd.DataFrame(results)
    # results_df.to_csv("all_pairs_results.csv", index=False)
    # print("Pair results saved to all_pairs_results.csv")

    results_df = pd.read_csv("all_pairs_results.csv")  # ← uncomment after first run

    # ── ADF and cointegration p-value matrices ────────────────────────────────
    adf_matrix   = pd.DataFrame(np.nan, index=ALL_PRODUCTS, columns=ALL_PRODUCTS)
    coint_matrix = pd.DataFrame(np.nan, index=ALL_PRODUCTS, columns=ALL_PRODUCTS)
    for _, row in results_df.iterrows():
        a, b = row["product_a"], row["product_b"]
        adf_matrix.loc[a, b]   = row["adf_pval"]
        adf_matrix.loc[b, a]   = row["adf_pval"]
        coint_matrix.loc[a, b] = row["coint_pval"]
        coint_matrix.loc[b, a] = row["coint_pval"]

    fig_adf = px.imshow(
        adf_matrix, color_continuous_scale="RdYlGn_r", zmin=0, zmax=0.1,
        title="ADF p-value Matrix — all 1225 pairs  (green = stationary spread)",
        width=1400, height=1200
    )
    fig_adf.write_html("adf_matrix.html")
    fig_adf.show()

    fig_coint = px.imshow(
        coint_matrix, color_continuous_scale="RdYlGn_r", zmin=0, zmax=0.1,
        title="Cointegration p-value Matrix — all 1225 pairs",
        width=1400, height=1200
    )
    fig_coint.write_html("coint_matrix.html")
    fig_coint.show()

    # ── Extract pairs passing both tests ──────────────────────────────────────
    COINT_THRESH = 0.05
    ADF_THRESH   = 0.05

    passing = results_df[
        (results_df["coint_pval"] < COINT_THRESH) &
        (results_df["adf_pval"]   < ADF_THRESH)
    ].copy().sort_values("adf_pval")

    print(f"\n=== PAIRS PASSING BOTH TESTS (coint < {COINT_THRESH}, ADF < {ADF_THRESH}) ===")
    print(f"  Found: {len(passing)}\n")
    if len(passing) > 0:
        print(passing[[
            "product_a","product_b","cat_a","cat_b",
            "coint_pval","adf_pval","hedge_ratio","spread_std","half_life"
        ]].to_string(index=False))
    else:
        print("  Nothing passed both. Top 20 by ADF p-val alone:\n")
        top20 = results_df.dropna(subset=["adf_pval"]).nsmallest(20, "adf_pval")
        print(top20[[
            "product_a","product_b","cat_a","cat_b",
            "coint_pval","adf_pval","hedge_ratio","spread_std","half_life"
        ]].to_string(index=False))

    # ── Amber hub analysis ─────────────────────────────────────────────────────
    amber_pairs = results_df[
        (results_df["product_a"].str.contains("AMBER") | results_df["product_b"].str.contains("AMBER")) &
        (results_df["coint_pval"] < 0.05) &
        (results_df["adf_pval"]   < 0.05)
    ].copy()

    print("\n=== AMBER PAIRS PASSING BOTH TESTS ===")
    print(amber_pairs[[
        "product_a","product_b",
        "coint_pval","adf_pval",
        "hedge_ratio","spread_mean","spread_std","half_life"
    ]].to_string(index=False))

    # ── Autocorrelation — amber ────────────────────────────────────────────────
    amber_mid = mid_matrix["UV_VISOR_AMBER"]

    fig_ac, axes = plt.subplots(2, 1, figsize=(14, 8))
    plot_acf(amber_mid.dropna(), lags=50, ax=axes[0],
             title="UV_VISOR_AMBER — mid price autocorrelation")
    plot_acf(amber_mid.diff().dropna(), lags=50, ax=axes[1],
             title="UV_VISOR_AMBER — first difference autocorrelation")
    plt.tight_layout()
    plt.savefig("amber_autocorr.png", dpi=150)
    plt.show()

    print("\n=== AMBER AUTOCORRELATION (numerical) ===")
    for lag in range(1, 6):
        print(f"  lag-{lag}: {amber_mid.autocorr(lag=lag):+.4f}")

    print("\n=== LAG-1 AUTOCORR — ALL VISORS ===")
    for p in CATEGORIES["uv_visor"]:
        print(f"  {p:<30} lag-1: {mid_matrix[p].autocorr(1):+.4f}")

    # ── Lag-1 autocorrelation + median move — all 50 products ─────────────────
    print("\n=== LAG-1 AUTOCORR + MEDIAN MOVE — ALL 50 PRODUCTS ===")
    ac_results = []
    for p in ALL_PRODUCTS:
        s = mid_matrix[p]
        ac_results.append({
            "product":     p,
            "category":    next(k for k, v in CATEGORIES.items() if p in v),
            "lag1":        round(s.autocorr(1), 4),
            "lag2":        round(s.autocorr(2), 4),
            "lag3":        round(s.autocorr(3), 4),
            "median_move": round(s.diff().abs().median(), 2),
        })
    ac_df = pd.DataFrame(ac_results).sort_values("lag1")
    print(ac_df.to_string(index=False))
    print(f"\n  lag-1 < -0.2 (strong reversion): {(ac_df['lag1'] < -0.2).sum()} products")
    print(f"  lag-1 < -0.1 (mild reversion):    {(ac_df['lag1'] < -0.1).sum()} products")

    # ── Bid-ask spread vs median move — top tier products ─────────────────────
    TOP_TIER = [
        "MICROCHIP_SQUARE", "MICROCHIP_OVAL", "UV_VISOR_AMBER",
        "SLEEP_POD_POLYESTER", "PANEL_1X4", "MICROCHIP_TRIANGLE",
        "SLEEP_POD_SUEDE", "GALAXY_SOUNDS_PLANETARY_RINGS",
        "SLEEP_POD_COTTON", "ROBOT_LAUNDRY", "PANEL_2X2"
    ]

    print("\n=== BID-ASK SPREAD vs MEDIAN MOVE — TOP TIER ===")
    for p in TOP_TIER:
        rows      = prices[prices["product"] == p].copy()
        ba_spread = (rows["ask_price_1"] - rows["bid_price_1"]).median()
        med_move  = mid_matrix[p].diff().abs().median()
        print(f"  {p:<35} bid-ask: {ba_spread:.1f}  |  move: {med_move:.1f}  |  ratio: {med_move/ba_spread:.2f}x")
