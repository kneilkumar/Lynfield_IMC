import pandas as pd
import glob
import matplotlib.pyplot as plt

# =========================
# LOAD ALL DAYS
# =========================
files = sorted(glob.glob("prices_round_3_day_*.csv"))

df_list = []
day_offset = 0

for f in files:
    temp = pd.read_csv(f, sep=";")

    temp["timestamp"] = pd.to_numeric(temp["timestamp"], errors="coerce")
    temp["mid_price"] = pd.to_numeric(temp["mid_price"], errors="coerce")

    temp["timestamp"] = temp["timestamp"] + day_offset
    day_offset = temp["timestamp"].max() + 1

    df_list.append(temp)

df = pd.concat(df_list)

# =========================
# FILTER PRODUCTS
# =========================
velvet = df[df["product"] == "VELVETFRUIT_EXTRACT"].copy()
hydrogel = df[df["product"] == "HYDROGEL_PACK"].copy()

# =========================
# SORT (IMPORTANT for metrics)
# =========================
velvet = velvet.sort_values("timestamp")
hydrogel = hydrogel.sort_values("timestamp")

# =========================
# RETURNS (volatility base)
# =========================
velvet["returns"] = velvet["mid_price"].pct_change()
hydrogel["returns"] = hydrogel["mid_price"].pct_change()

# =========================
# ROLLING VOLATILITY
# =========================
window = 50

velvet["volatility"] = velvet["returns"].rolling(window).std()
hydrogel["volatility"] = hydrogel["returns"].rolling(window).std()

# =========================
# Z-SCORE (mean reversion signal)
# =========================
velvet["zscore"] = (velvet["mid_price"] - velvet["mid_price"].mean()) / velvet["mid_price"].std()
hydrogel["zscore"] = (hydrogel["mid_price"] - hydrogel["mid_price"].mean()) / hydrogel["mid_price"].std()

# =========================
# 1. PRICE (KEEP YOUR ORIGINAL)
# =========================
plt.figure()
plt.plot(velvet["timestamp"], velvet["mid_price"])
plt.title("Velvetfruit Price (Full Timeline)")
plt.show()

plt.figure()
plt.plot(hydrogel["timestamp"], hydrogel["mid_price"])
plt.title("Hydrogel Price (Full Timeline)")
plt.show()

# =========================
# 2. RETURNS (VOLATILITY BASE)
# =========================
plt.figure()
plt.plot(velvet["timestamp"], velvet["returns"])
plt.title("Velvetfruit Returns (Noise / Movement Speed)")
plt.show()

plt.figure()
plt.plot(hydrogel["timestamp"], hydrogel["returns"])
plt.title("Hydrogel Returns (Noise / Movement Speed)")
plt.show()

# =========================
# 3. ROLLING VOLATILITY (KEY PLOT)
# =========================
plt.figure()
plt.plot(velvet["timestamp"], velvet["volatility"], label="Velvetfruit")
plt.plot(hydrogel["timestamp"], hydrogel["volatility"], label="Hydrogel")
plt.title("Rolling Volatility (window=50)")
plt.legend()
plt.show()

# =========================
# 4. Z-SCORE (MEAN REVERSION STRENGTH)
# =========================
plt.figure()
plt.plot(velvet["timestamp"], velvet["zscore"], label="Velvetfruit")
plt.plot(hydrogel["timestamp"], hydrogel["zscore"], label="Hydrogel")
plt.axhline(0, linestyle="--")
plt.title("Z-Score (Mean Reversion Signal Strength)")
plt.legend()
plt.show()