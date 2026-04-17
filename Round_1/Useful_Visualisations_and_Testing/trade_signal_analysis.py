import pandas as pd

# =========================
# SETTINGS
# =========================
PRODUCT = "ASH_COATED_OSMIUM"  # change to osmium later

files = [
    "../Data/trades_round_1_day_-2.csv",
    "../Data/trades_round_1_day_-1.csv",
    "../Data/trades_round_1_day_0.csv"
]

# =========================
# LOAD + FILTER
# =========================
df = pd.concat([pd.read_csv(f, sep=";") for f in files])

# clean column names (important)
df.columns = df.columns.str.strip()

# filter by product
df = df[df["symbol"] == PRODUCT].copy()

df = df.sort_values("timestamp").reset_index(drop=True)

# =========================
# STEP 1: TRADE SIZES
# =========================
print(f"\n=== {PRODUCT} ===")
print("\nTop trade sizes:")
size_counts = df["quantity"].value_counts()
print(size_counts.head(10))

target_size = size_counts.index[0]
print(f"\nTesting size: {target_size}")

# =========================
# STEP 2: PRICE IMPACT
# =========================
df["next_price"] = df["price"].shift(-1)

df_target = df[df["quantity"] == target_size].copy()
df_target["price_change"] = df_target["next_price"] - df_target["price"]

# =========================
# RESULTS
# =========================
print("\n=== PRICE CHANGE ===")
print(df_target["price_change"].describe())

mean_move = df_target["price_change"].mean()
print(f"\nMean price change: {mean_move:.4f}")

# =========================
# INTERPRETATION
# =========================
if mean_move > 0.1:
    print("\n✅ BUY SIGNAL")
elif mean_move < -0.1:
    print("\n✅ SELL SIGNAL")
else:
    print("\n❌ NO SIGNAL")