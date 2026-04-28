import pandas as pd
import matplotlib.pyplot as plt

# ─── Load data ─────────────────────────────────────────────
df0 = pd.read_csv('Data/prices_round_3_day_0.csv', delimiter=';')
df1 = pd.read_csv('Data/prices_round_3_day_1.csv', delimiter=';')
df2 = pd.read_csv('Data/prices_round_3_day_2.csv', delimiter=';')

df = pd.concat([df0, df1, df2]).reset_index(drop=True)

# ─── Standardise asset column ──────────────────────────────
if 'product' in df.columns:
    df['asset'] = df['product']
else:
    df['asset'] = df['symbol']

# ─── Remove bad data (no bids/asks) ────────────────────────
df = df[df['mid_price'] > 0].copy()


# ─── Plot each day separately (BEST VIEW) ──────────────────
def plot_asset_by_day(asset_name):
    temp = df[df['asset'] == asset_name].copy()

    days = sorted(temp['day'].unique())

    fig, axes = plt.subplots(len(days), 1, figsize=(12, 8), sharex=False)

    # handle case where only 1 subplot
    if len(days) == 1:
        axes = [axes]

    for i, d in enumerate(days):
        day_data = temp[temp['day'] == d]

        axes[i].plot(day_data['timestamp'], day_data['mid_price'])
        axes[i].set_title(f"{asset_name} — Day {d}")
        axes[i].set_xlabel("Time")
        axes[i].set_ylabel("Price")
        axes[i].grid()

    plt.tight_layout()
    plt.show()


# ─── Run plots ─────────────────────────────────────────────
plot_asset_by_day("VELVETFRUIT_EXTRACT")
plot_asset_by_day("HYDROGEL_PACK")