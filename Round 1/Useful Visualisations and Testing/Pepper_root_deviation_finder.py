import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

#Setup
Datafiles = [
    "../Data/prices_round_1_day_-2.csv",
    "../Data/prices_round_1_day_-1.csv",
    "../Data/prices_round_1_day_0.csv"
]
Product = "INTARIAN_PEPPER_ROOT"

#Load Data
def load_data(file_name):
    df = pd.read_csv(file_name, sep = ";")
    df = df[df["product"] == Product].copy()
    df = df[df["mid_price"] > 0]
    df = df.sort_values("timestamp").reset_index(drop = True)
    return df

# #Get wall mid
# def compute_wall_mid(df):

#Linear Regression
def compute_linear_trend(df, total_increase = 1000):
    start_price = df["mid_price"].iloc[0]
    n = len(df)
    slope = total_increase / (n - 1)
    df["trend"] = start_price + slope * np.arange(n)
    return df

#Compute Deviations
def compute_deviations(df):
    df["deviation"] = df["mid_price"] - df["trend"]
    df["deviation_rounded"] = df["deviation"].round().astype(int)
    return df

#Count Deviations
def get_distribution(df):
    counts = Counter(df["deviation_rounded"])
    return dict(sorted(counts.items()))

#Print
def print_distribution(dist):
    print("\n=== Deviation Distribution ===")
    for k, v in dist.items():
        print(f"{k}: {v}")

#Bar Chard
def plot_distribution(dist, title):
    x = list(dist.keys())
    y = list(dist.values())

    plt.figure(figsize=(10, 5))
    plt.bar(x, y)
    plt.title(title)
    plt.xlabel("Deviation (XIRECS)")
    plt.ylabel("Frequency")
    plt.grid()
    plt.show()


for file in Datafiles:
    print(f"\n===== {file} =====")

    df = load_data(file)
    df = compute_linear_trend(df)
    df = compute_deviations(df)
    dist = get_distribution(df)
    print_distribution(dist)
    plot_distribution(dist, title=f"Deviation Distribution - {file}")


