import numpy as np
import pandas as pd
import plotly.express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
from scipy.optimize import brentq
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from numpy.fft import fft, fftfreq
import matplotlib.pyplot as plt

# ── Load & merge ──────────────────────────────────────────────────────────────
prices_1 = pd.read_csv('ROUND_4/prices_round_4_day_1.csv', delimiter=';')
prices_2 = pd.read_csv('ROUND_4/prices_round_4_day_2.csv', delimiter=';')
prices_3 = pd.read_csv('ROUND_4/prices_round_4_day_3.csv', delimiter=';')
trades_1 = pd.read_csv('ROUND_4/trades_round_4_day_1.csv', delimiter=';')
trades_2 = pd.read_csv('ROUND_4/trades_round_4_day_2.csv', delimiter=';')
trades_3 = pd.read_csv('ROUND_4/trades_round_4_day_3.csv', delimiter=';')

trades_1 = pd.merge(trades_1, prices_1['day'], left_index=True, right_index=True)
trades_2 = pd.merge(trades_2, prices_2['day'], left_index=True, right_index=True)
trades_3 = pd.merge(trades_3, prices_3['day'], left_index=True, right_index=True)

prices = pd.concat([prices_1, prices_2, prices_3]).reset_index(drop=True)
trades = pd.concat([trades_1, trades_2, trades_3]).reset_index(drop=True)

print(list(prices.columns.values))
print(list(trades.columns.values))
print("Products:", prices['product'].unique())

prices['global_ts'] = prices['day'] * 1_000_000 + prices['timestamp']
trades['global_ts'] = trades['day'] * 1_000_000 + trades['timestamp']
trades['qty_normalised'] = trades['quantity'] / trades['quantity'].abs().max()

prices['spread_1'] = prices['ask_price_1'] - prices['bid_price_1']
prices['spread_2'] = prices['ask_price_2'] - prices['bid_price_2']
prices['spread_3'] = prices['ask_price_3'] - prices['bid_price_3']

EXTRACT = 'VELVETFRUIT_EXTRACT'
GEL     = 'HYDROGEL_PACK'
VOUCHER_STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VOUCHER_NAMES   = [f'VEV_{k}' for k in VOUCHER_STRIKES]

extract_price = prices[prices['product'] == EXTRACT]
gel_price     = prices[prices['product'] == GEL]
extract_price = extract_price[extract_price['mid_price'] != 0.0]
gel_price     = gel_price[gel_price['mid_price'] != 0.0]

extract_trades = trades[trades['symbol'] == EXTRACT]
gel_trades     = trades[trades['symbol'] == GEL]

voucher_prices = {name: prices[(prices['product'] == name) & (prices['mid_price'] != 0.0)]
                  for name in VOUCHER_NAMES}
voucher_trades = {name: trades[trades['symbol'] == name] for name in VOUCHER_NAMES}

spread = extract_price['mid_price'].values - (0.6465 * gel_price['mid_price'].values + 11709.13)
print("Spread std:", spread.std())
print("Spread mean:", spread.mean())
print("Spread max:", np.abs(spread.max()))
print("Spread min:", spread.min())


# ══════════════════════════════════════════════════════════════════════════════
# EXISTING HELPERS — labels and titles fixed throughout
# ══════════════════════════════════════════════════════════════════════════════

def wall_mid_prices(price_df):
    bid_df = price_df[['bid_volume_1', 'bid_volume_2', 'bid_volume_3',
                        'bid_price_1',  'bid_price_2',  'bid_price_3']].fillna(0)
    ask_df = price_df[['ask_volume_1', 'ask_volume_2', 'ask_volume_3',
                        'ask_price_1',  'ask_price_2',  'ask_price_3']].fillna(0)
    weighted_bids = (bid_df['bid_volume_1'] * bid_df['bid_price_1'] +
                     bid_df['bid_volume_2'] * bid_df['bid_price_2'] +
                     bid_df['bid_volume_3'] * bid_df['bid_price_3'])
    total_bid_vol = bid_df['bid_volume_1'] + bid_df['bid_volume_2'] + bid_df['bid_volume_3']
    bid_vwap      = weighted_bids / total_bid_vol
    weighted_asks = (ask_df['ask_volume_1'] * ask_df['ask_price_1'] +
                     ask_df['ask_volume_2'] * ask_df['ask_price_2'] +
                     ask_df['ask_volume_3'] * ask_df['ask_price_3'])
    total_ask_vol = ask_df['ask_volume_1'] + ask_df['ask_volume_2'] + ask_df['ask_volume_3']
    ask_vwap      = weighted_asks / total_ask_vol
    price_df['vwap_estimate'] = (bid_vwap + ask_vwap) / 2

    order_book_bids = price_df[['bid_price_1', 'bid_volume_1', 'bid_price_2',
                                 'bid_volume_2', 'bid_price_3', 'bid_volume_3']].dropna(how='all')
    order_book_asks = price_df[['ask_price_1', 'ask_volume_1', 'ask_price_2',
                                 'ask_volume_2', 'ask_price_3', 'ask_volume_3']].dropna(how='all')

    order_book_bids['wall_bid_price'] = order_book_bids[['bid_volume_1', 'bid_volume_2', 'bid_volume_3']].idxmax(axis=1, skipna=True)
    order_book_bids['wall_bid_price'] = np.where(order_book_bids['wall_bid_price'] == 'bid_volume_1', order_book_bids['bid_price_1'],
                                         np.where(order_book_bids['wall_bid_price'] == 'bid_volume_2', order_book_bids['bid_price_2'],
                                                  order_book_bids['bid_price_3']))
    order_book_asks['wall_ask_price'] = order_book_asks[['ask_volume_1', 'ask_volume_2', 'ask_volume_3']].idxmax(axis=1, skipna=True)
    order_book_asks['wall_ask_price'] = np.where(order_book_asks['wall_ask_price'] == 'ask_volume_1', order_book_asks['ask_price_1'],
                                         np.where(order_book_asks['wall_ask_price'] == 'ask_volume_2', order_book_asks['ask_price_2'],
                                                  order_book_asks['ask_price_3']))
    wall_mids = pd.merge(order_book_asks['wall_ask_price'], order_book_bids['wall_bid_price'],
                         left_index=True, right_index=True)
    wall_mids['wall_mid_price'] = (wall_mids['wall_ask_price'] + wall_mids['wall_bid_price']) / 2
    return price_df.join(wall_mids[['wall_mid_price']], how='left')


def total_volume_plot(price_df, mid_df):
    price_df['total_ask_vol'] = price_df['ask_volume_1'].fillna(0) + price_df['ask_volume_2'].fillna(0) + price_df['ask_volume_3'].fillna(0)
    price_df['total_bid_vol'] = price_df['bid_volume_1'].fillna(0) + price_df['bid_volume_2'].fillna(0) + price_df['bid_volume_3'].fillna(0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=price_df['global_ts'], y=price_df['total_ask_vol'], line=dict(color='blue'),  name='total ask vol', yaxis='y1'))
    fig.add_trace(go.Scatter(x=price_df['global_ts'], y=price_df['total_bid_vol'], line=dict(color='red'),   name='total bid vol', yaxis='y1'))
    fig.add_trace(go.Scatter(x=price_df['global_ts'], y=mid_df['mid_price'],       line=dict(color='green'), name='mid price',     yaxis='y2'))
    fig.update_layout(title='Total Order Book Volume vs Mid Price',
                      xaxis_title='global timestamp',
                      yaxis=dict(title='order book volume'),
                      yaxis2=dict(title='mid price', overlaying='y', side='right'),
                      legend=dict(x=0.01, y=0.99))
    return fig


def microprice(price_df):
    ba = price_df[['bid_volume_1', 'bid_volume_2', 'bid_volume_3',
                   'bid_price_1',  'bid_price_2',  'bid_price_3',
                   'ask_volume_1', 'ask_volume_2', 'ask_volume_3',
                   'ask_price_1',  'ask_price_2',  'ask_price_3']].fillna(0)
    for i in [1, 2, 3]:
        num = ba[f'bid_price_{i}'] * ba[f'ask_volume_{i}'] + ba[f'ask_price_{i}'] * ba[f'bid_volume_{i}']
        ba[f'microprice_{i}'] = num / (ba[f'ask_volume_{i}'] + ba[f'bid_volume_{i}'] + 1e-8)
    return ba


def imbalances(price_df):
    for i in [1, 2, 3]:
        price_df[f'imbalance_{i}'] = ((price_df[f'bid_volume_{i}'] - price_df[f'ask_volume_{i}']) /
                                      (price_df[f'bid_volume_{i}'] + price_df[f'ask_volume_{i}']))
    total_bid = price_df['bid_volume_1'] + price_df['bid_volume_2'] + price_df['bid_volume_3']
    total_ask = price_df['ask_volume_1'] + price_df['ask_volume_2'] + price_df['ask_volume_3']
    price_df['total_imbalance'] = (total_bid - total_ask) / (total_bid + total_ask)
    return price_df


def resampler(df, resample_freq):
    df = df.drop('product', axis=1)
    df = df.groupby(np.arange(len(df)) // resample_freq).mean()
    return df


def overlay_trade_data(figure, trade_df, y_value, buy_or_sell):
    trade_scatter = px.scatter(data_frame=trade_df, x='global_ts', y=y_value, color=buy_or_sell)
    for trace in trade_scatter.data:
        trace.showlegend = True
        figure.add_trace(trace)
    return figure


def plot_spike_trajectories(price_df, window=20, threshold_multiplier=2.0, lookahead=200):
    price_df = price_df.reset_index(drop=True)
    rolling_mean = price_df['wall_mid_price'].rolling(window).mean()
    rolling_std  = price_df['wall_mid_price'].rolling(window).std()
    spike_mask   = (price_df['wall_mid_price'] - rolling_mean).abs() > threshold_multiplier * rolling_std

    while spike_mask.sum() == 0 and threshold_multiplier > 0.5:
        threshold_multiplier -= 0.25
        spike_mask = (price_df['wall_mid_price'] - rolling_mean).abs() > threshold_multiplier * rolling_std

    spike_indices = price_df.index[spike_mask].tolist()
    print(f"threshold_multiplier={threshold_multiplier:.2f}, spikes found: {len(spike_indices)}")

    trajectories = []
    for idx in spike_indices:
        if idx + lookahead >= len(price_df):
            continue
        window_prices = price_df['wall_mid_price'].iloc[idx:idx + lookahead].values
        trajectories.append(window_prices - window_prices[0])

    if not trajectories:
        print("Still no spikes found")
        return None

    avg_trajectory = np.mean(trajectories, axis=0)
    fig = go.Figure()
    for t in trajectories:
        fig.add_trace(go.Scatter(y=t, mode='lines', line=dict(color='rgba(150,150,150,1)'), showlegend=False))
    fig.add_trace(go.Scatter(y=avg_trajectory, mode='lines', line=dict(color='red', width=3), name='average trajectory'))
    fig.update_layout(title=f'Post-Spike Price Trajectories ({len(trajectories)} spikes, threshold={threshold_multiplier:.2f}σ)',
                      xaxis_title='ticks after spike',
                      yaxis_title='price change from spike',
                      legend=dict(x=0.01, y=0.99))
    return fig


def plot_imbalance(price_df, trade_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=price_df['global_ts'], y=price_df['imbalance_1'], line=dict(color='blue'),   name='imbalance level 1'))
    fig.add_trace(go.Scatter(x=price_df['global_ts'], y=price_df['imbalance_2'], line=dict(color='orange'), name='imbalance level 2'))
    fig.add_trace(go.Scatter(x=price_df['global_ts'], y=price_df['imbalance_3'], line=dict(color='green'),  name='imbalance level 3'))
    fig.add_trace(go.Scatter(x=trade_df['global_ts'], y=trade_df['quantity'], mode='markers',
                             marker=dict(color='rgba(200,0,0,0.7)', size=6), name='trade quantity'))
    fig.update_layout(title='Order Book Imbalance by Level with Trade Quantity',
                      xaxis_title='global timestamp',
                      yaxis_title='imbalance (-1=sell side, +1=buy side)',
                      legend=dict(x=0.01, y=0.99))
    return fig


def plot_prices(mid_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mid_df['global_ts'], y=mid_df['mid_price'],      line=dict(color='blue'),   name='mid price'))
    fig.add_trace(go.Scatter(x=mid_df['global_ts'], y=mid_df['wall_mid_price'], line=dict(color='red'),    name='wall mid price'))
    fig.add_trace(go.Scatter(x=mid_df['global_ts'], y=mid_df['vwap_estimate'],  line=dict(color='green'),  name='vwap estimate'))
    fig.add_trace(go.Scatter(x=mid_df['global_ts'], y=mid_df['microprice_1'],   line=dict(color='orange'), name='microprice level 1'))
    fig.add_trace(go.Scatter(x=mid_df['global_ts'], y=mid_df['microprice_2'],   line=dict(color='purple'), name='microprice level 2'))
    fig.add_trace(go.Scatter(x=mid_df['global_ts'], y=mid_df['microprice_3'],   line=dict(color='brown'),  name='microprice level 3'))
    fig.update_layout(title='Fair Value Estimates Over Time',
                      xaxis_title='global timestamp',
                      yaxis_title='price',
                      legend=dict(x=0.01, y=0.99))
    return fig


def plot_divergence(mid_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mid_df['global_ts'], y=mid_df['microprice_1'] - mid_df['mid_price'],
                             line=dict(color='orange'), name='microprice 1 divergence', yaxis='y1'))
    fig.add_trace(go.Scatter(x=mid_df['global_ts'], y=mid_df['microprice_2'] - mid_df['mid_price'],
                             line=dict(color='purple'), name='microprice 2 divergence', yaxis='y1'))
    fig.add_trace(go.Scatter(x=mid_df['global_ts'], y=mid_df['microprice_3'] - mid_df['mid_price'],
                             line=dict(color='brown'),  name='microprice 3 divergence', yaxis='y1'))
    fig.add_trace(go.Scatter(x=mid_df['global_ts'], y=mid_df['mid_price'],
                             line=dict(color='green'),  name='mid price', yaxis='y2'))
    fig.add_hline(y=0, line=dict(color='black', width=0.5))
    fig.update_layout(title='Microprice Divergence from Mid (positive = microprice above mid)',
                      xaxis_title='global timestamp',
                      yaxis=dict(title='divergence (price ticks)'),
                      yaxis2=dict(title='mid price', overlaying='y', side='right'),
                      legend=dict(x=0.01, y=0.99))
    return fig


def add_vlines(price_df, figure):
    newdays = [np.argmax(price_df['day'] == 1), np.argmax(price_df['day'] == 2), np.argmax(price_df['day'] == 3)]
    ts = [price_df['global_ts'].iloc[newdays[0]], price_df['global_ts'].iloc[newdays[1]], price_df['global_ts'].iloc[newdays[2]]]
    for t in ts:
        figure.add_vline(x=t, line_color='red', line_dash='dash',
                         annotation_text='day boundary', annotation_position='top right')
    return figure


def imbalance_fv_plot(price_df, mid_df):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=mid_df['global_ts'], y=mid_df['wall_mid_price'], name='wall mid price'), secondary_y=False)
    fig.add_trace(go.Scatter(x=mid_df['global_ts'], y=mid_df['mid_price'],      name='mid price'),      secondary_y=False)
    fig.add_trace(go.Scatter(x=price_df['global_ts'], y=price_df['imbalance_3'], name='imbalance level 3'), secondary_y=True)
    fig.update_layout(title='Fair Value vs Level 3 Imbalance',
                      xaxis_title='global timestamp',
                      legend=dict(x=0.01, y=0.99))
    fig.update_yaxes(title_text='price', secondary_y=False)
    fig.update_yaxes(title_text='imbalance (-1 to +1)', secondary_y=True)
    return fig


def plot_spread(price_df, spread):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=price_df['global_ts'], y=price_df[spread],
                             line=dict(color='blue'), name=spread))
    fig.update_layout(title=f'Bid-Ask {spread} Over Time',
                      xaxis_title='global timestamp',
                      yaxis_title='spread (price ticks)',
                      legend=dict(x=0.01, y=0.99))
    return fig


def plotting_wrapper(product_price, product_trades, resample, resample_freq):
    if resample:
        product_price = resampler(product_price, resample_freq)
    microprice_info  = microprice(product_price)
    midprice_info    = wall_mid_prices(product_price)
    midprice_info    = pd.merge(midprice_info, microprice_info, left_index=True, right_index=True)
    imbalance_info   = imbalances(product_price)
    divplot          = plot_divergence(midprice_info)
    midplot          = plot_prices(midprice_info)
    midplot          = overlay_trade_data(midplot, product_trades, 'price', buy_or_sell='buyer')
    midplot2 =  overlay_trade_data(midplot, product_trades, 'price', buy_or_sell='seller')
    volplot          = total_volume_plot(product_price, midprice_info)
    spreadplot       = plot_spread(product_price, 'spread_1')
    imbalanceplot    = plot_imbalance(imbalance_info, product_trades)
    spikes           = plot_spike_trajectories(midprice_info)
    imbalance_mid_pl = imbalance_fv_plot(imbalance_info, midprice_info)
    for plots in [midplot, midplot2, divplot, volplot, spreadplot, imbalanceplot, imbalance_mid_pl]:
        plots = add_vlines(product_price, plots)
        plots.show()
    if spikes:
        spikes.show()


# ══════════════════════════════════════════════════════════════════════════════
# NEW: TEMPORAL STRUCTURE ANALYSIS (ACF + FFT)
# ══════════════════════════════════════════════════════════════════════════════

def analyse_temporal_structure(price_df, product_name, series_col='mid_price', lags=200):
    """
    Run ACF and FFT on a price series to check for temporal structure.
    ACF: slow decay = random walk/momentum. Oscillating = mean reversion/cycle.
    FFT: dominant spike at frequency f = cycle every 1/f ticks.
    Runs per day and on full series.
    """
    series = price_df[series_col].dropna()

    # ACF
    fig, ax = plt.subplots(figsize=(12, 4))
    plot_acf(series, lags=lags, ax=ax)
    ax.set_title(f'{product_name} — Autocorrelation of {series_col} (lags={lags})')
    ax.set_xlabel('lag (ticks)')
    ax.set_ylabel('autocorrelation')
    plt.tight_layout()
    plt.show()

    # FFT — full series
    signal = series.values
    freqs  = fftfreq(len(signal))
    power  = np.abs(fft(signal)) ** 2
    pos    = freqs > 0
    dom_freq  = freqs[pos][np.argmax(power[pos])]
    dom_cycle = int(1 / dom_freq) if dom_freq > 0 else 0
    print(f"\n{product_name} — Full series dominant cycle: {dom_cycle} ticks")

    # FFT — per day
    freqs = fftfreq(len(price_df[series_col].dropna()))
    power = np.abs(fft(price_df[series_col].dropna().values)) ** 2
    pos = freqs > 0  # derived from full freqs, same size
    dom_freq = freqs[pos][np.argmax(power[pos])]
    dom_cycle = int(1 / dom_freq) if dom_freq > 0 else 0

    for day in sorted(price_df['day'].unique()):
        subset = price_df[price_df['day'] == day][series_col].dropna().values
        if len(subset) < 10:
            continue
        f = fftfreq(len(subset))
        p = np.abs(fft(subset)) ** 2
        pos_day = f > 0  # local mask, only used inside loop
        df_ = f[pos_day][np.argmax(p[pos_day])]
        cyc = int(1 / df_) if df_ > 0 else 0
        print(f"  Day {day} dominant cycle: {cyc} ticks")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=freqs[pos], y=power[pos], mode='lines',
                             line=dict(color='steelblue'), name='power'))
    fig.add_vline(x=dom_freq, line=dict(color='red', dash='dash'),
                  annotation_text=f'dominant: {dom_cycle} ticks')


# ══════════════════════════════════════════════════════════════════════════════
# NEW: CROSS-PRODUCT RELATIONSHIP ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyse_product_relationships(price_df_a, price_df_b, name_a, name_b):
    """
    Check correlation and cointegration between two products.
    Correlation: how much they move together.
    Cointegration: whether their spread is stationary (stable long-run relationship).
    Also computes the cointegrating vector via Johansen.
    """
    combined = pd.merge_asof(
        price_df_a[['global_ts', 'mid_price']].rename(columns={'mid_price': name_a}).sort_values('global_ts'),
        price_df_b[['global_ts', 'mid_price']].rename(columns={'mid_price': name_b}).sort_values('global_ts'),
        on='global_ts', direction='nearest'
    ).dropna()

    # Correlation
    corr = combined[name_a].corr(combined[name_b])
    print(f"\n{name_a} vs {name_b}")
    print(f"  Pearson correlation: {corr:.4f}")

    # Returns correlation
    combined[f'{name_a}_ret'] = combined[name_a].diff()
    combined[f'{name_b}_ret'] = combined[name_b].diff()
    ret_corr = combined[f'{name_a}_ret'].corr(combined[f'{name_b}_ret'])
    print(f"  Returns correlation: {ret_corr:.4f}")

    # Simple ratio
    combined['ratio']  = combined[name_a] / combined[name_b]
    combined['spread'] = combined[name_a] - combined[name_b]
    print(f"  Ratio  mean={combined['ratio'].mean():.4f}  std={combined['ratio'].std():.4f}")
    print(f"  Spread mean={combined['spread'].mean():.2f}  std={combined['spread'].std():.2f}")

    # Engle-Granger cointegration test
    score, pvalue, crits = coint(combined[name_a], combined[name_b])
    print(f"  Cointegration p-value: {pvalue:.4f}  ({'cointegrated' if pvalue < 0.05 else 'NOT cointegrated'})")

    from statsmodels.tsa.vector_ar.vecm import coint_johansen
    # Johansen cointegrating vector
    jres = coint_johansen(combined[[name_a, name_b]], det_order=0, k_ar_diff=1)
    beta = jres.evec[:, 0]
    beta_norm = beta / beta[0]
    print(f"  Johansen cointegrating vector: {name_a} - {-beta_norm[1]:.4f} * {name_b}")

    # Cointegration spread using Johansen vector
    combined['coint_spread'] = combined[name_a] + beta_norm[1] * combined[name_b]
    print(f"  Cointegration spread mean={combined['coint_spread'].mean():.2f}  std={combined['coint_spread'].std():.2f}")
    print(f"  Spread autocorr lag-1: {combined['coint_spread'].autocorr(1):.4f}")

    # Lead-lag cross correlation
    print(f"\n  Lead-lag cross-correlation ({name_a} vs {name_b}):")
    for lag in range(-5, 6):
        cc = combined[f'{name_a}_ret'].corr(combined[f'{name_b}_ret'].shift(lag))
        print(f"    lag {lag:+d}: {cc:.4f}")

    combined = pd.DataFrame({
        'extract': extract_price['mid_price'].values,
        'gel': gel_price['mid_price'].values,
    })

    result = coint_johansen(combined, det_order=0, k_ar_diff=1)
    print("Eigenvectors (columns are cointegrating vectors):")
    print(result.evec)
    print("\nEigenvalues:")
    print(result.eig)
    print("\nTrace statistic:")
    print(result.lr1)
    print("\nCritical values (90%, 95%, 99%):")
    print(result.cvt)

    # Plot ratio and spread over time
    fig = make_subplots(rows=2, cols=1, subplot_titles=[
        f'{name_a}/{name_b} ratio over time',
        f'Cointegration spread ({name_a} - {-beta_norm[1]:.3f}*{name_b}) over time'
    ])
    fig.add_trace(go.Scatter(x=combined['global_ts'], y=combined['ratio'],
                             line=dict(color='steelblue'), name='price ratio'), row=1, col=1)
    fig.add_trace(go.Scatter(x=combined['global_ts'], y=combined['coint_spread'],
                             line=dict(color='darkorange'), name='coint spread'), row=2, col=1)
    fig.update_xaxes(title_text='global timestamp', row=1, col=1)
    fig.update_xaxes(title_text='global timestamp', row=2, col=1)
    fig.update_yaxes(title_text='ratio', row=1, col=1)
    fig.update_yaxes(title_text='spread (price units)', row=2, col=1)
    fig.update_layout(title=f'{name_a} vs {name_b} — Relationship Analysis',
                      legend=dict(x=0.01, y=0.99))
    fig.show()

    return combined


# ══════════════════════════════════════════════════════════════════════════════
# NEW: MARKOUT ANALYSIS (ADVERSE SELECTION)
# ══════════════════════════════════════════════════════════════════════════════

def compute_markout(trade_df, price_df, horizons=[10, 50, 100, 200]):
    """
    For each executed trade, measure where mid price ends up N ticks later.
    Markout = mid[t+N] - mid[t].
    Negative average markout = adverse selection (market moves against your fill).
    Near-zero = fair fills, market making is clean.
    """
    price_df = price_df[['global_ts', 'mid_price']].sort_values('global_ts')
    trade_df = trade_df.sort_values('global_ts').copy()

    trade_df = pd.merge_asof(trade_df,
                             price_df.rename(columns={'mid_price': 'mid_at_trade'}),
                             on='global_ts', direction='nearest')

    for h in horizons:
        future = price_df.copy()
        future['global_ts'] = future['global_ts'] - h
        future = future.rename(columns={'mid_price': f'mid_t{h}'})
        trade_df = pd.merge_asof(trade_df, future, on='global_ts', direction='nearest')
        trade_df[f'markout_{h}'] = trade_df[f'mid_t{h}'] - trade_df['mid_at_trade']

    return trade_df


def plot_markout_profile(marked_df, product_name, horizons=[10, 50, 100, 200]):
    """
    Average markout at each horizon.
    Flat at zero = fair fills.
    Drifts negative = being adversely selected (market moves against you after fill).
    Drifts positive = fills are favourable (momentum).
    """
    avgs = [marked_df[f'markout_{h}'].mean() for h in horizons]
    stds = [marked_df[f'markout_{h}'].std()  for h in horizons]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=horizons, y=avgs, mode='lines+markers',
                             line=dict(color='red', width=2), name='avg markout',
                             error_y=dict(type='data', array=stds, visible=True)))
    fig.add_hline(y=0, line=dict(color='black', dash='dash'),
                  annotation_text='zero (fair fills)')
    fig.update_layout(title=f'{product_name} — Average Markout Profile (adverse selection check)',
                      xaxis_title='horizon (ticks after trade)',
                      yaxis_title='average mid price change after fill',
                      legend=dict(x=0.01, y=0.99))
    fig.show()


def plot_markout_by_size(marked_df, product_name, horizons=[10, 50, 100, 200]):
    """
    Split trades into small and large by median quantity.
    Large trades with worse markout = informed flow.
    """
    median_qty = marked_df['quantity'].abs().median()
    small = marked_df[marked_df['quantity'].abs() <= median_qty]
    large = marked_df[marked_df['quantity'].abs() >  median_qty]

    fig = go.Figure()
    for subset, label, col in [(small, 'small trades', 'steelblue'), (large, 'large trades', 'crimson')]:
        avgs = [subset[f'markout_{h}'].mean() for h in horizons]
        fig.add_trace(go.Scatter(x=horizons, y=avgs, mode='lines+markers',
                                 line=dict(color=col, width=2), name=label))
    fig.add_hline(y=0, line=dict(color='black', dash='dash'))
    fig.update_layout(title=f'{product_name} — Markout by Trade Size (large = potential informed flow)',
                      xaxis_title='horizon (ticks after trade)',
                      yaxis_title='average mid price change after fill',
                      legend=dict(x=0.01, y=0.99))
    fig.show()


def plot_trade_vs_mid(marked_df, product_name):
    """
    Distribution of trade price vs mid price at time of trade.
    Centred at zero = mid is fair value.
    Skewed = systematic bias in quote placement.
    """
    marked_df['dev_from_mid'] = marked_df['price'] - marked_df['mid_at_trade']
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=marked_df['dev_from_mid'], nbinsx=50,
                               marker_color='steelblue', opacity=0.8, name='trade price vs mid'))
    fig.add_vline(x=0, line=dict(color='red', dash='dash'), annotation_text='mid price')
    fig.add_vline(x=marked_df['dev_from_mid'].mean(), line=dict(color='green', dash='dot'),
                  annotation_text=f"mean={marked_df['dev_from_mid'].mean():.2f}")
    fig.update_layout(title=f'{product_name} — Trade Price Deviation from Mid (spread structure)',
                      xaxis_title='trade price - mid price at trade time',
                      yaxis_title='trade count',
                      legend=dict(x=0.01, y=0.99))
    fig.show()


def markout_wrapper(trade_df, price_df, product_name, horizons=[10, 50, 100, 200]):
    """
    Full markout analysis pipeline for a single product.
    """
    marked = compute_markout(trade_df, price_df, horizons)
    plot_trade_vs_mid(marked, product_name)
    plot_markout_profile(marked, product_name, horizons)
    plot_markout_by_size(marked, product_name, horizons)
    return marked


# ══════════════════════════════════════════════════════════════════════════════
# BLACK-SCHOLES UTILITIES (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def bs_call_price(S, K, T, sigma, r=0.0):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_delta(S, K, T, sigma, r=0.0):
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)


def implied_vol(market_price, S, K, T, r=0.0):
    intrinsic = max(S - K, 0.0)
    if market_price <= intrinsic or S <= 0 or T <= 0:
        return np.nan
    try:
        iv = brentq(lambda sigma: bs_call_price(S, K, T, sigma, r) - market_price,
                    1e-6, 10.0, xtol=1e-6, maxiter=200)
        return iv
    except (ValueError, RuntimeError):
        return np.nan


# ══════════════════════════════════════════════════════════════════════════════
# VOUCHER TABLE (vectorised)
# ══════════════════════════════════════════════════════════════════════════════

_iv_vec    = np.vectorize(implied_vol,  otypes=[float])
_delta_vec = np.vectorize(bs_delta,     otypes=[float])


def build_voucher_wide(extract_df, voucher_prices_dict, T,
                       cache_path='ROUND_3/voucher_wide.csv'):
    try:
        wide = pd.read_csv(cache_path)
        print(f"Loaded wide df from cache: {wide.shape}")
        return wide
    except FileNotFoundError:
        pass

    base = extract_df[['global_ts', 'mid_price']].rename(columns={'mid_price': 'S'})
    for name, K in zip(VOUCHER_NAMES, VOUCHER_STRIKES):
        print(f"Processing {name}...")
        v = (voucher_prices_dict[name][['global_ts', 'mid_price']]
             .rename(columns={'mid_price': f'mid_{name}'}))
        base = pd.merge(base, v, on='global_ts', how='left')
        S   = base['S'].values
        mid = base[f'mid_{name}'].values
        base[f'intrinsic_{name}']  = np.maximum(S - K, 0)
        base[f'time_value_{name}'] = mid - base[f'intrinsic_{name}'].values
        ivs = _iv_vec(mid, S, K, T)
        base[f'iv_{name}'] = ivs
        valid  = ~np.isnan(ivs)
        deltas = np.full(len(base), np.nan)
        deltas[valid] = _delta_vec(S[valid], K, T, ivs[valid])
        base[f'delta_{name}'] = deltas

    base.to_csv(cache_path, index=False)
    print(f"Saved wide df to {cache_path}: {base.shape}")
    return base


# ══════════════════════════════════════════════════════════════════════════════
# VOUCHER PLOTS (titles and labels fixed)
# ══════════════════════════════════════════════════════════════════════════════

def plot_moneyness(extract_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=extract_df['global_ts'], y=extract_df['mid_price'],
                             line=dict(color='black', width=2), name='extract mid price'))
    colours = px.colors.sample_colorscale('RdYlGn', [i / (len(VOUCHER_STRIKES) - 1)
                                                      for i in range(len(VOUCHER_STRIKES))])
    for K, col in zip(VOUCHER_STRIKES, colours):
        fig.add_hline(y=K, line=dict(color=col, dash='dot', width=1),
                      annotation_text=str(K), annotation_position='right')
    fig.update_layout(title='Extract Mid Price vs Voucher Strike Ladder (above line = ITM)',
                      xaxis_title='global timestamp',
                      yaxis_title='price',
                      legend=dict(x=0.01, y=0.99))
    return fig


def plot_iv_smile(wide_df):
    fig = go.Figure()
    sample_ts = wide_df['global_ts'].iloc[::max(1, len(wide_df) // 10)].values
    for ts in sample_ts:
        row = wide_df[wide_df['global_ts'] == ts].iloc[0]
        ivs = [row[f'iv_{name}'] for name in VOUCHER_NAMES]
        fig.add_trace(go.Scatter(x=VOUCHER_STRIKES, y=ivs, mode='lines+markers',
                                 name=f'ts={ts}', opacity=0.6))
    fig.update_layout(title='Implied Volatility Smile — IV vs Strike at Sample Timestamps',
                      xaxis_title='strike price',
                      yaxis_title='implied volatility',
                      legend=dict(x=0.01, y=0.99))
    return fig


def plot_time_value(wide_df):
    fig = go.Figure()
    colours = px.colors.sample_colorscale('Viridis', [i / (len(VOUCHER_NAMES) - 1)
                                                       for i in range(len(VOUCHER_NAMES))])
    for name, col in zip(VOUCHER_NAMES, colours):
        fig.add_trace(go.Scatter(x=wide_df['global_ts'], y=wide_df[f'time_value_{name}'],
                                 line=dict(color=col), name=name))
    fig.add_hline(y=0, line=dict(color='red', dash='dash', width=1),
                  annotation_text='hard arb boundary (negative = buy voucher arb)')
    fig.update_layout(title='Time Value per Voucher Over Time (market price - intrinsic value)',
                      xaxis_title='global timestamp',
                      yaxis_title='time value (price units)',
                      legend=dict(x=0.01, y=0.99))
    return fig


def plot_delta_surface(wide_df):
    fig = go.Figure()
    colours = px.colors.sample_colorscale('Plasma', [i / (len(VOUCHER_NAMES) - 1)
                                                      for i in range(len(VOUCHER_NAMES))])
    for name, col in zip(VOUCHER_NAMES, colours):
        fig.add_trace(go.Scatter(x=wide_df['global_ts'], y=wide_df[f'delta_{name}'],
                                 line=dict(color=col), name=name))
    fig.update_layout(title='BS Delta per Voucher Over Time (1=deep ITM, 0=deep OTM, 0.5=ATM)',
                      xaxis_title='global timestamp',
                      yaxis_title='delta',
                      yaxis=dict(range=[0, 1]),
                      legend=dict(x=0.01, y=0.99))
    return fig


def plot_realised_vol(extract_df, window=20):
    df = extract_df.copy()
    df['log_ret']      = np.log(df['mid_price'] / df['mid_price'].shift(1))
    df['realised_vol'] = df['log_ret'].rolling(window).std()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['global_ts'], y=df['realised_vol'],
                             line=dict(color='navy'), name=f'realised vol (window={window})'))
    fig.update_layout(title=f'Extract Realised Volatility — Rolling {window} Tick Std of Log Returns',
                      xaxis_title='global timestamp',
                      yaxis_title='realised volatility',
                      legend=dict(x=0.01, y=0.99))
    return fig


def plot_return_distribution(extract_df):
    log_rets = np.log(extract_df['mid_price'] / extract_df['mid_price'].shift(1)).dropna()
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=log_rets, nbinsx=100, name='observed log returns',
                               marker_color='steelblue', opacity=0.8))
    mu, sigma = log_rets.mean(), log_rets.std()
    x_range   = np.linspace(log_rets.min(), log_rets.max(), 200)
    normal_pdf = norm.pdf(x_range, mu, sigma) * len(log_rets) * (log_rets.max() - log_rets.min()) / 100
    fig.add_trace(go.Scatter(x=x_range, y=normal_pdf, line=dict(color='red', width=2), name='normal fit'))
    fig.update_layout(title='Extract Log Return Distribution vs Normal (fat tails = BS mispricing)',
                      xaxis_title='log return per tick',
                      yaxis_title='count',
                      legend=dict(x=0.01, y=0.99))
    return fig


def check_arbitrage_bounds(wide_df):
    violations = {'monotonicity': 0, 'call_spread': 0, 'butterfly': 0}
    mid_cols = [f'mid_{name}' for name in VOUCHER_NAMES]
    for i in range(len(VOUCHER_NAMES) - 1):
        c_low  = wide_df[mid_cols[i]]
        c_high = wide_df[mid_cols[i + 1]]
        dk     = VOUCHER_STRIKES[i + 1] - VOUCHER_STRIKES[i]
        violations['monotonicity'] += (c_low < c_high).sum()
        violations['call_spread']  += ((c_low - c_high) > dk).sum()
    for i in range(len(VOUCHER_NAMES) - 2):
        c1 = wide_df[mid_cols[i]]
        c2 = wide_df[mid_cols[i + 1]]
        c3 = wide_df[mid_cols[i + 2]]
        violations['butterfly'] += ((c1 - 2 * c2 + c3) < 0).sum()
    total_rows = len(wide_df)
    print(f"\nArbitrage bound check over {total_rows} timestamps:")
    for key, count in violations.items():
        print(f"  {key}: {count} violations ({100*count/total_rows:.2f}%)")
    return violations


def plot_iv_vs_realised_vol(wide_df, extract_df, window=20):
    df = extract_df.copy().set_index('global_ts')
    df['log_ret']      = np.log(df['mid_price'] / df['mid_price'].shift(1))
    df['realised_vol'] = df['log_ret'].rolling(window).std()
    wide = wide_df.copy()
    iv_cols = [f'iv_{name}' for name in VOUCHER_NAMES]
    def atm_iv(row):
        dists   = [abs(row['S'] - K) for K in VOUCHER_STRIKES]
        atm_idx = int(np.argmin(dists))
        return row[iv_cols[atm_idx]]
    wide['atm_iv'] = wide.apply(atm_iv, axis=1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=wide['global_ts'], y=wide['atm_iv'],
                             line=dict(color='orange'), name='ATM implied vol'))
    fig.add_trace(go.Scatter(x=df.reset_index()['global_ts'], y=df['realised_vol'],
                             line=dict(color='navy'), name=f'realised vol (window={window})'))
    fig.update_layout(title='ATM Implied Vol vs Realised Vol (gap = systematic vol selling edge)',
                      xaxis_title='global timestamp',
                      yaxis_title='volatility',
                      legend=dict(x=0.01, y=0.99))
    return fig


def voucher_plotting_wrapper(extract_df, voucher_prices_dict, T=1.0, rv_window=20):
    wide = build_voucher_wide(extract_df, voucher_prices_dict, T)
    check_arbitrage_bounds(wide)
    plots = [
        plot_moneyness(extract_df),
        plot_iv_smile(wide),
        plot_time_value(wide),
        plot_delta_surface(wide),
        plot_realised_vol(extract_df, rv_window),
        plot_return_distribution(extract_df),
        plot_iv_vs_realised_vol(wide, extract_df, rv_window),
    ]
    for fig in plots:
        fig.show()


# ══════════════════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════════════════

# ── Standard product EDA ──────────────────────────────────────────────────────
# plotting_wrapper(gel_price,     gel_trades,     True, 50)
# plotting_wrapper(extract_price, extract_trades, True, 50)

# ── Temporal structure ────────────────────────────────────────────────────────
# analyse_temporal_structure(gel_price,     'HYDROGEL_PACK')
# analyse_temporal_structure(extract_price, 'VELVETFRUIT_EXTRACT')

# ── Cross-product relationships ───────────────────────────────────────────────
combined = analyse_product_relationships(extract_price, gel_price,
                                         'VELVETFRUIT_EXTRACT', 'HYDROGEL_PACK')

# ── Markout / adverse selection ───────────────────────────────────────────────
# markout_wrapper(gel_trades,     gel_price,     'HYDROGEL_PACK')
# markout_wrapper(extract_trades, extract_price, 'VELVETFRUIT_EXTRACT')
#
# # ── Voucher surface ───────────────────────────────────────────────────────────
# voucher_plotting_wrapper(extract_price, voucher_prices, T=4.0)
