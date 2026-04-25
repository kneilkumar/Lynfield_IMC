import numpy as np
import pandas as pd
import plotly.express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
from scipy.optimize import brentq
from matplotlib import pyplot as plt

# ── Load & merge ──────────────────────────────────────────────────────────────
prices_1 = pd.read_csv('Round_3/Data/prices_round_3_day_0.csv', delimiter=';')
prices_2 = pd.read_csv('Round_3/Data/prices_round_3_day_1.csv', delimiter=';')
prices_3 = pd.read_csv('Round_3/Data/prices_round_3_day_2.csv', delimiter=';')
trades_1 = pd.read_csv('Round_3/Data/trades_round_3_day_0.csv', delimiter=';')
trades_2 = pd.read_csv('Round_3/Data/trades_round_3_day_1.csv', delimiter=';')
trades_3 = pd.read_csv('Round_3/Data/trades_round_3_day_2.csv', delimiter=';')

trades_1 = pd.merge(trades_1, prices_1['day'], left_index=True, right_index=True)
trades_2 = pd.merge(trades_2, prices_2['day'], left_index=True, right_index=True)
trades_3 = pd.merge(trades_3, prices_3['day'], left_index=True, right_index=True)

prices = pd.concat([prices_1, prices_2, prices_3]).reset_index(drop=True)
trades = pd.concat([trades_1, trades_2, trades_3]).reset_index(drop=True)

print(list(prices.columns.values))
print(list(trades.columns.values))
print("Products:", prices['product'].unique())

# Days are 0,1,2 so no offset needed
prices['global_ts'] = prices['day'] * 1_000_000 + prices['timestamp']
trades['global_ts'] = trades['day'] * 1_000_000 + trades['timestamp']
trades['qty_normalised'] = trades['quantity'] / trades['quantity'].abs().max()

prices['spread_1'] = prices['ask_price_1'] - prices['bid_price_1']
prices['spread_2'] = prices['ask_price_2'] - prices['bid_price_2']
prices['spread_3'] = prices['ask_price_3'] - prices['bid_price_3']

# ── Product names — adjust if the CSV uses different strings ──────────────────
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

voucher_prices = {name: prices[prices['product'] == name] for name in VOUCHER_NAMES}
voucher_trades = {name: trades[trades['symbol'] == name] for name in VOUCHER_NAMES}

# from statsmodels.graphics.tsaplots import plot_acf
# plot_acf(gel_price['mid_price'], lags=200)
# plt.show()
#
# from numpy.fft import fft, fftfreq
# signal = gel_price['mid_price'].values
# freqs = fftfreq(len(signal))
# power = np.abs(fft(signal))**2
# dominant_freq = freqs[np.argmax(power[1:len(signal)//2]) + 1]
# cycle_length = int(1 / dominant_freq)
# print(f"Dominant cycle: {cycle_length} ticks")


# ══════════════════════════════════════════════════════════════════════════════
# EXISTING HELPER FUNCTIONS (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def wall_mid_prices(price_df):
    bid_df = price_df[['bid_volume_1', 'bid_volume_2', 'bid_volume_3',
                        'bid_price_1',  'bid_price_2',  'bid_price_3']].fillna(0)
    ask_df = price_df[['ask_volume_1', 'ask_volume_2', 'ask_volume_3',
                        'ask_price_1',  'ask_price_2',  'ask_price_3']].fillna(0)

    weighted_bids  = (bid_df['bid_volume_1'] * bid_df['bid_price_1'] +
                      bid_df['bid_volume_2'] * bid_df['bid_price_2'] +
                      bid_df['bid_volume_3'] * bid_df['bid_price_3'])
    total_bid_vol  = bid_df['bid_volume_1'] + bid_df['bid_volume_2'] + bid_df['bid_volume_3']
    bid_vwap       = weighted_bids / total_bid_vol

    weighted_asks  = (ask_df['ask_volume_1'] * ask_df['ask_price_1'] +
                      ask_df['ask_volume_2'] * ask_df['ask_price_2'] +
                      ask_df['ask_volume_3'] * ask_df['ask_price_3'])
    total_ask_vol  = ask_df['ask_volume_1'] + ask_df['ask_volume_2'] + ask_df['ask_volume_3']
    ask_vwap       = weighted_asks / total_ask_vol
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
    fig.add_trace(go.Scatter(x=price_df['global_ts'], y=price_df['total_ask_vol'], line=dict(color='blue'),  name='total_ask_vol', yaxis='y1'))
    fig.add_trace(go.Scatter(x=price_df['global_ts'], y=price_df['total_bid_vol'], line=dict(color='red'),   name='total_bid_vol', yaxis='y1'))
    fig.add_trace(go.Scatter(x=price_df['global_ts'], y=mid_df['mid_price'],       line=dict(color='green'), name='mid_price',     yaxis='y2'))
    fig.update_layout(title='Total Volume vs Mid Price',
                      yaxis=dict(title='Volume'),
                      yaxis2=dict(title='Mid Price', overlaying='y', side='right'))
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


def overlay_trade_data(figure, trade_df, y_value, name=None):
    trade_scatter = px.scatter(data_frame=trade_df, x='global_ts', y=y_value)
    for trace in trade_scatter.data:
        trace.name = name or y_value
        trace.showlegend = True
        figure.add_trace(trace)
    return figure


def plot_spike_trajectories(price_df, window=20, threshold_multiplier=2.0, lookahead=400):
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
    fig.add_trace(go.Scatter(y=avg_trajectory, mode='lines', line=dict(color='red', width=3), name='average'))
    fig.update_layout(title=f'{len(trajectories)} spikes, threshold={threshold_multiplier:.2f}σ',
                      xaxis_title='ticks after spike', yaxis_title='price change from spike')
    return fig


def plot_imbalance(price_df, trade_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=price_df['global_ts'], y=price_df['imbalance_1'], line=dict(color='blue'),   name='imbalance_1'))
    fig.add_trace(go.Scatter(x=price_df['global_ts'], y=price_df['imbalance_2'], line=dict(color='orange'), name='imbalance_2'))
    fig.add_trace(go.Scatter(x=price_df['global_ts'], y=price_df['imbalance_3'], line=dict(color='green'),  name='imbalance_3'))
    fig.add_trace(go.Scatter(x=trade_df['global_ts'], y=trade_df['quantity'], mode='markers',
                             marker=dict(color='rgba(200,0,0,0.7)', size=6), name='trade_qty'))
    fig.update_yaxes(title_text='imbalance / normalised quantity')
    return fig


def plot_prices(mid_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mid_df['global_ts'], y=mid_df['mid_price'],      line=dict(color='blue'),   name='mid_price'))
    fig.add_trace(go.Scatter(x=mid_df['global_ts'], y=mid_df['wall_mid_price'], line=dict(color='red'),    name='wall_mid_price'))
    fig.add_trace(go.Scatter(x=mid_df['global_ts'], y=mid_df['vwap_estimate'],  line=dict(color='green'),  name='vwap_estimate'))
    fig.add_trace(go.Scatter(x=mid_df['global_ts'], y=mid_df['microprice_1'],   line=dict(color='orange'), name='microprice_1'))
    fig.add_trace(go.Scatter(x=mid_df['global_ts'], y=mid_df['microprice_2'],   line=dict(color='purple'), name='microprice_2'))
    fig.add_trace(go.Scatter(x=mid_df['global_ts'], y=mid_df['microprice_3'],   line=dict(color='brown'),  name='microprice_3'))
    fig.update_layout(title='Prices')
    return fig


def plot_divergence(mid_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mid_df['global_ts'], y=mid_df['microprice_1'] - mid_df['mid_price'], line=dict(color='orange'), name='div_1', yaxis='y1'))
    fig.add_trace(go.Scatter(x=mid_df['global_ts'], y=mid_df['microprice_2'] - mid_df['mid_price'], line=dict(color='purple'), name='div_2', yaxis='y1'))
    fig.add_trace(go.Scatter(x=mid_df['global_ts'], y=mid_df['microprice_3'] - mid_df['mid_price'], line=dict(color='brown'),  name='div_3', yaxis='y1'))
    fig.add_trace(go.Scatter(x=mid_df['global_ts'], y=mid_df['mid_price'],                          line=dict(color='green'),  name='mid_price', yaxis='y2'))
    fig.add_hline(y=0, line=dict(color='black', width=0.5))
    fig.update_layout(title='Microprice Divergence from Mid',
                      yaxis=dict(title='Divergence'),
                      yaxis2=dict(title='Mid Price', overlaying='y', side='right'))
    return fig


def add_vlines(price_df, figure):
    newdays = [np.argmax(price_df['day'] == 1), np.argmax(price_df['day'] == 2)]
    ts = [price_df['global_ts'].iloc[newdays[0]], price_df['global_ts'].iloc[newdays[1]]]
    for t in ts:
        figure.add_vline(x=t, line_color='red', line_dash='dash')
    return figure


def imbalance_fv_plot(price_df, mid_df):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=mid_df['global_ts'], y=mid_df['wall_mid_price'], name='wall mid'),    secondary_y=False)
    fig.add_trace(go.Scatter(x=mid_df['global_ts'], y=mid_df['mid_price'],      name='mid price'),   secondary_y=False)
    fig.add_trace(go.Scatter(x=price_df['global_ts'], y=price_df['imbalance_3'], name='imbalance 3'), secondary_y=True)
    fig.update_yaxes(title_text="fair value", secondary_y=False)
    fig.update_yaxes(title_text="imbalance",  secondary_y=True)
    return fig


def plot_spread(price_df, spread):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=price_df['global_ts'], y=price_df[spread], line=dict(color='blue'), name='spread'))
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
    midplot          = overlay_trade_data(midplot, product_trades, 'price')
    volplot          = total_volume_plot(product_price, midprice_info)
    spreadplot       = plot_spread(product_price, 'spread_1')
    imbalanceplot    = plot_imbalance(imbalance_info, product_trades)
    spikes           = plot_spike_trajectories(midprice_info)
    imbalance_mid_pl = imbalance_fv_plot(imbalance_info, midprice_info)
    for plots in [midplot, spreadplot, imbalanceplot, imbalance_mid_pl, divplot, volplot]:
        plots = add_vlines(product_price, plots)
        plots.show()
    spikes.show()


# ══════════════════════════════════════════════════════════════════════════════
# NEW: BLACK-SCHOLES UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def bs_call_price(S, K, T, sigma, r=0.0):
    """
    Black-Scholes call price.  Returns 0 if T or sigma are non-positive.
    """
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_delta(S, K, T, sigma, r=0.0):
    """
    BS delta (= N(d1)) for a call.  At expiry returns 1 if ITM else 0.
    """
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)


def implied_vol(market_price, S, K, T, r=0.0):
    """
    Back out implied vol using Brent's method.
    Returns NaN if the price is below intrinsic value or the solve fails.
    """
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
# NEW: BUILD WIDE VOUCHER TABLE
# ══════════════════════════════════════════════════════════════════════════════

# Vectorised implied vol — wraps brentq but processes whole arrays at once
_iv_vec = np.vectorize(implied_vol, otypes=[float])
_delta_vec = np.vectorize(bs_delta, otypes=[float])


def build_voucher_wide(extract_df, voucher_prices_dict, T, cache_path='ROUND_3/voucher_wide.csv'):
    # Return cached version if it exists
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

        S = base['S'].values
        mid = base[f'mid_{name}'].values

        base[f'intrinsic_{name}'] = np.maximum(S - K, 0)
        base[f'time_value_{name}'] = mid - base[f'intrinsic_{name}'].values

        ivs = _iv_vec(mid, S, K, T)
        base[f'iv_{name}'] = ivs

        valid = ~np.isnan(ivs)
        deltas = np.full(len(base), np.nan)
        deltas[valid] = _delta_vec(S[valid], K, T, ivs[valid])
        base[f'delta_{name}'] = deltas

    base.to_csv(cache_path, index=False)
    print(f"Saved wide df to {cache_path}: {base.shape}")
    return base



# ══════════════════════════════════════════════════════════════════════════════
# NEW: VOUCHER PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_moneyness(extract_df):
    """
    Plot extract mid price with horizontal lines at every strike.
    Immediately shows which vouchers are ITM/OTM and by how much.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=extract_df['global_ts'], y=extract_df['mid_price'],
                             line=dict(color='black', width=2), name='extract mid'))
    colours = px.colors.sample_colorscale('RdYlGn', [i / (len(VOUCHER_STRIKES) - 1)
                                                      for i in range(len(VOUCHER_STRIKES))])
    for K, col in zip(VOUCHER_STRIKES, colours):
        fig.add_hline(y=K, line=dict(color=col, dash='dot', width=1),
                      annotation_text=str(K), annotation_position='right')
    fig.update_layout(title='Extract Price vs Strike Ladder (ITM/OTM)',
                      xaxis_title='global_ts', yaxis_title='price')
    return fig


def plot_iv_smile(wide_df):
    """
    Plot implied volatility smile: IV vs strike at a snapshot of timestamps.
    Samples ~10 evenly spaced timestamps so you see how the smile evolves.
    """
    fig = go.Figure()
    sample_ts = wide_df['global_ts'].iloc[::max(1, len(wide_df) // 10)].values
    for ts in sample_ts:
        row = wide_df[wide_df['global_ts'] == ts].iloc[0]
        ivs = [row[f'iv_{name}'] for name in VOUCHER_NAMES]
        fig.add_trace(go.Scatter(x=VOUCHER_STRIKES, y=ivs, mode='lines+markers',
                                 name=str(ts), opacity=0.6))
    fig.update_layout(title='Implied Volatility Smile Across Strikes',
                      xaxis_title='strike', yaxis_title='implied vol')
    return fig


def plot_time_value(wide_df):
    """
    Time value (market price minus intrinsic value) for each strike over time.
    Should peak near ATM; any negative values are hard arbitrage.
    """
    fig = go.Figure()
    colours = px.colors.sample_colorscale('Viridis', [i / (len(VOUCHER_NAMES) - 1)
                                                       for i in range(len(VOUCHER_NAMES))])
    for name, col in zip(VOUCHER_NAMES, colours):
        fig.add_trace(go.Scatter(x=wide_df['global_ts'], y=wide_df[f'time_value_{name}'],
                                 line=dict(color=col), name=name))
    fig.add_hline(y=0, line=dict(color='red', dash='dash', width=1),
                  annotation_text='hard arb boundary')
    fig.update_layout(title='Time Value per Voucher (negative = arbitrage)',
                      xaxis_title='global_ts', yaxis_title='time value')
    return fig


def plot_delta_surface(wide_df):
    """
    Delta for each voucher over time.  Deep ITM → 1, deep OTM → 0, ATM ≈ 0.5.
    Use this to size delta hedges against the extract.
    """
    fig = go.Figure()
    colours = px.colors.sample_colorscale('Plasma', [i / (len(VOUCHER_NAMES) - 1)
                                                      for i in range(len(VOUCHER_NAMES))])
    for name, col in zip(VOUCHER_NAMES, colours):
        fig.add_trace(go.Scatter(x=wide_df['global_ts'], y=wide_df[f'delta_{name}'],
                                 line=dict(color=col), name=name))
    fig.update_layout(title='BS Delta per Voucher Over Time',
                      xaxis_title='global_ts', yaxis_title='delta',
                      yaxis=dict(range=[0, 1]))
    return fig


def plot_realised_vol(extract_df, window=20):
    """
    Rolling realised volatility of extract mid returns.
    Compare to the implied vols from the smile — if IV > RV, vouchers are
    systematically overpriced (sell delta-hedged); if IV < RV, they're cheap.
    """
    df = extract_df.copy()
    df['log_ret'] = np.log(df['mid_price'] / df['mid_price'].shift(1))
    df['realised_vol'] = df['log_ret'].rolling(window).std()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['global_ts'], y=df['realised_vol'],
                             line=dict(color='navy'), name=f'realised vol (w={window})'))
    fig.update_layout(title=f'Extract Realised Volatility (rolling {window} ticks)',
                      xaxis_title='global_ts', yaxis_title='vol')
    return fig


def plot_return_distribution(extract_df):
    """
    Histogram of extract log returns.  BS assumes normality — fat tails or skew
    here tells you which strikes are systematically mis-priced by the model.
    """
    log_rets = np.log(extract_df['mid_price'] / extract_df['mid_price'].shift(1)).dropna()
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=log_rets, nbinsx=100, name='log returns',
                               marker_color='steelblue', opacity=0.8))
    # overlay normal fit
    mu, sigma = log_rets.mean(), log_rets.std()
    x_range = np.linspace(log_rets.min(), log_rets.max(), 200)
    normal_pdf = norm.pdf(x_range, mu, sigma) * len(log_rets) * (log_rets.max() - log_rets.min()) / 100
    fig.add_trace(go.Scatter(x=x_range, y=normal_pdf, line=dict(color='red', width=2), name='normal fit'))
    fig.update_layout(title='Extract Log Return Distribution vs Normal',
                      xaxis_title='log return', yaxis_title='count')
    return fig


def check_arbitrage_bounds(wide_df):
    """
    Check three model-free no-arbitrage conditions across all timesteps.
    Prints a summary and returns a dict of violation counts.

    Checks:
      1. Monotonicity:   C(K_i) >= C(K_{i+1})  for all adjacent strikes
      2. Call spread:    C(K_i) - C(K_{i+1}) <= K_{i+1} - K_i
      3. Butterfly convexity: C(K1) - 2*C(K2) + C(K3) >= 0
         for all consecutive triples (K1 < K2 < K3)
    """
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
    print(f"Arbitrage bound check over {total_rows} timestamps:")
    for key, count in violations.items():
        print(f"  {key}: {count} violations ({100*count/total_rows:.2f}%)")
    return violations


def plot_iv_vs_realised_vol(wide_df, extract_df, window=20):
    """
    Compare ATM implied vol (from the voucher closest to current S) against
    rolling realised vol.  Persistent gap = systematic edge to trade.
    """
    df = extract_df.copy().set_index('global_ts')
    df['log_ret']      = np.log(df['mid_price'] / df['mid_price'].shift(1))
    df['realised_vol'] = df['log_ret'].rolling(window).std()

    # pick ATM voucher at each timestamp as the one with smallest |S - K|
    wide = wide_df.copy()
    iv_cols = [f'iv_{name}' for name in VOUCHER_NAMES]
    def atm_iv(row):
        dists = [abs(row['S'] - K) for K in VOUCHER_STRIKES]
        atm_idx = int(np.argmin(dists))
        return row[iv_cols[atm_idx]]
    wide['atm_iv'] = wide.apply(atm_iv, axis=1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=wide['global_ts'], y=wide['atm_iv'],
                             line=dict(color='orange'), name='ATM implied vol'))
    fig.add_trace(go.Scatter(x=df.reset_index()['global_ts'], y=df['realised_vol'],
                             line=dict(color='navy'), name=f'realised vol (w={window})'))
    fig.update_layout(title='ATM Implied Vol vs Realised Vol',
                      xaxis_title='global_ts', yaxis_title='vol')
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# NEW: VOUCHER PLOTTING WRAPPER
# ══════════════════════════════════════════════════════════════════════════════

def voucher_plotting_wrapper(extract_df, voucher_prices_dict, T=1.0, rv_window=100):
    """
    Full EDA pipeline for the voucher surface.

    Args:
        extract_df:          Cleaned price df for VELVETFRUIT_EXTRACT.
        voucher_prices_dict: Dict {voucher_name: price_df} for all 10 vouchers.
        T:                   Time to expiry passed to BS (use 1.0 until we know the
                             actual round structure; only affects IV scale, not ordering).
        rv_window:           Window for realised vol computation.
    """
    # wide = build_voucher_wide(extract_df, voucher_prices_dict, T)
    wide = pd.read_csv('voucher_wide.csv')

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

# plotting_wrapper(extract_price, extract_trades, True, 50)
plotting_wrapper(gel_price,     gel_trades,     True, 1)

# voucher_plotting_wrapper(extract_price, voucher_prices, T=5.0)
