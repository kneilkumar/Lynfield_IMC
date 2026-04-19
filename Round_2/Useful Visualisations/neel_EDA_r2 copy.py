import numpy as np
import pandas as pd
import plotly.express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots


# ── Load & merge ──────────────────────────────────────────────────────────────
prices_1 = pd.read_csv('ROUND_2/prices_round_2_day_-1.csv', delimiter=';')
prices_2 = pd.read_csv('ROUND_2/prices_round_2_day_0.csv', delimiter=';')
prices_3 = pd.read_csv('ROUND_2/prices_round_2_day_1.csv', delimiter=';')
trades_1 = pd.read_csv('ROUND_2/trades_round_2_day_-1.csv', delimiter=';')
trades_2 = pd.read_csv('ROUND_2/trades_round_2_day_0.csv', delimiter=';')
trades_3 = pd.read_csv('ROUND_2/trades_round_2_day_1.csv', delimiter=';')

trades_1 = pd.merge(trades_1, prices_1['day'], left_index=True, right_index=True)
trades_2 = pd.merge(trades_2, prices_2['day'], left_index=True, right_index=True)
trades_3 = pd.merge(trades_3, prices_3['day'], left_index=True, right_index=True)


prices = pd.concat([prices_1,prices_2, prices_3]).reset_index(drop=True)
trades = pd.concat([trades_1, trades_2, trades_3]).reset_index(drop=True)

print(list(prices.columns.values))
print(list(trades.columns.values))

prices['global_ts'] = (prices['day'] + 2) * 1_000_000 + prices['timestamp']
trades['global_ts'] = (trades['day'] + 2) * 1_000_000 + trades['timestamp']
trades['qty_normalised'] = trades['quantity'] / trades['quantity'].abs().max()

prices['spread_1'] = prices['ask_price_1'] - prices['bid_price_1']
prices['spread_2'] = prices['ask_price_2'] - prices['bid_price_2']
prices['spread_3'] = prices['ask_price_3'] - prices['bid_price_3']

root_price = prices[prices['product'] == 'INTARIAN_PEPPER_ROOT']
ash_price = prices[prices['product'] == 'ASH_COATED_OSMIUM']

root_trades = trades[trades['symbol'] == 'INTARIAN_PEPPER_ROOT']
ash_trades = trades[trades['symbol'] == 'ASH_COATED_OSMIUM']

# EDA - Helper functions


def wall_mid_prices(price_df):
    bid_df = price_df[['bid_volume_1', 'bid_volume_2', 'bid_volume_3', 'bid_price_1', 'bid_price_2', 'bid_price_3']].fillna(0)
    ask_df = price_df[['ask_volume_1', 'ask_volume_2', 'ask_volume_3', 'ask_price_1', 'ask_price_2', 'ask_price_3']].fillna(0)
    weighted_bids = (bid_df['bid_volume_1'] * bid_df['bid_price_1']) + (bid_df['bid_volume_2'] * bid_df['bid_price_2']) + (bid_df['bid_volume_3'] * bid_df['bid_price_3'])
    total_bid_vol = bid_df['bid_volume_1'] + bid_df['bid_volume_3'] + bid_df['bid_volume_2']
    bid_vwap = weighted_bids/total_bid_vol
    weighted_asks = (ask_df['ask_volume_1'] * ask_df['ask_price_1']) + (ask_df['ask_volume_2'] * ask_df['ask_price_2']) + (ask_df['ask_volume_3'] * ask_df['ask_price_3'])
    total_bid_asks = ask_df['ask_volume_1'] + ask_df['ask_volume_3'] + ask_df['ask_volume_2']
    ask_vwap = weighted_asks/total_bid_asks
    fv_vwap_est = (bid_vwap + ask_vwap)/2
    price_df['vwap_estimate'] = fv_vwap_est

    order_book_bids = price_df[['bid_price_1', 'bid_volume_1', 'bid_price_2', 'bid_volume_2',
                                'bid_price_3', 'bid_volume_3']].dropna(how='all')
    order_book_asks = price_df[['ask_price_1', 'ask_volume_1',
                                'ask_price_2', 'ask_volume_2', 'ask_price_3', 'ask_volume_3']].dropna(how='all')
    order_book_bids['wall_bid_price'] = order_book_bids[['bid_volume_1', 'bid_volume_2', 'bid_volume_3']].idxmax(axis=1, skipna=True)
    order_book_bids['wall_bid_price'] = np.where(order_book_bids['wall_bid_price'] == 'bid_volume_1', order_book_bids['bid_price_1'],
                                                 np.where(order_book_bids['wall_bid_price'] == 'bid_volume_2', order_book_bids['bid_price_2'], order_book_bids['bid_price_3']))

    order_book_asks['wall_ask_price'] = order_book_asks[['ask_volume_1', 'ask_volume_2', 'ask_volume_3']].idxmax(axis=1, skipna=True)
    order_book_asks['wall_ask_price'] = np.where(order_book_asks['wall_ask_price'] == 'ask_volume_1', order_book_asks['ask_price_1'],
                                                 np.where(order_book_asks['wall_ask_price'] == 'ask_volume_2', order_book_asks['ask_price_2'], order_book_asks['ask_price_3']))
    wall_mids = pd.merge(order_book_asks['wall_ask_price'], order_book_bids['wall_bid_price'], left_index=True, right_index=True)
    wall_mids['wall_mid_price'] = (wall_mids['wall_ask_price'] + wall_mids['wall_bid_price'])/2

    midinfo = price_df.join(wall_mids[['wall_mid_price']], how='left')

    return midinfo


def imbalances(price_df):
    """
    Compute order book imbalance at each level and in aggregate.

    Imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume).
    Ranges from -1 (fully ask-side) to +1 (fully bid-side).
    Positive imbalance suggests buying pressure, negative suggests selling pressure.

    Args:
        price_df: Price dataframe with bid/ask volume columns for levels 1-3.

    Returns:
        Input dataframe with imbalance_1, imbalance_2, imbalance_3,
        and total_imbalance columns added in place.
    """
    price_df['imbalance_1'] = (price_df['bid_volume_1'] - price_df['ask_volume_1'])/(price_df['bid_volume_1'] + price_df['ask_volume_1'])
    price_df['imbalance_2'] = (price_df['bid_volume_2'] - price_df['ask_volume_2'])/(price_df['bid_volume_2'] + price_df['ask_volume_2'])
    price_df['imbalance_3'] = (price_df['bid_volume_3'] - price_df['ask_volume_3'])/(price_df['bid_volume_3'] + price_df['ask_volume_3'])
    total_bid_vol = price_df['bid_volume_1'] + price_df['bid_volume_3'] + price_df['bid_volume_2']
    total_ask_vol = price_df['ask_volume_1'] + price_df['ask_volume_3'] + price_df['ask_volume_2']
    price_df['total_imbalance'] = (total_bid_vol - total_ask_vol)/(total_bid_vol + total_ask_vol)
    return price_df


def resampler(df, resample_freq):
    """
    Downsample a dataframe by averaging every resample_freq rows.

    Used to reduce noise and rendering overhead before plotting.
    Drops the product column as it is non-numeric.

    Args:
        df: Price dataframe.
        resample_freq: Number of rows to average into one.

    Returns:
        Resampled dataframe with numeric columns averaged.
    """
    df = df.drop('product', axis=1)
    df = df.groupby(np.arange(len(df)) // resample_freq).mean()
    return df


def overlay_trade_data(figure, trade_df, y_value, name=None):
    """
    Overlay executed trade data as scatter points on an existing figure.

    Args:
        figure: Existing plotly Figure to add traces to.
        trade_df: Trades dataframe with a global_ts column.
        y_value: Column name in trade_df to plot on the y-axis (e.g. 'price', 'quantity').

    Returns:
        Figure with trade scatter points added.
    """
    trade_scatter = px.scatter(data_frame=trade_df, x='global_ts', y=y_value)
    for trace in trade_scatter.data:
        trace.name = name or y_value
        trace.showlegend = True
        figure.add_trace(trace)
    return figure


def plot_spike_trajectories(price_df, window=20, threshold_multiplier=2.0, lookahead=200):
    """
    Plot normalised price trajectories following wall mid price spikes.

    A spike is defined as a deviation from the rolling mean exceeding
    threshold_multiplier * rolling_std. If no spikes are found, the threshold
    is auto-lowered in 0.25 steps until spikes are detected or 0.5 is reached.
    Consecutive spikes within lookahead ticks of each other are deduplicated
    to avoid overlapping trajectories inflating the average.

    Each trajectory is anchored to 0 at the spike timestamp so trajectories
    are comparable across different price levels. The red average line shows
    the typical post-spike behaviour: upward slope = momentum,
    return to 0 = mean reversion.

    Args:
        price_df: DataFrame with wall_mid_price column, output of wall_mid_prices().
        window: Rolling window size for mean and std computation.
        threshold_multiplier: Initial number of standard deviations to define a spike.
        lookahead: Number of ticks to plot after each spike.

    Returns:
        Plotly Figure with individual trajectories in grey and average in red,
        or None if no spikes found after threshold lowering.
    """
    price_df = price_df.reset_index(drop=True)
    rolling_mean = price_df['wall_mid_price'].rolling(window).mean()
    rolling_std = price_df['wall_mid_price'].rolling(window).std()

    spike_mask = (price_df['wall_mid_price'] - rolling_mean).abs() > threshold_multiplier * rolling_std

    # auto-lower threshold until we find something
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
        normalised = window_prices - window_prices[0]
        trajectories.append(normalised)

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
    """
    Plot per-level order book imbalance with normalised trade quantity overlaid.

    Trade quantity is normalised to [-1, 1] to share the same axis as imbalance values.

    Args:
        price_df: Price dataframe with imbalance_1/2/3 columns, output of imbalances().
        trade_df: Trades dataframe with global_ts and quantity columns.

    Returns:
        Plotly Figure with imbalance lines and trade quantity bars.
    """
    trade_df = trade_df.copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=price_df['global_ts'], y=price_df['imbalance_1'], line=dict(color='blue'), name='imbalance_1'))
    fig.add_trace(go.Scatter(x=price_df['global_ts'], y=price_df['imbalance_2'], line=dict(color='orange'), name='imbalance_2'))
    fig.add_trace(go.Scatter(x=price_df['global_ts'], y=price_df['imbalance_3'], line=dict(color='green'), name='imbalance_3'))
    fig.add_trace(go.Scatter(x=trade_df['global_ts'], y=np.zeros(len(trade_df)), mode='markers',
                             marker=dict(color='rgba(200,0,0,0.7)', size=6), name='trade_qty'))
    fig.update_yaxes(title_text='imbalance / normalised quantity')
    return fig


def plot_mids(mid_df):
    """
    Plot mid price and wall mid price on the same axis.

    Args:
        mid_df: DataFrame with global_ts, mid_price, and wall_mid_price columns,
                output of wall_mid_prices().

    Returns:
        Plotly Figure with mid price in blue and wall mid price in red.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mid_df['global_ts'], y=mid_df['mid_price'], line=dict(color='blue'), name='mid_price'))
    fig.add_trace(go.Scatter(x=mid_df['global_ts'], y=mid_df['wall_mid_price'], line=dict(color='red'), name='wall_mid_price'))
    fig.add_trace(go.Scatter(x=mid_df['global_ts'], y=mid_df['vwap_estimate'], line=dict(color='green'), name='vwap_estimate'))
    return fig


def add_vlines(price_df,figure):
    """
    Add vertical dashed lines at day boundaries (day 0 and day 1 transitions).

    Args:
        price_df: Price dataframe with day and global_ts columns.
        figure: Existing plotly Figure to add lines to.

    Returns:
        Figure with day boundary vlines added.
    """
    newdays = [np.argmax(price_df['day'] == 0), np.argmax(price_df['day'] == 1)]
    ts = [price_df['global_ts'].iloc[newdays[0]], price_df['global_ts'].iloc[newdays[1]]]
    for t in ts:
        figure.add_vline(x=t, line_color='red', line_dash='dash')
    return figure


def plot_spread(price_df, spread):
    """
    Plot a single spread column over time.

    Args:
        price_df: Price dataframe with global_ts and the specified spread column.
        spread: Column name to plot (e.g. 'spread_1').

    Returns:
        Plotly Figure with spread over time.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=price_df['global_ts'], y=price_df[spread], line=dict(color='blue'), name='spread'))
    return fig


def plotting_wrapper(product_price, product_trades, resample, resample_freq):
    """
    Run the full EDA pipeline for a single product and display all plots.

    Optionally resamples price data, then produces and shows:
        1. Mid price + wall mid price with executed trade prices overlaid
        2. Spread over time
        3. Order book imbalance with normalised trade quantity overlaid
        4. Spike trajectory analysis

    Day boundary vlines are added to plots 1-3.

    Args:
        product_price: Filtered price dataframe for a single product.
        product_trades: Filtered trades dataframe for the same product.
        resample: Whether to downsample the price data before plotting.
        resample_freq: Rows to average per sample if resample is True.
    """
    if resample:
        product_price = resampler(product_price, resample_freq)
    midprice_info = wall_mid_prices(product_price)
    imbalance_info = imbalances(product_price)
    midplot = plot_mids(midprice_info)
    midplot = overlay_trade_data(midplot, product_trades, 'price', name='trade_prices')
    spreadplot = plot_spread(product_price, 'spread_1')
    imbalanceplot = plot_imbalance(imbalance_info, product_trades)
    spikes = plot_spike_trajectories(midprice_info)
    for plots in [midplot, spreadplot, imbalanceplot]:
        plots = add_vlines(product_price, plots)
        plots.show()
    spikes.show()


plotting_wrapper(ash_price, ash_trades, True, 50)

