import numpy as np
import pandas as pd
import plotly.express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots
import math

# ══════════════════════════════════════════════════════════════════════════════
# LOAD & PREP  (same as template — extend prices/trades already in memory)
# ══════════════════════════════════════════════════════════════════════════════

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

prices['global_ts'] = prices['day'] * 1_000_000 + prices['timestamp']
trades['global_ts'] = trades['day'] * 1_000_000 + trades['timestamp']

EXTRACT = 'VELVETFRUIT_EXTRACT'

extract_price = prices[(prices['product'] == EXTRACT) & (prices['mid_price'] != 0.0)].copy()

VOUCHER_STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VOUCHER_NAMES   = [f'VEV_{k}' for k in VOUCHER_STRIKES]

# Pull all voucher price rows and trade rows
voucher_prices = {
    k: prices[(prices['product'] == f'VEV_{k}') & (prices['mid_price'] != 0.0)].copy()
    for k in VOUCHER_STRIKES
}
voucher_trades = {
    k: trades[trades['symbol'] == f'VEV_{k}'].copy()
    for k in VOUCHER_STRIKES
}

# Single flat df of all voucher trades — used for cross-strike views
all_voucher_trades = pd.concat(
    [df.assign(strike=k) for k, df in voucher_trades.items()],
    ignore_index=True
)

TRADER_PALETTE = px.colors.qualitative.Dark24
MARKOUT_HORIZONS = [10, 50, 100, 500]

# ── Options pricing helpers ───────────────────────────────────────────────────

TTE_DAYS_BASE = 5
TICKS_PER_DAY = 1_000_000

def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _bs_call(S, K, T, sigma):
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(S - K, 0.0)
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return S * _norm_cdf(d1) - K * _norm_cdf(d2)

def _bs_vega(S, K, T, sigma):
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrtT)
    return S * math.exp(-0.5 * d1 ** 2) / math.sqrt(2.0 * math.pi) * sqrtT

def implied_vol(price, S, K, T, sigma0=0.30):
    intrinsic = max(S - K, 0.0)
    if price <= intrinsic + 1e-4 or T <= 0:
        return np.nan
    sigma = sigma0
    for _ in range(50):
        p = _bs_call(S, K, T, sigma)
        v = _bs_vega(S, K, T, sigma)
        if v < 1e-8:
            break
        sigma -= (p - price) / v
        sigma = max(sigma, 1e-6)
    return sigma if sigma > 0 else np.nan

def compute_tte(global_ts):
    frac = (global_ts % TICKS_PER_DAY) / TICKS_PER_DAY
    return max((TTE_DAYS_BASE - frac) / 365.0, 1e-6)

# ── Generic helpers (matching template style) ─────────────────────────────────

def attach_mid(trade_df, price_df):
    p = price_df[['global_ts', 'mid_price']].sort_values('global_ts')
    return pd.merge_asof(trade_df.sort_values('global_ts'), p,
                         on='global_ts', direction='nearest')

def add_day_vlines(fig, price_df, row=None, col=None):
    for day in sorted(price_df['day'].unique()):
        ts = price_df[price_df['day'] == day]['global_ts'].iloc[0]
        kwargs = dict(x=ts, line_color='grey', line_dash='dash', opacity=0.4,
                      annotation_text=f'day {day}', annotation_position='top right')
        if row is not None:
            kwargs.update(row=row, col=col)
        fig.add_vline(**kwargs)
    return fig

def trader_color(trader_list):
    return {t: TRADER_PALETTE[i % len(TRADER_PALETTE)] for i, t in enumerate(trader_list)}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 0 — WHO EVEN TRADES VOUCHERS?
# Run this first. Read the printout before looking at any chart.
# ══════════════════════════════════════════════════════════════════════════════

def voucher_participant_summary():
    """
    Per strike: total volume, unique traders, top buyer, top seller.
    This is your map of the voucher market before you look at anything else.
    """
    rows = []
    for k in VOUCHER_STRIKES:
        df = voucher_trades[k]
        if df.empty:
            continue
        buyers      = df['buyer'].value_counts().to_dict()
        sellers     = df['seller'].value_counts().to_dict()
        all_traders = set(df['buyer'].unique()) | set(df['seller'].unique())
        rows.append({
            'strike':       k,
            'total_volume': df['quantity'].sum(),
            'n_traders':    len(all_traders),
            'top_buyer':    max(buyers,  key=buyers.get)  if buyers  else None,
            'top_seller':   max(sellers, key=sellers.get) if sellers else None,
            'traders':      sorted(all_traders),
        })
    summary = pd.DataFrame(rows)
    print("\n=== VOUCHER PARTICIPANT SUMMARY ===")
    print(summary[['strike', 'total_volume', 'n_traders',
                   'top_buyer', 'top_seller']].to_string(index=False))
    print("\nAll traders per strike:")
    for _, row in summary.iterrows():
        print(f"  VEV_{int(row['strike'])}: {row['traders']}")
    return summary


# ══════════════════════════════════════════════════════════════════════════════
# PLOT V1 — PER-STRIKE VOLUME BREAKDOWN BY TRADER (BUY vs SELL)
# Stacked bar. Shows who dominates each strike and which direction.
# ══════════════════════════════════════════════════════════════════════════════

def plot_volume_by_trader_and_strike():
    rows = []
    for k in VOUCHER_STRIKES:
        df = voucher_trades[k]
        if df.empty:
            continue
        for trader in pd.concat([df['buyer'], df['seller']]).unique():
            buy_vol  = df[df['buyer']  == trader]['quantity'].sum()
            sell_vol = df[df['seller'] == trader]['quantity'].sum()
            if buy_vol  > 0:
                rows.append({'strike': k, 'trader': trader, 'side': 'BUY',  'volume': buy_vol})
            if sell_vol > 0:
                rows.append({'strike': k, 'trader': trader, 'side': 'SELL', 'volume': sell_vol})

    vol_df      = pd.DataFrame(rows)
    all_traders = sorted(vol_df['trader'].unique())
    colors      = trader_color(all_traders)

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=['BUY volume by strike & trader',
                                        'SELL volume by strike & trader'])
    for col_idx, side in enumerate(['BUY', 'SELL'], start=1):
        sub = vol_df[vol_df['side'] == side]
        for trader in all_traders:
            t_sub = sub[sub['trader'] == trader]
            if t_sub.empty:
                continue
            fig.add_trace(go.Bar(
                x=t_sub['strike'].astype(str), y=t_sub['volume'],
                name=trader, marker_color=colors[trader],
                legendgroup=trader, showlegend=(col_idx == 1),
            ), row=1, col=col_idx)

    fig.update_layout(barmode='stack',
                      title='Voucher Volume by Trader and Strike',
                      xaxis_title='strike', yaxis_title='volume',
                      legend=dict(x=1.01, y=1))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# PLOT V2 — DIRECTION BIAS HEATMAP
# buy_ratio = buy_vol / (buy_vol + sell_vol) per trader per strike.
# 1.0 = pure buyer,  0.0 = pure seller,  0.5 = market maker.
# Fastest way to spot informed directional traders on vouchers.
# ══════════════════════════════════════════════════════════════════════════════

def plot_direction_bias_heatmap():
    rows = []
    for k in VOUCHER_STRIKES:
        df = voucher_trades[k]
        if df.empty:
            continue
        for trader in pd.concat([df['buyer'], df['seller']]).unique():
            buy_vol  = df[df['buyer']  == trader]['quantity'].sum()
            sell_vol = df[df['seller'] == trader]['quantity'].sum()
            total    = buy_vol + sell_vol
            if total == 0:
                continue
            rows.append({'strike': k, 'trader': trader,
                         'buy_ratio': buy_vol / total, 'total_volume': total})

    bias_df = pd.DataFrame(rows)
    pivot   = bias_df.pivot(index='trader', columns='strike', values='buy_ratio')

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[str(c) for c in pivot.columns],
        y=pivot.index.tolist(),
        colorscale='RdYlGn', zmid=0.5, zmin=0.0, zmax=1.0,
        text=np.round(pivot.values, 2), texttemplate='%{text}',
        colorbar=dict(title='buy ratio<br>1=pure buyer<br>0=pure seller'),
        hovertemplate='trader: %{y}<br>strike: %{x}<br>buy ratio: %{z:.2f}<extra></extra>'
    ))
    fig.update_layout(title='Trader Direction Bias by Strike (buy ratio)',
                      xaxis_title='strike', yaxis_title='trader')
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# PLOT V3 — VOUCHER TRADES OVERLAID ON EXTRACT MID PRICE
# When does each trader buy/sell each voucher relative to EXTRACT moves?
# Informed call buyers should enter before EXTRACT rises.
# ══════════════════════════════════════════════════════════════════════════════

def plot_voucher_trades_on_extract(strikes_to_plot=None, traders_to_plot=None):
    if strikes_to_plot is None:
        strikes_to_plot = VOUCHER_STRIKES
    if traders_to_plot is None:
        top = (all_voucher_trades[all_voucher_trades['strike'].isin(strikes_to_plot)]
               .melt(id_vars=[], value_vars=['buyer', 'seller'])
               ['value'].value_counts().head(8).index.tolist())
        traders_to_plot = [t for t in top if t != 'SUBMISSION']

    colors = trader_color(traders_to_plot)
    n      = len(strikes_to_plot)
    fig    = make_subplots(rows=n, cols=1, shared_xaxes=True,
                           subplot_titles=[f'VEV_{k}' for k in strikes_to_plot],
                           vertical_spacing=0.04)

    for row_idx, k in enumerate(strikes_to_plot, start=1):
        df = voucher_trades[k]
        if df.empty:
            continue
        fig.add_trace(go.Scatter(
            x=extract_price['global_ts'], y=extract_price['mid_price'],
            line=dict(color='rgba(150,150,150,0.4)', width=1),
            name='EXTRACT mid', showlegend=(row_idx == 1), legendgroup='extract_mid'
        ), row=row_idx, col=1)

        t_with_mid = attach_mid(df, extract_price)
        for trader in traders_to_plot:
            col   = colors[trader]
            buys  = t_with_mid[t_with_mid['buyer']  == trader]
            sells = t_with_mid[t_with_mid['seller'] == trader]
            if not buys.empty:
                fig.add_trace(go.Scatter(
                    x=buys['global_ts'], y=buys['mid_price'], mode='markers',
                    marker=dict(symbol='triangle-up', size=8, color=col, opacity=0.8),
                    name=f'{trader} BUY', legendgroup=trader,
                    showlegend=(row_idx == 1)
                ), row=row_idx, col=1)
            if not sells.empty:
                fig.add_trace(go.Scatter(
                    x=sells['global_ts'], y=sells['mid_price'], mode='markers',
                    marker=dict(symbol='triangle-down', size=8, color=col, opacity=0.8,
                                line=dict(color='black', width=0.5)),
                    name=f'{trader} SELL', legendgroup=trader,
                    showlegend=(row_idx == 1)
                ), row=row_idx, col=1)

    fig.update_layout(title='Voucher Counterparty Trades on EXTRACT Mid',
                      height=300 * n, legend=dict(x=1.01, y=1))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# PLOT V4 — IMPLIED VOL EACH TRADER TRANSACTS AT
# Did they buy cheap IV or pay up? MM = tight stable IV. Informed = spikes.
# ══════════════════════════════════════════════════════════════════════════════

def compute_trade_ivs():
    """Attach implied vol to every voucher trade. Slow — run once."""
    ext_mid = extract_price[['global_ts', 'mid_price']].sort_values('global_ts')
    rows = []
    for k in VOUCHER_STRIKES:
        df = voucher_trades[k].copy()
        if df.empty:
            continue
        df = pd.merge_asof(df.sort_values('global_ts'), ext_mid,
                           on='global_ts', direction='nearest')
        df = df.rename(columns={'mid_price': 'S'})
        df['iv']        = df.apply(
            lambda r: implied_vol(r['price'], r['S'], k, compute_tte(r['global_ts'])), axis=1)
        df['strike']    = k
        df['moneyness'] = df['S'] - k
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def plot_iv_by_trader(trade_iv_df, min_trades=10):
    """Box plot of IV transacted per trader across all strikes."""
    if trade_iv_df.empty:
        return None
    counts  = (trade_iv_df.melt(id_vars=[], value_vars=['buyer', 'seller'])
               ['value'].value_counts())
    valid   = counts[counts >= min_trades].index.tolist()
    colors  = trader_color(valid)
    rows    = []
    for trader in valid:
        sub = trade_iv_df[(trade_iv_df['buyer'] == trader) |
                          (trade_iv_df['seller'] == trader)].copy()
        sub['trader'] = trader
        rows.append(sub[['trader', 'iv', 'strike', 'moneyness']])
    if not rows:
        return None
    df = pd.concat(rows).dropna(subset=['iv'])
    fig = go.Figure()
    for trader in valid:
        fig.add_trace(go.Box(
            y=df[df['trader'] == trader]['iv'],
            name=trader, boxmean=True, marker_color=colors[trader]
        ))
    fig.update_layout(title='IV Distribution by Trader (all strikes)',
                      yaxis_title='implied vol', xaxis_title='trader',
                      showlegend=False)
    return fig


def plot_iv_heatmap_by_trader_and_strike(trade_iv_df, min_trades=5):
    """Mean IV transacted — trader × strike. Reveals strike concentration."""
    if trade_iv_df.empty:
        return None
    rows = []
    for k in VOUCHER_STRIKES:
        sub = trade_iv_df[trade_iv_df['strike'] == k].dropna(subset=['iv'])
        if sub.empty:
            continue
        for trader in pd.concat([sub['buyer'], sub['seller']]).unique():
            t_sub = sub[(sub['buyer'] == trader) | (sub['seller'] == trader)]
            if len(t_sub) < min_trades:
                continue
            rows.append({'strike': k, 'trader': trader, 'mean_iv': t_sub['iv'].mean()})
    if not rows:
        return None
    pivot = pd.DataFrame(rows).pivot(index='trader', columns='strike', values='mean_iv')
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=[str(c) for c in pivot.columns], y=pivot.index.tolist(),
        colorscale='Viridis', text=np.round(pivot.values, 3), texttemplate='%{text}',
        colorbar=dict(title='mean IV'),
        hovertemplate='trader: %{y}<br>strike: %{x}<br>mean IV: %{z:.3f}<extra></extra>'
    ))
    fig.update_layout(title='Mean IV Transacted — Trader × Strike',
                      xaxis_title='strike', yaxis_title='trader')
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# PLOT V5 — MARKOUT ON EXTRACT AFTER VOUCHER TRADE
# Buying a call → expect EXTRACT to rise. Does it?
# Upward sloping markout = informed voucher buyer.
# This is how you find the Mark 67 equivalent on the options side.
# ══════════════════════════════════════════════════════════════════════════════

def compute_voucher_markout(strikes=None):
    """For each voucher trade, compute EXTRACT mid move after the trade."""
    if strikes is None:
        strikes = VOUCHER_STRIKES
    ext = extract_price[['global_ts', 'day', 'mid_price']].sort_values('global_ts')
    rows = []
    for k in strikes:
        df = voucher_trades[k].copy()
        if df.empty:
            continue
        df = pd.merge_asof(df.sort_values('global_ts'),
                           ext[['global_ts', 'mid_price']].rename(
                               columns={'mid_price': 'extract_mid_at_trade'}),
                           on='global_ts', direction='nearest')
        df['strike'] = k
        for h in MARKOUT_HORIZONS:
            future = ext.copy()
            future['global_ts'] = future['global_ts'] - h
            future = future.rename(columns={'mid_price': f'extract_mid_t{h}',
                                            'day':       f'day_t{h}'})
            df = pd.merge_asof(df, future[['global_ts', f'extract_mid_t{h}', f'day_t{h}']],
                               on='global_ts', direction='forward')
            if f'day_t{h}' in df.columns:
                df[f'extract_mid_t{h}'] = df[f'extract_mid_t{h}'].where(
                    df[f'day_t{h}'] == df['day'], np.nan)
                df = df.drop(columns=[f'day_t{h}'])
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def plot_voucher_markout_by_trader(markout_df, min_trades=10):
    """Per-trader markout curve across all strikes."""
    if markout_df.empty:
        return None
    counts = (markout_df.melt(id_vars=[], value_vars=['buyer', 'seller'])
              ['value'].value_counts())
    valid  = counts[counts >= min_trades].index.tolist()
    colors = trader_color(valid)

    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color='black', dash='dash', width=1),
                  annotation_text='no edge')
    for trader in valid:
        mask = (markout_df['buyer'] == trader) | (markout_df['seller'] == trader)
        sub  = markout_df[mask].copy()
        sub['side_sign'] = np.where(sub['buyer'] == trader, 1, -1)
        avgs = []
        for h in MARKOUT_HORIZONS:
            col_name = f'extract_mid_t{h}'
            if col_name not in sub.columns:
                avgs.append(np.nan); continue
            mo = (sub[col_name] - sub['extract_mid_at_trade']) * sub['side_sign']
            avgs.append(mo.mean())
        fig.add_trace(go.Scatter(
            x=MARKOUT_HORIZONS, y=avgs, mode='lines+markers',
            line=dict(color=colors[trader], width=2), marker=dict(size=7),
            name=f'{trader} (n={mask.sum()})'
        ))
    fig.update_layout(
        title='Voucher Trader Markout — EXTRACT mid change after voucher trade',
        xaxis_title='horizon (ticks)', yaxis_title='avg EXTRACT move in trade direction',
        legend=dict(x=1.01, y=1))
    return fig


def plot_voucher_markout_by_strike(markout_df, trader):
    """For one trader: does their edge concentrate at specific strikes?"""
    if markout_df.empty:
        return None
    mask = (markout_df['buyer'] == trader) | (markout_df['seller'] == trader)
    sub  = markout_df[mask].copy()
    sub['side_sign'] = np.where(sub['buyer'] == trader, 1, -1)
    active = [k for k in VOUCHER_STRIKES if k in sub['strike'].values]
    if not active:
        print(f"No voucher trades for {trader}"); return None

    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color='black', dash='dash', width=1))
    for k in active:
        s_sub = sub[sub['strike'] == k]
        avgs  = []
        for h in MARKOUT_HORIZONS:
            col_name = f'extract_mid_t{h}'
            if col_name not in s_sub.columns:
                avgs.append(np.nan); continue
            mo = (s_sub[col_name] - s_sub['extract_mid_at_trade']) * s_sub['side_sign']
            avgs.append(mo.mean())
        fig.add_trace(go.Scatter(
            x=MARKOUT_HORIZONS, y=avgs, mode='lines+markers',
            line=dict(width=2), marker=dict(size=7),
            name=f'VEV_{k} (n={len(s_sub)})'
        ))
    fig.update_layout(
        title=f'{trader} — EXTRACT Markout by Strike',
        xaxis_title='horizon (ticks)', yaxis_title='avg EXTRACT move in direction',
        legend=dict(x=1.01, y=1))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# PLOT V6 — CROSS-STRIKE SIMULTANEOUS TRADING
# Traders buying/selling multiple strikes at the same timestamp are running
# spreads (risk reversals, butterflies). This is sophisticated behaviour.
# ══════════════════════════════════════════════════════════════════════════════

def find_cross_strike_traders(window_ticks=5):
    print(f"\n=== CROSS-STRIKE CO-TRADING (window={window_ticks} ticks) ===")
    for trader in sorted(all_voucher_trades
                         .melt(id_vars=[], value_vars=['buyer', 'seller'])
                         ['value'].unique()):
        mask = ((all_voucher_trades['buyer'] == trader) |
                (all_voucher_trades['seller'] == trader))
        sub  = all_voucher_trades[mask].sort_values('global_ts')
        if len(sub) < 2:
            continue
        cross = 0
        for i in range(len(sub) - 1):
            if (abs(sub.iloc[i+1]['global_ts'] - sub.iloc[i]['global_ts']) <= window_ticks
                    and sub.iloc[i+1]['strike'] != sub.iloc[i]['strike']):
                cross += 1
        if cross > 0:
            print(f"  {trader}: {cross} cross-strike events / {len(sub)} trades "
                  f"({100*cross/len(sub):.1f}%)")


def plot_cross_strike_heatmap(window_ticks=5):
    """How often are pairs of strikes traded near-simultaneously by the same trader?"""
    from itertools import combinations
    pair_counts = {(k1, k2): 0 for k1, k2 in combinations(VOUCHER_STRIKES, 2)}

    for trader in sorted(all_voucher_trades
                         .melt(id_vars=[], value_vars=['buyer', 'seller'])
                         ['value'].unique()):
        mask = ((all_voucher_trades['buyer'] == trader) |
                (all_voucher_trades['seller'] == trader))
        sub  = all_voucher_trades[mask].sort_values('global_ts')
        for i in range(len(sub) - 1):
            if abs(sub.iloc[i+1]['global_ts'] - sub.iloc[i]['global_ts']) <= window_ticks:
                k1, k2 = sorted([sub.iloc[i]['strike'], sub.iloc[i+1]['strike']])
                if k1 != k2:
                    pair_counts[(k1, k2)] = pair_counts.get((k1, k2), 0) + 1

    labels = [str(k) for k in VOUCHER_STRIKES]
    z = np.zeros((len(VOUCHER_STRIKES), len(VOUCHER_STRIKES)))
    for (k1, k2), count in pair_counts.items():
        i = VOUCHER_STRIKES.index(k1); j = VOUCHER_STRIKES.index(k2)
        z[i, j] = count; z[j, i] = count

    fig = go.Figure(go.Heatmap(
        z=z, x=labels, y=labels, colorscale='Blues',
        text=z.astype(int), texttemplate='%{text}',
        hovertemplate='strike %{y} ↔ strike %{x}<br>co-trades: %{z}<extra></extra>'
    ))
    fig.update_layout(
        title=f'Cross-Strike Co-Trading Frequency (within {window_ticks} ticks)',
        xaxis_title='strike', yaxis_title='strike')
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# PLOT V7 — CUMULATIVE SIGNED INVENTORY PER TRADER (per strike)
# Who is accumulating net long/short options exposure and when do they flip?
# ══════════════════════════════════════════════════════════════════════════════

def plot_voucher_inventory_over_time(strikes_to_show=None):
    if strikes_to_show is None:
        strikes_to_show = [5200, 5300, 5400, 5500]

    all_traders = (all_voucher_trades[all_voucher_trades['strike'].isin(strikes_to_show)]
                   .melt(id_vars=[], value_vars=['buyer', 'seller'])
                   ['value'].value_counts().head(8).index.tolist())
    colors = trader_color(all_traders)
    n      = len(strikes_to_show)
    fig    = make_subplots(rows=n, cols=1, shared_xaxes=True,
                           subplot_titles=[f'VEV_{k}' for k in strikes_to_show],
                           vertical_spacing=0.05)

    for row_idx, k in enumerate(strikes_to_show, start=1):
        df = voucher_trades[k]
        if df.empty:
            continue
        for trader in all_traders:
            buys  = df[df['buyer']  == trader][['global_ts', 'quantity']].copy()
            sells = df[df['seller'] == trader][['global_ts', 'quantity']].copy()
            buys['signed']  =  buys['quantity']
            sells['signed'] = -sells['quantity']
            combined = pd.concat([buys[['global_ts', 'signed']],
                                   sells[['global_ts', 'signed']]]).sort_values('global_ts')
            if combined.empty:
                continue
            combined['cum'] = combined['signed'].cumsum()
            fig.add_trace(go.Scatter(
                x=combined['global_ts'], y=combined['cum'],
                line=dict(color=colors[trader], width=1.5),
                name=trader, legendgroup=trader, showlegend=(row_idx == 1)
            ), row=row_idx, col=1)

    fig.update_layout(title='Cumulative Voucher Inventory by Trader',
                      height=250 * n, legend=dict(x=1.01, y=1))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# MASTER RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_voucher_analysis():
    print("\n" + "="*60)
    print("VOUCHER COUNTERPARTY BEHAVIOUR ANALYSIS")
    print("="*60)

    # Step 0: read these printouts before touching any chart
    summary = voucher_participant_summary()
    find_cross_strike_traders(window_ticks=5)

    # V1: who trades what volume where
    plot_volume_by_trader_and_strike().show()

    # V2: direction bias — find the pure call buyers / pure sellers
    plot_direction_bias_heatmap().show()

    # V3: when they trade relative to EXTRACT spot
    plot_voucher_trades_on_extract().show()

    # V4: implied vol they transact at (slow — ~30s)
    print("\nComputing implied vols...")
    trade_iv_df = compute_trade_ivs()
    if not trade_iv_df.empty:
        fig_v4a = plot_iv_by_trader(trade_iv_df)
        if fig_v4a: fig_v4a.show()
        fig_v4b = plot_iv_heatmap_by_trader_and_strike(trade_iv_df)
        if fig_v4b: fig_v4b.show()

    # V5: EXTRACT markout after voucher trade — the key informed-trader test
    print("\nComputing voucher markouts...")
    markout_df = compute_voucher_markout()
    if not markout_df.empty:
        fig_v5 = plot_voucher_markout_by_trader(markout_df)
        if fig_v5: fig_v5.show()

        # After seeing V5, update this to whichever trader has the highest markout
        top_informed = 'Mark 67'
        fig_v5b = plot_voucher_markout_by_strike(markout_df, trader=top_informed)
        if fig_v5b: fig_v5b.show()

    # V6: cross-strike spread trading
    plot_cross_strike_heatmap(window_ticks=5).show()

    # V7: cumulative inventory — who accumulates and when they flip
    plot_voucher_inventory_over_time().show()


run_voucher_analysis()
