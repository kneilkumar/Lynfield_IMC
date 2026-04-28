import numpy as np
import pandas as pd
import plotly.express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots

# ══════════════════════════════════════════════════════════════════════════════
# LOAD & PREP
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
GEL     = 'HYDROGEL_PACK'

extract_price  = prices[(prices['product'] == EXTRACT) & (prices['mid_price'] != 0.0)].copy()
gel_price      = prices[(prices['product'] == GEL)     & (prices['mid_price'] != 0.0)].copy()
extract_trades = trades[trades['symbol'] == EXTRACT].copy()
gel_trades     = trades[trades['symbol'] == GEL].copy()

# Traders of interest — update once you've run classification
KEY_TRADERS = ['Mark 67', 'Mark 14', 'Mark 22', 'Mark 55', 'Mark 49']
HORIZONS    = [10, 20, 50, 150]


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def attach_mid(trade_df, price_df):
    """Attach prevailing mid price to each trade row."""
    p = price_df[['global_ts', 'mid_price']].sort_values('global_ts')
    return pd.merge_asof(trade_df.sort_values('global_ts'), p,
                         on='global_ts', direction='nearest')


def add_day_vlines(fig, price_df):
    for day in sorted(price_df['day'].unique()):
        ts = price_df[price_df['day'] == day]['global_ts'].iloc[0]
        fig.add_vline(x=ts, line_color='grey', line_dash='dash', opacity=0.4,
                      annotation_text=f'day {day}', annotation_position='top right')
    return fig


TRADER_PALETTE = px.colors.qualitative.Dark24


def trader_color(trader_list):
    return {t: TRADER_PALETTE[i % len(TRADER_PALETTE)] for i, t in enumerate(trader_list)}


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1: INDIVIDUAL TRADER TRADES OVERLAID ON MID PRICE
# Shows exactly when each key trader buys/sells relative to price moves.
# This is your primary tool for spotting Mark 67's edge visually.
# ══════════════════════════════════════════════════════════════════════════════

def plot_trader_trades_on_mid(price_df, trade_df, traders, title_suffix=''):
    """
    Mid price line + scatter of each trader's trades (buy=triangle-up, sell=triangle-down).
    Run once per product. Immediately shows if a trader is buying dips / selling peaks.
    """
    colors = trader_color(traders)
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=price_df['global_ts'], y=price_df['mid_price'],
        line=dict(color='rgba(100,100,100,0.5)', width=1),
        name='mid price', showlegend=True
    ))

    t_with_mid = attach_mid(trade_df, price_df)

    for trader in traders:
        col = colors[trader]
        buys  = t_with_mid[t_with_mid['buyer']  == trader]
        sells = t_with_mid[t_with_mid['seller'] == trader]

        if not buys.empty:
            fig.add_trace(go.Scatter(
                x=buys['global_ts'], y=buys['price'],
                mode='markers',
                marker=dict(symbol='triangle-up', size=9, color=col, opacity=0.85),
                name=f'{trader} BUY', legendgroup=trader
            ))
        if not sells.empty:
            fig.add_trace(go.Scatter(
                x=sells['global_ts'], y=sells['price'],
                mode='markers',
                marker=dict(symbol='triangle-down', size=9, color=col, opacity=0.85,
                            line=dict(color='black', width=0.5)),
                name=f'{trader} SELL', legendgroup=trader
            ))

    fig.update_layout(
        title=f'Individual Trader Entries vs Mid Price {title_suffix}',
        xaxis_title='global timestamp', yaxis_title='price',
        legend=dict(x=1.01, y=1)
    )
    return add_day_vlines(fig, price_df)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2: CUMULATIVE SIGNED INVENTORY PER TRADER
# Running net position proxy (buys - sells in quantity).
# A trader with a consistently rising line is accumulating long.
# Reversals tell you when they flip direction — cross with mid to see why.
# ══════════════════════════════════════════════════════════════════════════════

def plot_cumulative_inventory(trade_df, price_df, traders, title_suffix=''):
    colors = trader_color(traders)
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=price_df['global_ts'], y=price_df['mid_price'],
        line=dict(color='rgba(100,100,100,0.4)', width=1),
        name='mid price'
    ), secondary_y=True)

    for trader in traders:
        col    = colors[trader]
        buys   = trade_df[trade_df['buyer']  == trader][['global_ts', 'quantity']].copy()
        sells  = trade_df[trade_df['seller'] == trader][['global_ts', 'quantity']].copy()
        buys['signed']  =  buys['quantity']
        sells['signed'] = -sells['quantity']
        combined = pd.concat([buys[['global_ts', 'signed']],
                               sells[['global_ts', 'signed']]]).sort_values('global_ts')
        combined['cumulative'] = combined['signed'].cumsum()

        fig.add_trace(go.Scatter(
            x=combined['global_ts'], y=combined['cumulative'],
            line=dict(color=col, width=2),
            name=f'{trader} inventory'
        ), secondary_y=False)

    fig.update_layout(
        title=f'Cumulative Signed Inventory per Trader {title_suffix}',
        xaxis_title='global timestamp'
    )
    fig.update_yaxes(title_text='net inventory (+ = net long)', secondary_y=False)
    fig.update_yaxes(title_text='mid price', secondary_y=True)
    return add_day_vlines(fig, price_df)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 3: PER-TRADER MARKOUT CURVES
# All key traders on one chart. Immediately ranks who has information edge.
# Upward-sloping = informed. Flat/negative = noise/MM.
# ══════════════════════════════════════════════════════════════════════════════

def compute_markout_signed(trade_df, price_df, horizons):
    """Returns trade_df with markout columns and aggressor_side attached."""
    p = price_df[['global_ts', 'day', 'mid_price']].sort_values('global_ts')
    df = pd.merge_asof(trade_df.sort_values('global_ts'),
                       p[['global_ts', 'mid_price']].rename(columns={'mid_price': 'mid_at_trade'}),
                       on='global_ts', direction='nearest')

    # infer aggressor side
    ob = price_df[['global_ts', 'ask_price_1', 'bid_price_1']].sort_values('global_ts')
    df = pd.merge_asof(df, ob, on='global_ts', direction='nearest')
    df['aggressor_side'] = np.where(df['price'] >= df['ask_price_1'], 'buy',
                           np.where(df['price'] <= df['bid_price_1'], 'sell', 'ambiguous'))

    for h in horizons:
        future = p.copy()
        future['global_ts'] = future['global_ts'] - h
        future = future.rename(columns={'mid_price': f'mid_t{h}'})
        df = pd.merge_asof(df, future[['global_ts', 'day', f'mid_t{h}']],
                           on='global_ts', direction='forward',
                           suffixes=('', f'_f{h}'))
        day_col = f'day_f{h}' if f'day_f{h}' in df.columns else 'day_y'
        if day_col in df.columns:
            df[f'mid_t{h}'] = df[f'mid_t{h}'].where(df[day_col] == df['day'], np.nan)
            df = df.drop(columns=[day_col])
        sign = df['aggressor_side'].map({'buy': 1, 'sell': -1}).fillna(0)
        df[f'markout_{h}'] = (df[f'mid_t{h}'] - df['mid_at_trade']) * sign
        df = df.drop(columns=[f'mid_t{h}'])
    return df


def plot_per_trader_markout(trade_df, price_df, traders, horizons, title_suffix=''):
    """
    One line per trader showing mean signed markout at each horizon.
    The trader with the highest curve is your most informed counterparty.
    """
    colors  = trader_color(traders)
    markout_df = compute_markout_signed(trade_df, price_df, horizons)
    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color='black', dash='dash', width=1),
                  annotation_text='zero (no edge)')

    for trader in traders:
        col  = colors[trader]
        mask = (markout_df['buyer'] == trader) | (markout_df['seller'] == trader)
        sub  = markout_df[mask]
        if sub.empty:
            continue
        avgs = [sub[f'markout_{h}'].mean() for h in horizons]
        fig.add_trace(go.Scatter(
            x=horizons, y=avgs, mode='lines+markers',
            line=dict(color=col, width=2),
            marker=dict(size=7),
            name=trader
        ))

    fig.update_layout(
        title=f'Per-Trader Mean Signed Markout {title_suffix}',
        xaxis_title='horizon (ticks after trade)',
        yaxis_title='avg price change in trade direction',
        legend=dict(x=1.01, y=1)
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 4: TRADER INTERACTION HEATMAP
# Who trades with whom and how much volume.
# Instantly shows structural pairs (Mark 55 <-> Mark 14 will jump out).
# ══════════════════════════════════════════════════════════════════════════════

def plot_interaction_heatmap(trade_df, title_suffix=''):
    matrix = trade_df.groupby(['buyer', 'seller'])['quantity'].sum().reset_index()
    buyers  = sorted(matrix['buyer'].unique())
    sellers = sorted(matrix['seller'].unique())
    all_traders = sorted(set(buyers) | set(sellers))

    z = pd.DataFrame(0, index=all_traders, columns=all_traders)
    for _, row in matrix.iterrows():
        z.loc[row['buyer'], row['seller']] += row['quantity']

    fig = go.Figure(go.Heatmap(
        z=z.values,
        x=z.columns.tolist(),
        y=z.index.tolist(),
        colorscale='Blues',
        text=z.values.astype(int),
        texttemplate='%{text}',
        hovertemplate='buyer: %{y}<br>seller: %{x}<br>volume: %{z}<extra></extra>'
    ))
    fig.update_layout(
        title=f'Counterparty Interaction Volume Heatmap {title_suffix}',
        xaxis_title='seller',
        yaxis_title='buyer',
        xaxis=dict(tickangle=45)
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 5: ROLLING AGGRESSION RATE PER TRADER OVER TIME
# Shows if a trader shifts from passive to aggressive (and when).
# A sudden spike in aggression rate = they just got a signal.
# ══════════════════════════════════════════════════════════════════════════════

def plot_rolling_aggression(trade_df, price_df, traders, window=50, title_suffix=''):
    colors = trader_color(traders)
    ob = price_df[['global_ts', 'ask_price_1', 'bid_price_1']].sort_values('global_ts')
    df = pd.merge_asof(trade_df.sort_values('global_ts'), ob,
                       on='global_ts', direction='nearest')
    df['is_aggressor_buy']  = (df['price'] >= df['ask_price_1']).astype(int)
    df['is_aggressor_sell'] = (df['price'] <= df['bid_price_1']).astype(int)
    df['is_aggressor'] = np.maximum(df['is_aggressor_buy'], df['is_aggressor_sell'])

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=price_df['global_ts'], y=price_df['mid_price'],
        line=dict(color='rgba(100,100,100,0.3)', width=1),
        name='mid price'
    ), secondary_y=True)

    for trader in traders:
        col  = colors[trader]
        mask = (df['buyer'] == trader) | (df['seller'] == trader)
        sub  = df[mask].sort_values('global_ts')
        if len(sub) < window:
            continue
        sub['rolling_aggression'] = sub['is_aggressor'].rolling(window, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=sub['global_ts'], y=sub['rolling_aggression'],
            line=dict(color=col, width=2),
            name=f'{trader} aggression (w={window})'
        ), secondary_y=False)

    fig.update_layout(
        title=f'Rolling Aggression Rate per Trader (window={window}) {title_suffix}',
        xaxis_title='global timestamp'
    )
    fig.update_yaxes(title_text='aggression rate (0=passive, 1=always aggressive)', secondary_y=False)
    fig.update_yaxes(title_text='mid price', secondary_y=True)
    return add_day_vlines(fig, price_df)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 6: CROSS-PRODUCT VIEW FOR SPREAD TRADERS (Mark 14 / Mark 22)
# GEL and EXTRACT mid prices on same chart with trader's trades overlaid.
# Shows which leg they do first and whether they're trading the spread.
# ══════════════════════════════════════════════════════════════════════════════

def plot_cross_product_trader(gel_price, extract_price,
                               gel_trades, extract_trades, trader):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=[f'{trader} — GEL trades on GEL mid',
                                        f'{trader} — EXTRACT trades on EXTRACT mid'])

    fig.add_trace(go.Scatter(x=gel_price['global_ts'], y=gel_price['mid_price'],
                             line=dict(color='steelblue', width=1), name='GEL mid'),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=extract_price['global_ts'], y=extract_price['mid_price'],
                             line=dict(color='darkorange', width=1), name='EXTRACT mid'),
                  row=2, col=1)

    for product_trades, price_df, row, product_label in [
        (gel_trades,     gel_price,     1, 'GEL'),
        (extract_trades, extract_price, 2, 'EXTRACT'),
    ]:
        t = attach_mid(product_trades, price_df)
        buys  = t[t['buyer']  == trader]
        sells = t[t['seller'] == trader]
        if not buys.empty:
            fig.add_trace(go.Scatter(
                x=buys['global_ts'], y=buys['price'], mode='markers',
                marker=dict(symbol='triangle-up', size=10, color='green'),
                name=f'{trader} {product_label} BUY'), row=row, col=1)
        if not sells.empty:
            fig.add_trace(go.Scatter(
                x=sells['global_ts'], y=sells['price'], mode='markers',
                marker=dict(symbol='triangle-down', size=10, color='red'),
                name=f'{trader} {product_label} SELL'), row=row, col=1)

    fig.update_layout(title=f'{trader} — Cross-Product Trading Behaviour',
                      legend=dict(x=1.01, y=1))
    fig.update_xaxes(title_text='global timestamp', row=2, col=1)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 7: COINTEGRATION SPREAD WITH TRADER TRADES OVERLAID
# The spread z-score over time with Mark 14/22 entry points marked.
# Shows whether they're trading mean reversion correctly.
# ══════════════════════════════════════════════════════════════════════════════

def plot_spread_with_traders(gel_price, extract_price,
                              gel_trades, extract_trades,
                              traders, beta=-0.6465, mu=11709.13):
    combined = pd.merge_asof(
        extract_price[['global_ts', 'day', 'mid_price']].rename(
            columns={'mid_price': 'extract_mid'}).sort_values('global_ts'),
        gel_price[['global_ts', 'mid_price']].rename(
            columns={'mid_price': 'gel_mid'}).sort_values('global_ts'),
        on='global_ts', direction='nearest'
    )
    combined['spread']  = combined['extract_mid'] + beta * combined['gel_mid'] - mu
    spread_mean = combined['spread'].mean()
    spread_std  = combined['spread'].std()
    combined['zscore']  = (combined['spread'] - spread_mean) / spread_std

    colors = trader_color(traders)
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=combined['global_ts'], y=combined['zscore'],
        line=dict(color='steelblue', width=1.5), name='spread z-score'
    ), secondary_y=False)

    for level, col, label in [(2, 'red', '+2σ'), (-2, 'red', '-2σ'),
                               (1, 'orange', '+1σ'), (-1, 'orange', '-1σ'),
                               (0, 'black', 'mean')]:
        fig.add_hline(y=level, line=dict(color=col, dash='dash', width=0.8),
                      annotation_text=label, secondary_y=False)

    # overlay trader extract trades on secondary y (raw price for reference)
    for trader in traders:
        col = colors[trader]
        t   = attach_mid(extract_trades, extract_price)
        sub = t[(t['buyer'] == trader) | (t['seller'] == trader)]
        if sub.empty:
            continue
        # map their trade timestamps to the nearest spread z-score
        sub_z = pd.merge_asof(sub.sort_values('global_ts'),
                               combined[['global_ts', 'zscore']],
                               on='global_ts', direction='nearest')
        fig.add_trace(go.Scatter(
            x=sub_z['global_ts'], y=sub_z['zscore'],
            mode='markers',
            marker=dict(color=col, size=8, symbol='circle-open', line=dict(width=2)),
            name=f'{trader} EXTRACT trade'
        ), secondary_y=False)

    fig.update_layout(
        title=f'Cointegration Spread Z-Score (EXTRACT + {beta}×GEL − {mu:.0f}) with Trader Entries',
        xaxis_title='global timestamp',
        legend=dict(x=1.01, y=1)
    )
    fig.update_yaxes(title_text='z-score', secondary_y=False)
    return add_day_vlines(fig, combined)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 8: MARK 67 DEEP DIVE — trade timing vs mid price change
# For each of Mark 67's trades, what did price do next.
# If he's informed, you'll see consistent directional moves after his trades.
# ══════════════════════════════════════════════════════════════════════════════

def plot_informed_trader_deep_dive(trade_df, price_df, trader, horizons, title_suffix=''):
    markout_df = compute_markout_signed(trade_df, price_df, horizons)
    sub = markout_df[(markout_df['buyer'] == trader) | (markout_df['seller'] == trader)].copy()

    if sub.empty:
        print(f"No trades found for {trader}")
        return None

    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=[f'{trader} — Trade price vs mid at trade',
                                        f'{trader} — Markout distribution at each horizon'])

    # Top: trade price deviation from mid
    sub['dev'] = sub['price'] - sub['mid_at_trade']
    fig.add_trace(go.Scatter(
        x=sub['global_ts'], y=sub['dev'], mode='markers',
        marker=dict(color=sub['dev'].apply(lambda x: 'green' if x > 0 else 'red'),
                    size=7, opacity=0.7),
        name='trade price - mid'
    ), row=1, col=1)
    fig.add_hline(y=0, line=dict(color='black', dash='dash'), row=1, col=1)

    # Bottom: box plot of markout distribution per horizon
    for h in horizons:
        col_name = f'markout_{h}'
        if col_name not in sub.columns:
            continue
        fig.add_trace(go.Box(
            y=sub[col_name].dropna(), name=f'h={h}',
            boxmean=True,
            marker_color='steelblue'
        ), row=2, col=1)

    fig.update_layout(
        title=f'{trader} Deep Dive {title_suffix}',
        legend=dict(x=1.01, y=1)
    )
    fig.update_yaxes(title_text='trade price - mid', row=1, col=1)
    fig.update_yaxes(title_text='signed markout', row=2, col=1)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# MASTER RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_all(product_price, product_trades, gel_price, extract_price,
            gel_trades, extract_trades, product_label):

    print(f"\n{'='*60}")
    print(f"TRADER BEHAVIOUR EDA — {product_label}")
    print(f"{'='*60}")

    # 1. All key traders on mid price — eyeball who's informed immediately
    fig1 = plot_trader_trades_on_mid(product_price, product_trades,
                                      KEY_TRADERS, title_suffix=f'({product_label})')
    fig1.show()

    # 2. Cumulative inventory — who's accumulating and when
    fig2 = plot_cumulative_inventory(product_trades, product_price,
                                      KEY_TRADERS, title_suffix=f'({product_label})')
    fig2.show()

    # 3. Markout curves — ranks traders by information quality
    fig3 = plot_per_trader_markout(product_trades, product_price,
                                    KEY_TRADERS, HORIZONS,
                                    title_suffix=f'({product_label})')
    fig3.show()

    # 4. Interaction heatmap — structural relationships
    fig4 = plot_interaction_heatmap(product_trades, title_suffix=f'({product_label})')
    fig4.show()

    # 5. Rolling aggression — detects when traders go directional
    fig5 = plot_rolling_aggression(product_trades, product_price,
                                    KEY_TRADERS, window=200,
                                    title_suffix=f'({product_label})')
    fig5.show()

    # 6. Mark 67 deep dive — your most informed trader
    fig6 = plot_informed_trader_deep_dive(product_trades, product_price,
                                           'Mark 67', HORIZONS,
                                           title_suffix=f'({product_label})')
    if fig6:
        fig6.show()


# ── Cross-product plots (run once, not per-product) ───────────────────────────
def run_cross_product():
    # Spread z-score with trader entries
    fig_spread = plot_spread_with_traders(
        gel_price, extract_price,
        gel_trades, extract_trades,
        traders=['Mark 14', 'Mark 22']
    )
    fig_spread.show()

    # Per-trader cross-product view
    for trader in ['Mark 14', 'Mark 22']:
        fig = plot_cross_product_trader(
            gel_price, extract_price,
            gel_trades, extract_trades,
            trader=trader
        )
        fig.show()


# ── RUN ───────────────────────────────────────────────────────────────────────
run_all(extract_price, extract_trades,
        gel_price, extract_price,
        gel_trades, extract_trades,
        product_label='EXTRACT')

run_all(gel_price, gel_trades,
        gel_price, extract_price,
        gel_trades, extract_trades,
        product_label='GEL')

run_cross_product()
# print(extract_trades[extract_trades['buyer'] == 'Mark 67'].sort_values('global_ts')['global_ts'].diff().dropna().values)
gelburst = gel_trades[gel_trades['buyer'] == 'Mark 14'].sort_values('global_ts')['global_ts']
extractburst = extract_trades[extract_trades['buyer'] == 'Mark 14'].sort_values('global_ts')['global_ts']

fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    subplot_titles=['Mark 14 Buys — GEL', 'Mark 14 Buys — EXTRACT'],
                    vertical_spacing=0.12)

for row, (bursts, color) in enumerate([(gelburst, 'steelblue'), (extractburst, 'tomato')], start=1):
    if len(bursts) > 1:
        ts_range = pd.Series(range(int(bursts.min()), int(bursts.max()), 500))
        rate = ts_range.apply(lambda t: ((bursts >= t) & (bursts < t + 1000)).sum())
        fig.add_trace(go.Scatter(x=ts_range, y=rate, mode='lines',
                                 line=dict(color=color, width=1.5),
                                 name=['GEL', 'EXTRACT'][row-1]), row=row, col=1)

    # Rug as scatter on x-axis
    fig.add_trace(go.Scatter(x=bursts, y=[0]*len(bursts), mode='markers',
                             marker=dict(symbol='line-ns', size=8, color=color,
                                         line=dict(color=color, width=1)),
                             name=f"{'GEL' if row==1 else 'EXTRACT'} events",
                             showlegend=False), row=row, col=1)

fig.update_layout(title='Mark 14 Burst Detection', height=500, template='plotly_dark')
# fig.show()

def mark14_quote_behaviour(gel_price, gel_trades):
    """
    What prices does Mark 14 actually trade at vs prevailing mid.
    Tells us his effective spread and how tight he quotes.
    """
    t = attach_mid(gel_trades, gel_price)
    m14 = t[(t['buyer'] == 'Mark 14') | (t['seller'] == 'Mark 14')].copy()
    m14['dev_from_mid'] = m14['price'] - m14['mid_price']
    m14['side'] = np.where(m14['buyer'] == 'Mark 14', 'buy', 'sell')

    print("Mark 14 GEL — trade price deviation from mid:")
    print(m14.groupby('side')['dev_from_mid'].describe())
    return m14


def mark38_trade_behaviour(gel_price, gel_trades):
    """
    What prices does Mark 38 trade at vs prevailing mid.
    Since he buys high and sells low, his deviations should be opposite to Mark 14.
    """
    t = attach_mid(gel_trades, gel_price)
    m38 = t[(t['buyer'] == 'Mark 38') | (t['seller'] == 'Mark 38')].copy()
    m38['dev_from_mid'] = m38['price'] - m38['mid_price']
    m38['side'] = np.where(m38['buyer'] == 'Mark 38', 'buy', 'sell')

    print("Mark 38 GEL — trade price deviation from mid:")
    print(m38.groupby('side')['dev_from_mid'].describe())
    return m38


def quote_competition_analysis(gel_price, gel_trades):
    """
    For each Mark 38 trade, what was the prevailing bid/ask at that moment.
    Tells us exactly what spread Mark 14 is quoting when Mark 38 hits him.
    This is the spread we need to undercut.
    """
    ob = gel_price[['global_ts', 'ask_price_1', 'bid_price_1',
                     'ask_price_2', 'bid_price_2',
                     'mid_price']].sort_values('global_ts')

    m38_trades = gel_trades[(gel_trades['buyer'] == 'Mark 38') |
                             (gel_trades['seller'] == 'Mark 38')].sort_values('global_ts')

    merged = pd.merge_asof(m38_trades, ob, on='global_ts', direction='nearest')
    merged['side'] = np.where(merged['buyer'] == 'Mark 38', 'buy', 'sell')
    merged['distance_from_best_ask'] = merged['price'] - merged['ask_price_1']
    merged['distance_from_best_bid'] = merged['price'] - merged['bid_price_1']

    print("\nMark 38 buys — distance from best ask (negative = below ask, 0 = hit ask):")
    print(merged[merged['side'] == 'buy']['distance_from_best_ask'].describe())
    print("\nMark 38 sells — distance from best bid (positive = above bid, 0 = hit bid):")
    print(merged[merged['side'] == 'sell']['distance_from_best_bid'].describe())

    return merged


def effective_spread_comparison(gel_price, gel_trades):
    """
    Compare effective spread of Mark 14 vs Mark 38 side by side.
    The gap between them is your undercutting opportunity.
    """
    t = attach_mid(gel_trades, gel_price)

    results = {}
    for trader in ['Mark 14', 'Mark 38']:
        sub = t[(t['buyer'] == trader) | (t['seller'] == trader)].copy()
        sub['side'] = np.where(sub['buyer'] == trader, 'buy', 'sell')
        sub['dev'] = sub['price'] - sub['mid_price']
        buys  = sub[sub['side'] == 'buy']['dev'].mean()
        sells = sub[sub['side'] == 'sell']['dev'].mean()
        eff_spread = sells - buys
        results[trader] = {'avg_buy_dev': buys, 'avg_sell_dev': sells,
                           'effective_spread': eff_spread}
        print(f"{trader}: avg buy dev={buys:.2f}, avg sell dev={sells:.2f}, "
              f"effective spread={eff_spread:.2f}")

    return results


# ── RUN ───────────────────────────────────────────────────────────────────────
m14_behaviour = mark14_quote_behaviour(gel_price, gel_trades)
m38_behaviour = mark38_trade_behaviour(gel_price, gel_trades)
m38_ob        = quote_competition_analysis(gel_price, gel_trades)
spreads       = effective_spread_comparison(gel_price, gel_trades)
