import numpy as np
import pandas as pd
import plotly.express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots

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


# ---------------------- MODULE 1: DATA PREP ------------------ #

def data_prep(trade_df, price_df):
    price_df = price_df[['global_ts', 'ask_price_1', 'bid_price_1', 'mid_price']].sort_values('global_ts')
    trade_df = trade_df.sort_values('global_ts').copy()

    combined_df = pd.merge_asof(trade_df, price_df, on='global_ts', direction='nearest')

    n_before = len(combined_df)

    aggressor_buy = combined_df[combined_df['price'] >= combined_df['ask_price_1']].copy()
    aggressor_buy['trader'] = aggressor_buy['buyer']
    aggressor_buy['aggressor_side'] = 'buy'

    aggressor_sell = combined_df[combined_df['price'] <= combined_df['bid_price_1']].copy()
    aggressor_sell['trader'] = aggressor_sell['seller']
    aggressor_sell['aggressor_side'] = 'sell'

    ambiguous = combined_df[
        (combined_df['price'] < combined_df['ask_price_1']) &
        (combined_df['price'] > combined_df['bid_price_1'])
        ].copy()

    n_after = len(aggressor_buy) + len(aggressor_sell)
    n_ambiguous = len(ambiguous)
    print(f"Trades classified: {n_after}/{n_before} | Ambiguous/dropped: {n_ambiguous}")

    result = pd.concat([aggressor_buy, aggressor_sell], ignore_index=True)
    result = result.sort_values('global_ts').reset_index(drop=True)

    return result, ambiguous


results = data_prep(extract_trades,extract_price)
res2 = data_prep(gel_trades, gel_price)



# ---------------------- MODULE 2: MARKOUT ------------------ #

def markout(price_df, trade_df, horizons):
    price_df = price_df[['global_ts', 'day', 'mid_price']].sort_values('global_ts')
    trade_df = trade_df.sort_values('global_ts').copy()

    trade_df = pd.merge_asof(trade_df,
                             price_df[['global_ts', 'mid_price']].rename(columns={'mid_price': 'mid_at_trade'}),
                             on='global_ts', direction='nearest')

    for h in horizons:
        future = price_df.copy()
        future['global_ts'] = future['global_ts'] - h
        future = future.rename(columns={'mid_price': f'mid_t{h}'})
        trade_df = pd.merge_asof(trade_df, future[['global_ts', 'day', f'mid_t{h}']],
                                 on='global_ts', direction='forward',
                                 suffixes=('', f'_future_{h}'))

        day_col = f'day_future_{h}' if f'day_future_{h}' in trade_df.columns else 'day_y'
        if day_col in trade_df.columns:
            trade_df[f'mid_t{h}'] = trade_df[f'mid_t{h}'].where(
                trade_df[day_col] == trade_df['day'], other=float('nan')
            )
            trade_df = trade_df.drop(columns=[day_col])

        raw_markout = trade_df[f'mid_t{h}'] - trade_df['mid_at_trade']
        trade_df[f'markout_{h}'] = raw_markout * trade_df['aggressor_side'].map({'buy': 1, 'sell': -1})
        trade_df = trade_df.drop(columns=[f'mid_t{h}'])

    n_total = len(trade_df)
    for h in horizons:
        n_nan = trade_df[f'markout_{h}'].isna().sum()
        print(f"markout_{h}: {n_nan}/{n_total} NaN (day boundary or missing)")


    return trade_df


markout_info = markout(extract_price, results[0], [10,20,50,150])
per_trader_markout = markout_info.groupby('trader')[[f'markout_{h}' for h in [10,20,50,150]]].mean()
print(per_trader_markout)
markout_info_2 = markout(gel_price, res2[0], [10,20,50,150])
per_trader_markout = markout_info_2.groupby('trader')[[f'markout_{h}' for h in [10,20,50,150]]].mean()
print(per_trader_markout)

# ----------------------- MODULE 3: TRADER PROFILES ---------------- #


def trader_profile(trade_df, price_df, res_df):
    total_traders = pd.concat([trade_df['buyer'], trade_df['seller']],axis=0)
    traders = total_traders.unique().tolist()
    combined_df = pd.merge_asof(trade_df, price_df, on='global_ts', direction='nearest')
    trade_interactions = combined_df.groupby(['buyer', 'seller'])['quantity'].sum()
    total_bought = combined_df.groupby(['buyer'])['quantity'].sum()
    total_sold = combined_df.groupby(['seller'])['quantity'].sum()
    mean_activity = []
    median_activity = []
    for trader in traders:
        buy_df = combined_df[combined_df['buyer'] == trader]['quantity']
        sell_df = combined_df[combined_df['seller'] == trader]['quantity']
        mean_activity.append(pd.concat([buy_df, sell_df], ignore_index=True, axis=0).mean())
        median_activity.append(pd.concat([buy_df, sell_df], ignore_index=True,axis=0).median())
    mean_traded_qty = pd.Series(index=traders, data=mean_activity)
    median_traded_qty = pd.Series(index=traders, data=median_activity)
    aggressors = res_df['trader'].value_counts()/pd.concat([trade_df['buyer'], trade_df['seller']],axis=0).value_counts()
    aggressive_trades = res_df['trader'].value_counts() / res_df['trader'].shape[0]
    trader_profile_df = pd.DataFrame({'mean_vol_traded':mean_traded_qty,
                                      'median_vol_traded': median_traded_qty,
                                      'total_bought': total_bought,
                                      'total_sold': total_sold,
                                      'aggression_rate': aggressors,
                                      'aggressive_trades': aggressive_trades,})
    return trader_profile_df, trade_interactions


trader_data = trader_profile(extract_trades, extract_price, results[0])
pass
# ----------------------- MODULE 4: TIMING AND PRICE IMPACT ---------------- #


def timing_price_impact(res_df, timestamp, trader_profile_df, price_df):
    trader_profile_df['pre_exceed'] = 0.
    trader_profile_df['post_exceed'] = 0.
    traders = pd.concat([res_df['buyer'], res_df['seller']], axis=0).unique().tolist()
    price_pre = price_df.copy()
    price_pre['global_ts'] = price_pre['global_ts'] + timestamp
    price_pre = price_pre.rename(columns={'mid_price': f'mid_t{-timestamp}'})
    res_df = pd.merge_asof(res_df, price_pre[['global_ts', 'day', f'mid_t{-timestamp}']], on='global_ts', direction='forward', suffixes=('', f'_mid_t{-timestamp}'))
    day_col = f'day_mid_t{-timestamp}' if f'day_mid_t{-timestamp}' in res_df.columns else 'day_y'
    if day_col in res_df.columns:
        res_df[f'mid_t{-timestamp}'] = res_df[f'mid_t{-timestamp}'].where(
            res_df[day_col] == res_df['day'], other=float('nan')
        )
        res_df = res_df.drop(columns=[day_col])
    average_pre = np.mean(np.abs(res_df['mid_price'] - res_df[f'mid_t{-timestamp}']))
    price_post = price_df.copy()
    price_post['global_ts'] = price_post['global_ts'] - timestamp
    price_post = price_post.rename(columns={'mid_price': f'mid_t{timestamp}'})
    res_df = pd.merge_asof(res_df, price_post[['global_ts', 'day', f'mid_t{timestamp}']], on='global_ts', direction='forward', suffixes=('', f'_mid_t{timestamp}'))
    day_col = f'day_mid_t{timestamp}' if f'day_mid_t{timestamp}' in res_df.columns else 'day_y'
    if day_col in res_df.columns:
        res_df[f'mid_t{timestamp}'] = res_df[f'mid_t{timestamp}'].where(
            res_df[day_col] == res_df['day'], other=float('nan')
        )
        res_df = res_df.drop(columns=[day_col])
    average_post = np.mean(np.abs(res_df[f'mid_t{timestamp}'] - res_df['mid_price']))
    for trader in traders:
        buy_df = res_df[res_df['buyer'] == trader]
        sell_df = res_df[res_df['seller'] == trader]
        temp_df = pd.concat([buy_df, sell_df], axis=0)
        trader_pre = np.mean(np.abs(temp_df['mid_price'] - temp_df[f'mid_t{-timestamp}']))
        trader_post = np.mean(np.abs(temp_df[f'mid_t{timestamp}'] - temp_df['mid_price']))
        if trader_pre > average_pre:
            trader_profile_df.loc[trader, 'pre_exceed'] = trader_pre/average_pre
        if trader_post > average_post:
            trader_profile_df.loc[trader, 'post_exceed'] = trader_post/average_post
    return trader_profile_df


# ----------------------- MODULE 5: CROSS PRODUCT BEHAVIOUR ---------------- #

# Module 5: Cross-Product Behaviour
# Filter to trades involving GEL and EXTRACT. For each trader,
# find timestamps where they traded both products within a
# 5-timestamp window. Record direction of each leg.
# Output: per trader, how often they trade the spread (same direction both legs vs opposite).
# This tells you who knows about the cointegration relationship.
def cross_prod(timestamp, gel, extract, res_df, trader_profile):
    trader_profile['same_leg_freq'] = 0.
    trader_profile['opp_leg_freq'] = 0.

    m14_buy_gel = gel[gel['buyer'] == 'Mark 14']
    m14_sell_gel = gel[gel['seller'] == 'Mark 14']
    m14_buy_extract = extract[extract['buyer'] == 'Mark 14']
    m14_sell_extract = extract[extract['seller'] == 'Mark 14']
    m14_gel = pd.concat([m14_buy_gel.assign(gel_side='buy'),
                         m14_sell_gel.assign(gel_side='sell')], axis=0).sort_values('global_ts')
    m14_extract = pd.concat([m14_buy_extract.assign(extract_side='buy'),
                             m14_sell_extract.assign(extract_side='sell')], axis=0).sort_values('global_ts')
    m14 = pd.merge_asof(m14_gel, m14_extract, on='global_ts', tolerance=timestamp,
                        direction='nearest', suffixes=('_gel', '_extract'))

    m22_buy_gel = gel[gel['buyer'] == 'Mark 22']
    m22_sell_gel = gel[gel['seller'] == 'Mark 22']
    m22_buy_extract = extract[extract['buyer'] == 'Mark 22']
    m22_sell_extract = extract[extract['seller'] == 'Mark 22']
    m22_gel = pd.concat([m22_buy_gel.assign(gel_side='buy'),
                         m22_sell_gel.assign(gel_side='sell')], axis=0).sort_values('global_ts')
    m22_extract = pd.concat([m22_buy_extract.assign(extract_side='buy'),
                             m22_sell_extract.assign(extract_side='sell')], axis=0).sort_values('global_ts')
    m22 = pd.merge_asof(m22_gel, m22_extract, on='global_ts', tolerance=timestamp,
                        direction='nearest', suffixes=('_gel', '_extract'))

    for trader, merged in [('Mark 14', m14), ('Mark 22', m22)]:
        paired = merged.dropna(subset=['extract_side'])
        total = len(paired)
        if total == 0:
            print(f"{trader}: no paired trades found at tolerance={timestamp}")
            continue
        same = (paired['gel_side'] == paired['extract_side']).sum()
        opp = total - same
        trader_profile.loc[trader, 'same_leg_freq'] = same / total
        trader_profile.loc[trader, 'opp_leg_freq'] = opp / total
        print(f"{trader}: {total} paired trades | same={same / total:.2%} | opp={opp / total:.2%}")


cross_prod(100, gel_trades, extract_trades, results[0], trader_data[0])

