# # prosperity4btest cli my_trader.py 3

# from datamodel import OrderDepth, TradingState, Order
# from typing import Dict, List, Tuple, Optional
# import json
# import math

# # ─── Position limits ──────────────────────────────────────────────────────────

# POSITION_LIMITS: Dict[str, int] = {
#     "HYDROGEL_PACK": 200,
#     "VELVETFRUIT_EXTRACT": 200,
#     **{f"VEV_{k}": 300 for k in [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]},
# }

# # ─── Parameters ───────────────────────────────────────────────────────────────

# # VEV spot
# VEV_EMA_ALPHA      = 0.02
# VEV_PRIMARY_QTY    = 30
# VEV_SECONDARY_QTY  = 15
# VEV_EMA_THRESH     = 12.0
# VEV_MM_SPREAD      = 3
# VEV_MM_QTY        = 2
# VEV_NEUTRAL_THRESH = 8.0
# VEV_MAX_POS        = 200

# # VEV_4000 — same three-mode strategy as spot but passive only (never cross spread)
# # Wide 20-tick spread means we NEVER hit ask/bid aggressively.
# # Instead post limit orders inside the spread and let market come to us.
# # Smaller qty than spot since spread is wider and fills will be less frequent.
# V4K_PRIMARY_QTY    = 15
# V4K_SECONDARY_QTY  = 8
# V4K_EMA_THRESH     = 12.0    # same signal threshold as spot
# V4K_MM_SPREAD      = 8       # post inside the 20-tick market spread
# V4K_MM_QTY        = 2
# V4K_NEUTRAL_THRESH = 8.0
# V4K_MAX_POS        = 100     # conservative — wide spread = slower turnover

# # HP market-making
# HP_EMA_ALPHA   = 0.001
# HP_HALF_SPREAD = 6
# HP_QTY         = 2

# # Options
# TTE_DAYS_BASE    = 5           # days remaining at start of round 3
# TICKS_PER_DAY    = 1_000_000
# OPT_STRIKES      = [5000, 5100, 5200]
# OPT_QTY          = 20
# OPT_MAX_POS      = 250
# OPT_VOL_WINDOW   = 100        # log-return samples for realised vol estimate
# OPT_BASE_SIGMA   = 0.15       # fallback annualised vol

# # IV residual time-series parameters
# OPT_RES_WINDOW     = 50       # rolling window for IV residual mean/std
# OPT_RES_MIN_WINDOW = 20       # minimum samples before trading
# OPT_Z_THRESH       = 1.5      # z-score threshold to trigger trade

# # ─── Helpers ──────────────────────────────────────────────────────────────────

# def _best_bid(depth: OrderDepth) -> Optional[float]:
#     return max(depth.buy_orders) if depth.buy_orders else None

# def _best_ask(depth: OrderDepth) -> Optional[float]:
#     return min(depth.sell_orders) if depth.sell_orders else None

# def _mid(depth: OrderDepth) -> Optional[float]:
#     b = _best_bid(depth); a = _best_ask(depth)
#     return (b + a) / 2.0 if b and a else None

# def _norm_cdf(x: float) -> float:
#     return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

# def _bs_call_price(S: float, K: float, T: float, sigma: float) -> float:
#     """Black-Scholes call price (r=0, no dividends)."""
#     if T <= 0:
#         return max(S - K, 0.0)
#     if sigma <= 0:
#         return max(S - K, 0.0)
#     sqrtT = math.sqrt(T)
#     d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrtT)
#     d2 = d1 - sigma * sqrtT
#     return S * _norm_cdf(d1) - K * _norm_cdf(d2)

# def _bs_vega(S: float, K: float, T: float, sigma: float) -> float:
#     """BS vega = dC/dsigma."""
#     if T <= 0 or sigma <= 0:
#         return 0.0
#     sqrtT = math.sqrt(T)
#     d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrtT)
#     phi = math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
#     return S * phi * sqrtT

# def _implied_vol(market_price: float, S: float, K: float, T: float,
#                  sigma0: float = 0.3) -> Optional[float]:
#     """
#     Newton-Raphson IV solve. Returns None if market_price <= intrinsic
#     or if it fails to converge.
#     """
#     intrinsic = max(S - K, 0.0)
#     if market_price <= intrinsic + 1e-6:
#         return None
#     sigma = sigma0
#     for _ in range(30):
#         price = _bs_call_price(S, K, T, sigma)
#         vega  = _bs_vega(S, K, T, sigma)
#         if vega < 1e-8:
#             break
#         diff  = price - market_price
#         sigma -= diff / vega
#         sigma  = max(sigma, 1e-6)
#         if abs(diff) < 1e-5:
#             return sigma
#     return sigma if sigma > 0 else None


# # ─── Trader ───────────────────────────────────────────────────────────────────

# class Trader:
#     """
#     Round 3 strategy:

#     1. VELVETFRUIT_EXTRACT — unchanged three-mode MR strategy from v1
#        (new daily extreme primary, EMA deviation secondary, passive MM neutral)

#     2. HYDROGEL_PACK — passive MM only, unchanged from v1

#     3. VEV options (5000/5100/5200) — IV smile scalping:
#        - Compute TTE accurately from timestamp
#        - Estimate annualised vol from rolling log-returns of VEV spot
#        - Compute implied vol for each strike from its mid price
#        - Fit a quadratic smile: IV = a*m^2 + b*m + c  (m = log(K/S))
#        - Trade any strike where |implied_vol - smile_iv| > SMILE_IV_EDGE
#          → sell if rich vs smile, buy if cheap vs smile
#        - Size conservatively: OPT_QTY, max OPT_MAX_POS

#     State: ema_vev, ema_hp, day_min, day_max, current_day,
#            returns_buf (list of last OPT_VOL_WINDOW log-returns),
#            last_vev_price
#     """

#     def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:

#         # ── Load persisted state ──────────────────────────────────────────────
#         try:
#             td = json.loads(state.traderData) if state.traderData else {}
#         except Exception:
#             td = {}

#         ema_vev      = float(td.get("ema_vev",      5250.0))
#         ema_hp       = float(td.get("ema_hp",        9990.0))
#         day_min      = float(td.get("day_min",        1e9))
#         day_max      = float(td.get("day_max",       -1e9))
#         current_day  = int(td.get("current_day",     -1))
#         returns_buf  = list(td.get("returns_buf",    []))
#         last_vev_px  = td.get("last_vev_px",         None)
#         if last_vev_px is not None:
#             last_vev_px = float(last_vev_px)

#         result: Dict[str, List[Order]] = {}

#         # ── Day rollover ──────────────────────────────────────────────────────
#         this_day = state.timestamp // TICKS_PER_DAY
#         if this_day != current_day:
#             day_min     = 1e9
#             day_max     = -1e9
#             current_day = this_day

#         # ── TTE calculation ───────────────────────────────────────────────────
#         # Each day ticks 0..999999. We're in round 3 => TTE_DAYS_BASE days left
#         # at the START of today. Subtract fraction of today already elapsed.
#         tick_within_day = state.timestamp % TICKS_PER_DAY
#         frac_day_elapsed = tick_within_day / TICKS_PER_DAY
#         tte_days = TTE_DAYS_BASE - frac_day_elapsed
#         T = max(tte_days / 365.0, 1e-6)

#         # ── Market data ───────────────────────────────────────────────────────
#         vev_depth = state.order_depths.get("VELVETFRUIT_EXTRACT")
#         hp_depth  = state.order_depths.get("HYDROGEL_PACK")

#         vev_mid = _mid(vev_depth) if vev_depth else None
#         hp_mid  = _mid(hp_depth)  if hp_depth  else None

#         # ── Update EMAs ───────────────────────────────────────────────────────
#         if vev_mid is not None:
#             ema_vev = VEV_EMA_ALPHA * vev_mid + (1 - VEV_EMA_ALPHA) * ema_vev
#         if hp_mid is not None:
#             ema_hp  = HP_EMA_ALPHA  * hp_mid  + (1 - HP_EMA_ALPHA)  * ema_hp

#         # ── Rolling vol from VEV log-returns ──────────────────────────────────
#         if vev_mid is not None and last_vev_px is not None and last_vev_px > 0:
#             ret = math.log(vev_mid / last_vev_px)
#             returns_buf.append(ret)
#         if vev_mid is not None:
#             last_vev_px = vev_mid

#         # Keep buffer bounded
#         if len(returns_buf) > OPT_VOL_WINDOW:
#             returns_buf = returns_buf[-OPT_VOL_WINDOW:]

#         # Annualise: ticks per second not defined, but we have ~1M ticks/day
#         # => each tick is 1/1M of a day. We'll annualise from per-tick vol.
#         # Actually simpler: compute realised vol per tick, scale to annual.
#         TICKS_PER_YEAR = TICKS_PER_DAY * 365
#         if len(returns_buf) >= 20:
#             mean = sum(returns_buf) / len(returns_buf)
#             var  = sum((r - mean) ** 2 for r in returns_buf) / len(returns_buf)
#             # var is per-tick variance; annualise
#             sigma_ann = math.sqrt(var * TICKS_PER_YEAR)
#             # Sanity clamp: options won't trade if vol is nonsense
#             sigma_ann = max(0.05, min(sigma_ann, 2.0))
#         else:
#             sigma_ann = OPT_BASE_SIGMA

#         S = vev_mid if vev_mid is not None else ema_vev

#         # ═══════════════════════════════════════════════════════════════════════
#         # 1. VELVETFRUIT_EXTRACT — three-mode strategy (unchanged from v1)
#         # ═══════════════════════════════════════════════════════════════════════
#         vev_pos    = state.position.get("VELVETFRUIT_EXTRACT", 0)
#         vev_orders: List[Order] = []

#         # Defaults — updated below if market data is present
#         new_daily_low  = False
#         new_daily_high = False
#         dev            = S - ema_vev

#         if vev_depth and vev_mid is not None:
#             bid = _best_bid(vev_depth)
#             ask = _best_ask(vev_depth)

#             old_min = day_min
#             old_max = day_max
#             day_min = min(day_min, vev_mid)
#             day_max = max(day_max, vev_mid)

#             new_daily_low  = vev_mid < old_min
#             new_daily_high = vev_mid > old_max
#             dev = vev_mid - ema_vev

#             # Mode 1: new daily extreme
#             if new_daily_low and vev_pos < VEV_MAX_POS:
#                 qty = min(VEV_PRIMARY_QTY, VEV_MAX_POS - vev_pos)
#                 if qty > 0 and ask:
#                     vev_orders.append(Order("VELVETFRUIT_EXTRACT", ask, qty))

#             elif new_daily_high and vev_pos > -VEV_MAX_POS:
#                 qty = min(VEV_PRIMARY_QTY, VEV_MAX_POS + vev_pos)
#                 if qty > 0 and bid:
#                     vev_orders.append(Order("VELVETFRUIT_EXTRACT", bid, -qty))

#             # Mode 2: EMA deviation
#             elif dev > VEV_EMA_THRESH and vev_pos > -VEV_MAX_POS:
#                 qty = min(VEV_SECONDARY_QTY, VEV_MAX_POS + vev_pos)
#                 if qty > 0 and bid:
#                     vev_orders.append(Order("VELVETFRUIT_EXTRACT", bid, -qty))

#             elif dev < -VEV_EMA_THRESH and vev_pos < VEV_MAX_POS:
#                 qty = min(VEV_SECONDARY_QTY, VEV_MAX_POS - vev_pos)
#                 if qty > 0 and ask:
#                     vev_orders.append(Order("VELVETFRUIT_EXTRACT", ask, qty))

#             # Mode 3: neutral regime passive MM
#             elif abs(dev) < VEV_NEUTRAL_THRESH:
#                 fv      = round(ema_vev)
#                 our_bid = fv - VEV_MM_SPREAD
#                 our_ask = fv + VEV_MM_SPREAD

#                 if ask and ask <= our_bid and vev_pos < VEV_MAX_POS:
#                     qty = min(VEV_MM_QTY, VEV_MAX_POS - vev_pos)
#                     if qty > 0:
#                         vev_orders.append(Order("VELVETFRUIT_EXTRACT", ask, qty))

#                 if bid and bid >= our_ask and vev_pos > -VEV_MAX_POS:
#                     qty = min(VEV_MM_QTY, VEV_MAX_POS + vev_pos)
#                     if qty > 0:
#                         vev_orders.append(Order("VELVETFRUIT_EXTRACT", bid, -qty))

#         if vev_orders:
#             result["VELVETFRUIT_EXTRACT"] = vev_orders

#         # ═══════════════════════════════════════════════════════════════════════
#         # 2. VEV_4000 — same three-mode strategy as spot, fully passive
#         #    Never cross the 20-tick spread. Post limit orders inside it.
#         #    Uses spot VEV signals (new_daily_low/high, dev) — same logic.
#         # ═══════════════════════════════════════════════════════════════════════
#         v4k_pos    = state.position.get("VEV_4000", 0)
#         v4k_depth  = state.order_depths.get("VEV_4000")
#         v4k_orders: List[Order] = []

#         if v4k_depth and vev_mid is not None:
#             v4k_bid = _best_bid(v4k_depth)
#             v4k_ask = _best_ask(v4k_depth)

#             if v4k_bid is not None and v4k_ask is not None:
#                 # Fair value: VEV_4000 is deep ITM delta≈1, so tracks spot
#                 # Use spot EMA offset by rough intrinsic (S - 4000)
#                 # For passive quoting we just use mid of v4k itself
#                 v4k_mid = (v4k_bid + v4k_ask) / 2.0

#                 # Post passive orders inside the spread, not crossing it
#                 passive_buy  = int(v4k_bid + 1)   # one tick above best bid
#                 passive_sell = int(v4k_ask - 1)   # one tick below best ask

#                 # Mode 1: new daily extreme — post passive limit in MR direction
#                 if new_daily_low and v4k_pos < V4K_MAX_POS:
#                     qty = min(V4K_PRIMARY_QTY, V4K_MAX_POS - v4k_pos)
#                     if qty > 0:
#                         v4k_orders.append(Order("VEV_4000", passive_buy, qty))

#                 elif new_daily_high and v4k_pos > -V4K_MAX_POS:
#                     qty = min(V4K_PRIMARY_QTY, V4K_MAX_POS + v4k_pos)
#                     if qty > 0:
#                         v4k_orders.append(Order("VEV_4000", passive_sell, -qty))

#                 # Mode 2: EMA deviation — smaller passive order
#                 elif dev > V4K_EMA_THRESH and v4k_pos > -V4K_MAX_POS:
#                     qty = min(V4K_SECONDARY_QTY, V4K_MAX_POS + v4k_pos)
#                     if qty > 0:
#                         v4k_orders.append(Order("VEV_4000", passive_sell, -qty))

#                 elif dev < -V4K_EMA_THRESH and v4k_pos < V4K_MAX_POS:
#                     qty = min(V4K_SECONDARY_QTY, V4K_MAX_POS - v4k_pos)
#                     if qty > 0:
#                         v4k_orders.append(Order("VEV_4000", passive_buy, qty))

#                 # Mode 3: neutral regime — passive MM inside spread
#                 elif abs(dev) < V4K_NEUTRAL_THRESH:
#                     our_bid = int(v4k_mid) - V4K_MM_SPREAD
#                     our_ask = int(v4k_mid) + V4K_MM_SPREAD

#                     if v4k_pos < V4K_MAX_POS:
#                         qty = min(V4K_MM_QTY, V4K_MAX_POS - v4k_pos)
#                         if qty > 0:
#                             v4k_orders.append(Order("VEV_4000", our_bid, qty))

#                     if v4k_pos > -V4K_MAX_POS:
#                         qty = min(V4K_MM_QTY, V4K_MAX_POS + v4k_pos)
#                         if qty > 0:
#                             v4k_orders.append(Order("VEV_4000", our_ask, -qty))

#         if v4k_orders:
#             result["VEV_4000"] = v4k_orders

#         # ═══════════════════════════════════════════════════════════════════════
#         # 3. HYDROGEL_PACK — passive MM (unchanged from v1)
#         # ═══════════════════════════════════════════════════════════════════════
#         hp_pos    = state.position.get("HYDROGEL_PACK", 0)
#         hp_orders: List[Order] = []

#         if hp_depth and hp_mid is not None:
#             hp_bid = _best_bid(hp_depth)
#             hp_ask = _best_ask(hp_depth)
#             fv      = round(ema_hp)
#             our_bid = fv - HP_HALF_SPREAD
#             our_ask = fv + HP_HALF_SPREAD

#             skew     = hp_pos / POSITION_LIMITS["HYDROGEL_PACK"]
#             buy_qty  = max(1, round(HP_QTY * (1 - max(0.0,  skew))))
#             sell_qty = max(1, round(HP_QTY * (1 + min(0.0, -skew))))
#             buy_qty  = min(buy_qty,  200 - hp_pos)
#             sell_qty = min(sell_qty, 200 + hp_pos)

#             if hp_ask and hp_ask <= our_bid and buy_qty > 0:
#                 hp_orders.append(Order("HYDROGEL_PACK", hp_ask,  buy_qty))
#             if hp_bid and hp_bid >= our_ask and sell_qty > 0:
#                 hp_orders.append(Order("HYDROGEL_PACK", hp_bid, -sell_qty))

#             if buy_qty > 0:
#                 hp_orders.append(Order("HYDROGEL_PACK", our_bid,  buy_qty))
#             if sell_qty > 0:
#                 hp_orders.append(Order("HYDROGEL_PACK", our_ask, -sell_qty))

#         if hp_orders:
#             result["HYDROGEL_PACK"] = hp_orders

#         # ═══════════════════════════════════════════════════════════════════════
#         # 4. OPTIONS — IV residual time-series scalping
#         # ═══════════════════════════════════════════════════════════════════════
#         #
#         # This is what the winners described:
#         #
#         # Each tick:
#         #   A) Compute IV for each strike from its market mid
#         #   B) Fit quadratic smile IV(m) = a*m^2 + b*m + c across all strikes
#         #      to get "fair" IV given moneyness
#         #   C) Compute residual = IV_obs - IV_smile(moneyness)
#         #      This strips out the moneyness effect, leaving pure IV deviation
#         #   D) Store residual in a rolling buffer PER STRIKE over time
#         #   E) Trade when residual deviates from its rolling mean by > Z_THRESH stds
#         #      → residual too high (IV rich) → sell
#         #      → residual too low (IV cheap) → buy
#         #   F) Convert IV signal back to price using BS for sizing intuition,
#         #      but trade at market bid/ask
#         #
#         # Key insight: we're trading the TIME SERIES of residuals, not the
#         # instantaneous cross-sectional deviation. With only 3 strikes the
#         # cross-section has 0 degrees of freedom — but the time series of
#         # each residual has plenty of signal (negative autocorrelation).

#         # Load residual buffers from state
#         res_bufs: Dict[int, List[float]] = {}
#         for K in OPT_STRIKES:
#             res_bufs[K] = list(td.get(f"res_{K}", []))

#         # ── Step A+B: compute IVs and fit smile ──────────────────────────────
#         ivs:      Dict[int, float] = {}
#         bids_map: Dict[int, float] = {}
#         asks_map: Dict[int, float] = {}

#         for K in OPT_STRIKES:
#             depth = state.order_depths.get(f"VEV_{K}")
#             if not depth:
#                 continue
#             b = _best_bid(depth)
#             a = _best_ask(depth)
#             if b is None or a is None:
#                 continue
#             mid = (b + a) / 2.0
#             iv  = _implied_vol(mid, S, K, T, sigma0=sigma_ann)
#             if iv is not None:
#                 ivs[K]      = iv
#                 bids_map[K] = b
#                 asks_map[K] = a

#         # Fit quadratic smile in log-moneyness space
#         # smile_iv[K] = fair IV for strike K given its moneyness
#         smile_iv: Dict[int, float] = {}

#         if len(ivs) >= 3:
#             pts = sorted([(math.log(K / S), ivs[K]) for K in ivs])
#             (m0, y0), (m1, y1), (m2, y2) = pts[0], pts[len(pts)//2], pts[-1]
#             det = (m0 - m1) * (m0 - m2) * (m1 - m2)
#             if abs(det) > 1e-12:
#                 a_c = (y0*(m1-m2) - y1*(m0-m2) + y2*(m0-m1)) / det
#                 b_c = (y0*(m2**2-m1**2) - y1*(m2**2-m0**2) + y2*(m1**2-m0**2)) / det
#                 c_c = (y0*m1*m2*(m1-m2) - y1*m0*m2*(m0-m2) + y2*m0*m1*(m0-m1)) / det
#                 for K in ivs:
#                     m = math.log(K / S)
#                     smile_iv[K] = a_c*m*m + b_c*m + c_c

#         elif len(ivs) == 2:
#             ks = sorted(ivs.keys())
#             m0 = math.log(ks[0]/S); y0 = ivs[ks[0]]
#             m1 = math.log(ks[1]/S); y1 = ivs[ks[1]]
#             dm = (m1 - m0) if abs(m1 - m0) > 1e-12 else 1e-12
#             for K in ivs:
#                 m = math.log(K / S)
#                 smile_iv[K] = y0 + (y1 - y0) * (m - m0) / dm

#         elif len(ivs) == 1:
#             # Only 1 strike — smile IS the IV, residual = 0, no signal
#             K = list(ivs.keys())[0]
#             smile_iv[K] = ivs[K]

#         # ── Step C+D: compute residuals and update buffers ────────────────────
#         residuals: Dict[int, float] = {}
#         for K in ivs:
#             if K in smile_iv:
#                 res = ivs[K] - smile_iv[K]
#                 residuals[K] = res
#                 res_bufs[K].append(res)
#                 if len(res_bufs[K]) > OPT_RES_WINDOW:
#                     res_bufs[K] = res_bufs[K][-OPT_RES_WINDOW:]

#         # ── Step E+F: trade on residual z-score ──────────────────────────────
#         for K in OPT_STRIKES:
#             product = f"VEV_{K}"
#             if K not in residuals:
#                 continue
#             if K not in bids_map or K not in asks_map:
#                 continue

#             buf = res_bufs[K]
#             if len(buf) < OPT_RES_MIN_WINDOW:
#                 continue  # not enough history yet

#             mean_res = sum(buf) / len(buf)
#             var_res  = sum((r - mean_res)**2 for r in buf) / len(buf)
#             std_res  = math.sqrt(var_res) if var_res > 1e-10 else None

#             if std_res is None:
#                 continue

#             z = (residuals[K] - mean_res) / std_res
#             pos = state.position.get(product, 0)
#             b   = bids_map[K]
#             a   = asks_map[K]
#             orders: List[Order] = []

#             # IV residual too high → option overpriced → SELL
#             if z > OPT_Z_THRESH and pos > -OPT_MAX_POS:
#                 qty = min(OPT_QTY, OPT_MAX_POS + pos)
#                 if qty > 0:
#                     orders.append(Order(product, b, -qty))

#             # IV residual too low → option underpriced → BUY
#             elif z < -OPT_Z_THRESH and pos < OPT_MAX_POS:
#                 qty = min(OPT_QTY, OPT_MAX_POS - pos)
#                 if qty > 0:
#                     orders.append(Order(product, a, qty))

#             if orders:
#                 result[product] = orders

#         # ── Persist state ─────────────────────────────────────────────────────
#         trader_data = json.dumps({
#             "ema_vev":     ema_vev,
#             "ema_hp":      ema_hp,
#             "day_min":     day_min,
#             "day_max":     day_max,
#             "current_day": current_day,
#             "returns_buf": returns_buf,
#             "last_vev_px": last_vev_px,
#             **{f"res_{K}": res_bufs[K] for K in OPT_STRIKES},
#         })

#         return result, 0, trader_data
