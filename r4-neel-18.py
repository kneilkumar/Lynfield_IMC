from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Tuple, Optional
import json
import math

# ─── Position limits ──────────────────────────────────────────────────────────

POSITION_LIMITS: Dict[str, int] = {
    "HYDROGEL_PACK": 200,
    "VELVETFRUIT_EXTRACT": 200,
    **{f"VEV_{k}": 300 for k in [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]},
}

# ─── Cointegration constants (strategy 1) ────────────────────────────────────
# Johansen vector: EXTRACT + 0.6465*GEL - 11709.13 = 0
# => FV_EXTRACT = 11709.13 - 0.6465 * GEL_mid
BETA      = 0.6465
INTERCEPT = 11709.13

# ─── GEL MM params (strategy 1) ──────────────────────────────────────────────
# Mark 14 quotes ±8 ticks around mid deterministically.
# Mark 38 is a pure aggressor — always hits best available quote, zero markout.
# Strategy: undercut Mark 14 by quoting ±7 to capture all of Mark 38's flow.
# Mark 38 buys and sells are balanced (515 vs 507) — no inventory risk.
GEL_HALF_SPREAD = 7
GEL_QTY         = 5
GEL_MAX_POS     = 200

# ─── EXTRACT Mode 1 — passive MM around FV (strategy 1) ──────────────────────
VEV_MM_HALF = 3
VEV_MM_QTY  = 5
VEV_MAX_POS = 200

# ─── EXTRACT Mode 2 — Mark 67 active (strategy 1) ────────────────────────────
# Mark 67: informed pure buyer, highest markout, never sells.
# When active: pull ask, skew bids up, optionally ride his move.
M67_WINDOW     = 20
M67_ASK_SKEW   = 8
M67_BID_SKEW   = 3
M67_FOLLOW_QTY = 10

# ─── EXTRACT Mode 3 — Falling regime ─────────────────────────────────────────
# When EXTRACT persistently trades below cointegration FV, the dominant trend
# is bearish. Mode 2 fights this by dragging us long at local highs — every
# Mark 67 spike is a sell opportunity in this regime, not a follow signal.
#
# Regime detection: slow EMA of (vev_mid - fv_vev). Negative → falling.
# Mode 3 overrides Mode 2 when bearish_regime is True:
#   - Sell into Mark 67's price spike instead of following him long
#   - Widen bid in Mode 1 so we don't accidentally accumulate longs passively
REGIME_EMA_ALPHA   = 0.003    # slow decay, ~333-tick memory
REGIME_BEAR_THRESH = -8.0     # ema_dev below this → bearish regime active
M3_SELL_SPIKE_QTY  = 15       # contracts to sell into Mark 67's spike
M3_BID_WIDEN       = 4        # extra ticks below FV on bid in bearish regime

# ─── VEV_4000 passive params (strategy 2, unchanged) ─────────────────────────
V4K_PRIMARY_QTY    = 15
V4K_SECONDARY_QTY  = 8
V4K_EMA_THRESH     = 12.0
V4K_MM_SPREAD      = 8
V4K_MM_QTY         = 2
V4K_NEUTRAL_THRESH = 8.0
V4K_MAX_POS        = 100

# ─── Options params (strategy 2, unchanged) ───────────────────────────────────
TTE_DAYS_BASE  = 5
TICKS_PER_DAY  = 1_000_000
OPT_STRIKES    = [5200, 5300, 5400, 5500]
OPT_MAX_POS    = 250
OPT_BASE_SIGMA = 0.30

OPT_PRICE_EDGE: Dict[int, float] = {
    5200: 2.0,
    5300: 1.5,
    5400: 1.0,
    5500: 1.0,
}
OPT_QTY         = 15
OPT_ROLL_WINDOW = 500
OPT_MIN_HISTORY = 20

# ─── Dead voucher params (strategy 2, unchanged) ──────────────────────────────
DEAD_VOUCHERS = ["VEV_6000", "VEV_6500"]
DV_LIMIT      = 300
DV_BUFFER     = int(0.8 * DV_LIMIT)   # 240

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _best_bid(depth: OrderDepth) -> Optional[float]:
    return max(depth.buy_orders) if depth.buy_orders else None

def _best_ask(depth: OrderDepth) -> Optional[float]:
    return min(depth.sell_orders) if depth.sell_orders else None

def _mid(depth: OrderDepth) -> Optional[float]:
    b = _best_bid(depth); a = _best_ask(depth)
    return (b + a) / 2.0 if b and a else None

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _bs_call(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return S * _norm_cdf(d1) - K * _norm_cdf(d2)

def _bs_vega(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return 0.0
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrtT)
    return S * math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi) * sqrtT

def _implied_vol(price: float, S: float, K: float, T: float,
                 sigma0: float = 0.30) -> Optional[float]:
    intrinsic = max(S - K, 0.0)
    if price <= intrinsic + 1e-4:
        return None
    sigma = sigma0
    for _ in range(50):
        p = _bs_call(S, K, T, sigma)
        v = _bs_vega(S, K, T, sigma)
        if v < 1e-8:
            break
        diff  = p - price
        sigma -= diff / v
        sigma  = max(sigma, 1e-6)
        if abs(diff) < 1e-5:
            return sigma
    return sigma if sigma > 0 else None

def _fit_smile_coeffs(moneyness: List[float], ivs: List[float]):
    """
    Fit quadratic IV(m) = a*m^2 + b*m + c via least squares.
    Returns (c, b, a) or None if insufficient data.
    m = K - S (linear moneyness as validated in data analysis).
    """
    n = len(moneyness)
    if n < 3:
        return None
    ms = moneyness
    ys = ivs
    S00 = float(n)
    S10 = sum(ms);               S20 = sum(m**2 for m in ms)
    S30 = sum(m**3 for m in ms); S40 = sum(m**4 for m in ms)
    T0  = sum(ys)
    T1  = sum(ms[i] * ys[i] for i in range(n))
    T2  = sum(ms[i]**2 * ys[i] for i in range(n))
    A   = [[S00, S10, S20], [S10, S20, S30], [S20, S30, S40]]
    rhs = [T0, T1, T2]
    for col in range(3):
        piv = A[col][col]
        if abs(piv) < 1e-12:
            return None
        for row in range(col + 1, 3):
            f = A[row][col] / piv
            for j in range(3):
                A[row][j] -= f * A[col][j]
            rhs[row] -= f * rhs[col]
    coeffs = [0.0, 0.0, 0.0]
    for row in range(2, -1, -1):
        coeffs[row] = rhs[row]
        for j in range(row + 1, 3):
            coeffs[row] -= A[row][j] * coeffs[j]
        if abs(A[row][row]) < 1e-12:
            return None
        coeffs[row] /= A[row][row]
    return tuple(coeffs)   # (c, b, a) → IV = a*m^2 + b*m + c


# ─── Trader ───────────────────────────────────────────────────────────────────

class Trader:
    """
    GEL (HYDROGEL_PACK):
    ────────────────────
    Strategy 1 logic. Mark 14 quotes ±8, Mark 38 is a pure aggressor.
    Quote ±7 to undercut Mark 14 and capture all of Mark 38's flow.
    Inventory skew applied as position approaches limits.

    EXTRACT (VELVETFRUIT_EXTRACT):
    ──────────────────────────────
    Strategy 1 logic. Fair value anchored to cointegration with GEL:
        FV = 11709.13 - 0.6465 * GEL_mid

    Mode 1 (Mark 67 inactive): passive MM ±3 ticks around FV.
    Mode 2 (Mark 67 active):   pull/widen ask, skew bids up, ride his move.

    VEV_4000:
    ─────────
    Strategy 2 logic unchanged. Passive three-mode MR, never cross 20-tick spread.

    OPTIONS (5200-5500):
    ────────────────────
    Strategy 2 logic unchanged. Rolling smile price-deviation scalping.
    Four strikes feed the smile history — quadratic fit is well-conditioned.

    DEAD VOUCHERS (6000/6500):
    ──────────────────────────
    Strategy 2 logic unchanged. Passive MM at 0/1, free edge.
    """

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:

        try:
            td = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            td = {}

        ema_gel         = float(td.get("ema_gel",         9990.0))
        ticks_since_m67 = int(td.get("ticks_since_m67",   M67_WINDOW + 1))
        day_min         = float(td.get("day_min",          1e9))
        day_max         = float(td.get("day_max",         -1e9))
        current_day     = int(td.get("current_day",       -1))
        smile_hist: List[List[float]] = td.get("smile_hist", [])
        ema_dev         = float(td.get("ema_dev", 0.0))

        result: Dict[str, List[Order]] = {}

        # ── Day rollover ──────────────────────────────────────────────────────
        this_day = state.timestamp // TICKS_PER_DAY
        if this_day != current_day:
            day_min     = 1e9
            day_max     = -1e9
            current_day = this_day

        # ── TTE ───────────────────────────────────────────────────────────────
        frac_elapsed = (state.timestamp % TICKS_PER_DAY) / TICKS_PER_DAY
        T = max((TTE_DAYS_BASE - frac_elapsed) / 365.0, 1e-6)

        # ── Market data ───────────────────────────────────────────────────────
        gel_depth = state.order_depths.get("HYDROGEL_PACK")
        vev_depth = state.order_depths.get("VELVETFRUIT_EXTRACT")

        gel_mid = _mid(gel_depth) if gel_depth else None
        vev_mid = _mid(vev_depth) if vev_depth else None

        # ── Cointegration FV for EXTRACT (only when gel_mid available) ────────
        fv_vev = (INTERCEPT - BETA * gel_mid) if gel_mid is not None else None

        # ── Regime detection (Mode 3) ─────────────────────────────────────────
        # Slow EMA of deviation from cointegration FV.
        # Persistently negative → falling regime → Mode 3 overrides Mode 2.
        if fv_vev is not None and vev_mid is not None:
            dev_from_fv = vev_mid - fv_vev
            ema_dev = REGIME_EMA_ALPHA * dev_from_fv + (1 - REGIME_EMA_ALPHA) * ema_dev
        bearish_regime = ema_dev < REGIME_BEAR_THRESH

        # ── Detect Mark 67 activity ───────────────────────────────────────────
        m67_active_this_tick = False
        for trade in state.market_trades.get("VELVETFRUIT_EXTRACT", []):
            if hasattr(trade, 'buyer') and trade.buyer == "Mark 67":
                m67_active_this_tick = True
                break

        if m67_active_this_tick:
            ticks_since_m67 = 0
        else:
            ticks_since_m67 += 1

        in_mode2 = ticks_since_m67 <= M67_WINDOW and not bearish_regime

        # ── Daily high/low tracking + EMA dev (used by VEV_4000) ─────────────
        new_daily_low  = False
        new_daily_high = False
        dev            = 0.0
        VEV_EMA_ALPHA  = 0.02
        ema_vev        = float(td.get("ema_vev", 5250.0))

        if vev_mid is not None:
            old_min = day_min; old_max = day_max
            day_min = min(day_min, vev_mid)
            day_max = max(day_max, vev_mid)
            new_daily_low  = vev_mid < old_min
            new_daily_high = vev_mid > old_max
            ema_vev = VEV_EMA_ALPHA * vev_mid + (1 - VEV_EMA_ALPHA) * ema_vev
            dev = vev_mid - ema_vev

        # ── Positions ─────────────────────────────────────────────────────────
        gel_pos = state.position.get("HYDROGEL_PACK",       0)
        vev_pos = state.position.get("VELVETFRUIT_EXTRACT", 0)

        # ═══════════════════════════════════════════════════════════════════════
        # 1. GEL — strategy 1: undercut Mark 14, farm Mark 38
        # ═══════════════════════════════════════════════════════════════════════
        gel_orders: List[Order] = []

        if gel_depth and gel_mid is not None:
            our_gel_bid = round(gel_mid) - GEL_HALF_SPREAD
            our_gel_ask = round(gel_mid) + GEL_HALF_SPREAD

            gel_skew     = gel_pos / GEL_MAX_POS
            gel_buy_qty  = max(1, round(GEL_QTY * (1 - max(0.0,  gel_skew))))
            gel_sell_qty = max(1, round(GEL_QTY * (1 + min(0.0, -gel_skew))))
            gel_buy_qty  = min(gel_buy_qty,  GEL_MAX_POS - gel_pos)
            gel_sell_qty = min(gel_sell_qty, GEL_MAX_POS + gel_pos)

            gel_best_ask = _best_ask(gel_depth)
            gel_best_bid = _best_bid(gel_depth)
            if gel_best_ask and gel_best_ask <= our_gel_bid and gel_buy_qty > 0:
                gel_orders.append(Order("HYDROGEL_PACK", gel_best_ask,  gel_buy_qty))
            if gel_best_bid and gel_best_bid >= our_gel_ask and gel_sell_qty > 0:
                gel_orders.append(Order("HYDROGEL_PACK", gel_best_bid, -gel_sell_qty))

            if gel_buy_qty > 0:
                gel_orders.append(Order("HYDROGEL_PACK", our_gel_bid,  gel_buy_qty))
            if gel_sell_qty > 0:
                gel_orders.append(Order("HYDROGEL_PACK", our_gel_ask, -gel_sell_qty))

        if gel_orders:
            result["HYDROGEL_PACK"] = gel_orders

        # ═══════════════════════════════════════════════════════════════════════
        # 2. EXTRACT — strategy 1: cointegration FV + Mark 67 Mode 1/2
        # ═══════════════════════════════════════════════════════════════════════
        vev_orders: List[Order] = []

        if vev_depth and fv_vev is not None:
            vev_bid = _best_bid(vev_depth)
            vev_ask = _best_ask(vev_depth)

            skew     = vev_pos / VEV_MAX_POS
            buy_qty  = max(1, round(VEV_MM_QTY * (1 - max(0.0,  skew))))
            sell_qty = max(1, round(VEV_MM_QTY * (1 + min(0.0, -skew))))
            buy_qty  = min(buy_qty,  VEV_MAX_POS - vev_pos)
            sell_qty = min(sell_qty, VEV_MAX_POS + vev_pos)

            if not in_mode2:
                # ── Mode 1: passive MM around cointegration FV ───────────────
                # In bearish regime: widen bid by M3_BID_WIDEN so we don't
                # passively accumulate longs against the dominant trend.
                bid_offset = VEV_MM_HALF + (M3_BID_WIDEN if bearish_regime else 0)
                our_bid = round(fv_vev) - bid_offset
                our_ask = round(fv_vev) + VEV_MM_HALF

                if vev_ask and vev_ask <= our_bid and buy_qty > 0:
                    vev_orders.append(Order("VELVETFRUIT_EXTRACT", vev_ask,  buy_qty))
                if vev_bid and vev_bid >= our_ask and sell_qty > 0:
                    vev_orders.append(Order("VELVETFRUIT_EXTRACT", vev_bid, -sell_qty))

                if buy_qty > 0:
                    vev_orders.append(Order("VELVETFRUIT_EXTRACT", our_bid,  buy_qty))
                if sell_qty > 0:
                    vev_orders.append(Order("VELVETFRUIT_EXTRACT", our_ask, -sell_qty))

            elif bearish_regime:
                # ── Mode 3: Mark 67 fires in a falling regime ─────────────────
                # He's buying at a local high in a downtrend — sell into his
                # spike rather than following him long.
                # Keep ask tight (FV + half_spread) to stay/get shorter.
                # Widen bid way down so we don't accidentally absorb his flow.
                our_bid = round(fv_vev) - VEV_MM_HALF - M3_BID_WIDEN
                our_ask = round(fv_vev) + VEV_MM_HALF

                spike_sell_qty = min(M3_SELL_SPIKE_QTY, VEV_MAX_POS + vev_pos)
                if spike_sell_qty > 0 and vev_bid:
                    vev_orders.append(Order("VELVETFRUIT_EXTRACT", vev_bid, -spike_sell_qty))

                if buy_qty > 0:
                    vev_orders.append(Order("VELVETFRUIT_EXTRACT", our_bid,  buy_qty))
                if sell_qty > 0:
                    vev_orders.append(Order("VELVETFRUIT_EXTRACT", our_ask, -sell_qty))

            else:
                # ── Mode 2: Mark 67 active, no dominant trend — follow him ────
                our_bid = round(fv_vev) + M67_BID_SKEW
                our_ask = round(fv_vev) + M67_ASK_SKEW

                follow_qty = min(M67_FOLLOW_QTY, VEV_MAX_POS - vev_pos)
                if follow_qty > 0 and vev_ask:
                    vev_orders.append(Order("VELVETFRUIT_EXTRACT", vev_ask, follow_qty))

                if buy_qty > 0:
                    vev_orders.append(Order("VELVETFRUIT_EXTRACT", our_bid,  buy_qty))
                if sell_qty > 0:
                    vev_orders.append(Order("VELVETFRUIT_EXTRACT", our_ask, -sell_qty))

        if vev_orders:
            result["VELVETFRUIT_EXTRACT"] = vev_orders

        # ═══════════════════════════════════════════════════════════════════════
        # 3. VEV_4000 — strategy 2 unchanged: passive three-mode MR
        # ═══════════════════════════════════════════════════════════════════════
        v4k_pos   = state.position.get("VEV_4000", 0)
        v4k_depth = state.order_depths.get("VEV_4000")
        v4k_orders: List[Order] = []

        if v4k_depth and vev_mid is not None:
            v4k_bid = _best_bid(v4k_depth)
            v4k_ask = _best_ask(v4k_depth)
            if v4k_bid is not None and v4k_ask is not None:
                v4k_mid      = (v4k_bid + v4k_ask) / 2.0
                passive_buy  = int(v4k_bid + 1)
                passive_sell = int(v4k_ask - 1)

                if new_daily_low and v4k_pos < V4K_MAX_POS:
                    qty = min(V4K_PRIMARY_QTY, V4K_MAX_POS - v4k_pos)
                    if qty > 0:
                        v4k_orders.append(Order("VEV_4000", passive_buy, qty))
                elif new_daily_high and v4k_pos > -V4K_MAX_POS:
                    qty = min(V4K_PRIMARY_QTY, V4K_MAX_POS + v4k_pos)
                    if qty > 0:
                        v4k_orders.append(Order("VEV_4000", passive_sell, -qty))
                elif dev > V4K_EMA_THRESH and v4k_pos > -V4K_MAX_POS:
                    qty = min(V4K_SECONDARY_QTY, V4K_MAX_POS + v4k_pos)
                    if qty > 0:
                        v4k_orders.append(Order("VEV_4000", passive_sell, -qty))
                elif dev < -V4K_EMA_THRESH and v4k_pos < V4K_MAX_POS:
                    qty = min(V4K_SECONDARY_QTY, V4K_MAX_POS - v4k_pos)
                    if qty > 0:
                        v4k_orders.append(Order("VEV_4000", passive_buy, qty))
                elif abs(dev) < V4K_NEUTRAL_THRESH:
                    our_bid = int(v4k_mid) - V4K_MM_SPREAD
                    our_ask = int(v4k_mid) + V4K_MM_SPREAD
                    if v4k_pos < V4K_MAX_POS:
                        qty = min(V4K_MM_QTY, V4K_MAX_POS - v4k_pos)
                        if qty > 0:
                            v4k_orders.append(Order("VEV_4000", our_bid, qty))
                    if v4k_pos > -V4K_MAX_POS:
                        qty = min(V4K_MM_QTY, V4K_MAX_POS + v4k_pos)
                        if qty > 0:
                            v4k_orders.append(Order("VEV_4000", our_ask, -qty))

        if v4k_orders:
            result["VEV_4000"] = v4k_orders

        # ═══════════════════════════════════════════════════════════════════════
        # 4. OPTIONS — strategy 2 unchanged: rolling smile scalping (5200-5500)
        #
        # Four strikes feed the smile history — quadratic fit is well-conditioned
        # (contrast with single-strike version which always returned None).
        # ═══════════════════════════════════════════════════════════════════════
        S = vev_mid if vev_mid is not None else ema_vev

        opt_ivs:  Dict[int, float] = {}
        opt_mids: Dict[int, float] = {}
        opt_bids: Dict[int, float] = {}
        opt_asks: Dict[int, float] = {}

        for K in OPT_STRIKES:
            depth = state.order_depths.get(f"VEV_{K}")
            if not depth:
                continue
            b = _best_bid(depth)
            a = _best_ask(depth)
            if b is None or a is None:
                continue
            mid_px = (b + a) / 2.0
            iv     = _implied_vol(mid_px, S, K, T, sigma0=OPT_BASE_SIGMA)
            if iv is not None:
                opt_ivs[K]  = iv
                opt_mids[K] = mid_px
                opt_bids[K] = b
                opt_asks[K] = a

        for K, iv in opt_ivs.items():
            smile_hist.append([float(K - S), iv])

        if len(smile_hist) > OPT_ROLL_WINDOW:
            smile_hist = smile_hist[-OPT_ROLL_WINDOW:]

        smile_coeffs = None
        if len(smile_hist) >= OPT_MIN_HISTORY:
            ms_list  = [pt[0] for pt in smile_hist]
            ivs_list = [pt[1] for pt in smile_hist]
            smile_coeffs = _fit_smile_coeffs(ms_list, ivs_list)

        if smile_coeffs is not None:
            c_coef, b_coef, a_coef = smile_coeffs

            for K in OPT_STRIKES:
                if K not in opt_ivs:
                    continue

                m         = float(K - S)
                smile_iv  = a_coef * m * m + b_coef * m + c_coef
                fair_px   = _bs_call(S, K, T, max(smile_iv, 1e-6))
                price_dev = opt_mids[K] - fair_px
                threshold = OPT_PRICE_EDGE.get(K, 2.0)

                # 5200/5400/5500 are smile anchors only — do not trade them
                if K != 5300:
                    continue

                pos     = state.position.get(f"VEV_{K}", 0)
                b_price = opt_bids[K]
                a_price = opt_asks[K]
                orders: List[Order] = []

                if price_dev > threshold and pos > -OPT_MAX_POS:
                    qty = min(OPT_QTY, OPT_MAX_POS + pos)
                    if qty > 0:
                        orders.append(Order(f"VEV_{K}", b_price, -qty))

                elif price_dev < -threshold and pos < OPT_MAX_POS:
                    qty = min(OPT_QTY, OPT_MAX_POS - pos)
                    if qty > 0:
                        orders.append(Order(f"VEV_{K}", a_price, qty))

                if orders:
                    result[f"VEV_{K}"] = orders

        # ═══════════════════════════════════════════════════════════════════════
        # 5. DEAD VOUCHERS — strategy 2 unchanged: passive MM on 6000/6500
        # ═══════════════════════════════════════════════════════════════════════
        for product in DEAD_VOUCHERS:
            depth = state.order_depths.get(product)
            if not depth:
                continue

            pos    = state.position.get(product, 0)
            orders: List[Order] = []

            for ask_price, ask_qty in depth.sell_orders.items():
                if ask_price == 0:
                    qty = min(-ask_qty, DV_LIMIT - pos)
                    if qty > 0:
                        orders.append(Order(product, 0, qty))
                        pos += qty

            for bid_price, bid_qty in depth.buy_orders.items():
                if bid_price == 1:
                    qty = min(bid_qty, DV_LIMIT + pos)
                    if qty > 0:
                        orders.append(Order(product, 1, -qty))
                        pos -= qty

            if pos < DV_BUFFER:
                qty = DV_LIMIT - pos
                if qty > 0:
                    orders.append(Order(product, 0, qty))

            if pos > -DV_BUFFER:
                qty = DV_LIMIT + pos
                if qty > 0:
                    orders.append(Order(product, 1, -qty))

            if orders:
                result[product] = orders

        # ── Persist ───────────────────────────────────────────────────────────
        trader_data = json.dumps({
            "ema_gel":         ema_gel,
            "ticks_since_m67": ticks_since_m67,
            "ema_vev":         ema_vev,
            "day_min":         day_min,
            "day_max":         day_max,
            "current_day":     current_day,
            "smile_hist":      smile_hist,
            "ema_dev":         ema_dev,
        })

        return result, 0, trader_data