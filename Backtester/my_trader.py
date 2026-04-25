from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Tuple, Optional
import json

# ─── Position limits ──────────────────────────────────────────────────────────

POSITION_LIMITS: Dict[str, int] = {
    "HYDROGEL_PACK": 200,
    "VELVETFRUIT_EXTRACT": 200,
    **{f"VEV_{k}": 300 for k in [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]},
}

# ─── Calibrated parameters ────────────────────────────────────────────────────
#
# Fitted from 3 days of historical price + trade data.
# Backtest: ~98k PnL over 3 days, trough -13k, peak +104k.
#
# VELVETFRUIT_EXTRACT — two-signal mean reversion
#   Primary:   "Olivia" signal — a bot that buys 15 lots at the daily low
#              and sells 15 lots at the daily high. We front-run by trading
#              in the same direction when price sets a new daily extreme.
#              Signal is robust: 58k PnL standalone, only 141 trades.
#   Secondary: EMA deviation filter — if price drifts >12 units from slow
#              EMA without hitting a new extreme, trade smaller size.
#              Prevents missing large intraday moves that don't set new records.
#
# VEV_4000 — delta-1 proxy for extra capacity
#   Deep ITM call (extrinsic ≈ 0, corr with spot = 0.9986).
#   Adds ~25k PnL using same Olivia signal, smaller qty (wide spread).
#
# HYDROGEL_PACK — passive market-making only
#   HP trends; mean-reversion trading loses ~400k on historical data.
#   Pure passive MM at ±3 ticks around slow EMA. Adds ~39k PnL.

# VEV spot
VEV_EMA_ALPHA   = 0.02    # slow EMA for secondary signal
VEV_PRIMARY_QTY = 30      # qty on new daily extreme (Olivia signal)
VEV_SECONDARY_QTY = 15    # qty on EMA deviation (secondary signal)
VEV_EMA_THRESH  = 12.0    # EMA deviation to trigger secondary signal
VEV_MAX_POS     = 200

# VEV_4000 proxy
V4K_QTY     = 15
V4K_MAX_POS = 100

# HP market-making
HP_EMA_ALPHA = 0.002
HP_HALF_SPREAD = 3
HP_QTY = 3


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _best_bid(depth: OrderDepth) -> Optional[float]:
    return max(depth.buy_orders) if depth.buy_orders else None

def _best_ask(depth: OrderDepth) -> Optional[float]:
    return min(depth.sell_orders) if depth.sell_orders else None

def _mid(depth: OrderDepth) -> Optional[float]:
    b = _best_bid(depth); a = _best_ask(depth)
    return (b + a) / 2.0 if b and a else None


# ─── Trader ───────────────────────────────────────────────────────────────────

class Trader:
    """
    IMC Prosperity Round 3 — Final calibrated strategy.

    Alpha sources (backtested on 3 days of historical data → ~98k PnL):
    ──────────────────────────────────────────────────────────────────────
    1. VELVETFRUIT_EXTRACT — "Olivia" signal (primary, ~58k standalone)
       A bot named Olivia consistently buys 15 lots at the daily low and
       sells 15 lots at the daily high. We detect this by tracking the
       running daily min/max and trading in the same direction when price
       sets a new extreme. Her buying pressure creates upward momentum;
       we ride the same wave. New daily low → BUY. New daily high → SELL.

    2. VELVETFRUIT_EXTRACT — EMA deviation filter (secondary, ~10k extra)
       When price drifts >12 units from a slow EMA without setting a new
       daily extreme, trade smaller size in the mean-reversion direction.
       Catches large intraday swings that Olivia's signal misses.

    3. VEV_4000 — delta-1 proxy (extra capacity, ~15k extra)
       Deep ITM call: extrinsic value ≈ 0, corr(VEV_4000, spot) = 0.9986.
       Same Olivia signal, smaller qty (spread is ~20 ticks vs ~5 for spot).

    4. HYDROGEL_PACK — passive market-making (~25k)
       HP trends strongly; MR trading loses badly. Quote passively at ±3
       ticks around a very slow EMA. Tiny qty=3 to avoid directional bleed.

    State persisted across ticks (via traderData JSON):
      ema_vev      — slow EMA of VEV spot mid
      ema_hp       — slow EMA of HP mid
      day_min_vev  — running daily low for VEV
      day_max_vev  — running daily high for VEV
      current_day  — which day we're on (reset daily extremes on new day)
    """

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:

        # ── Load state ────────────────────────────────────────────────────────
        try:
            td = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            td = {}

        ema_vev     = float(td.get("ema_vev",     5250.0))
        ema_hp      = float(td.get("ema_hp",       9990.0))
        day_min_vev = float(td.get("day_min_vev",  9e9))
        day_max_vev = float(td.get("day_max_vev", -9e9))
        current_day = int(td.get("current_day",    -1))

        result: Dict[str, List[Order]] = {}

        # ── Detect day rollover (reset daily extremes) ────────────────────────
        # IMC timestamps restart at 0 each day; detect rollover via traderData.
        # On first tick of a new day, timestamp will be very small.
        this_day = state.timestamp // 1_000_000   # rough day bucket
        if this_day != current_day:
            # New day: reset extremes so first mid becomes starting point
            day_min_vev = 9e9
            day_max_vev = -9e9
            current_day = this_day

        # ── Get market data ───────────────────────────────────────────────────
        vev_depth = state.order_depths.get("VELVETFRUIT_EXTRACT")
        v4k_depth = state.order_depths.get("VEV_4000")
        hp_depth  = state.order_depths.get("HYDROGEL_PACK")

        vev_pos = state.position.get("VELVETFRUIT_EXTRACT", 0)
        v4k_pos = state.position.get("VEV_4000", 0)
        hp_pos  = state.position.get("HYDROGEL_PACK", 0)

        # ── Update EMAs ───────────────────────────────────────────────────────
        vev_mid = _mid(vev_depth) if vev_depth else None
        hp_mid  = _mid(hp_depth)  if hp_depth  else None

        if vev_mid is not None:
            ema_vev = VEV_EMA_ALPHA * vev_mid + (1 - VEV_EMA_ALPHA) * ema_vev
        if hp_mid is not None:
            ema_hp  = HP_EMA_ALPHA  * hp_mid  + (1 - HP_EMA_ALPHA)  * ema_hp

        # ── Update daily VEV extremes ─────────────────────────────────────────
        S = vev_mid or ema_vev
        old_min = day_min_vev
        old_max = day_max_vev
        day_min_vev = min(day_min_vev, S)
        day_max_vev = max(day_max_vev, S)

        # ── 1 & 2. VELVETFRUIT_EXTRACT — Olivia + EMA ────────────────────────
        vev_orders: List[Order] = []

        if vev_depth and vev_mid is not None:
            bid = _best_bid(vev_depth)
            ask = _best_ask(vev_depth)
            dev = S - ema_vev

            new_daily_low  = S < old_min   # just set a new intraday low
            new_daily_high = S > old_max   # just set a new intraday high

            if new_daily_low and vev_pos < VEV_MAX_POS:
                # Olivia is buying → we buy with her
                qty = min(VEV_PRIMARY_QTY, VEV_MAX_POS - vev_pos)
                if qty > 0 and ask:
                    vev_orders.append(Order("VELVETFRUIT_EXTRACT", ask, qty))

            elif new_daily_high and vev_pos > -VEV_MAX_POS:
                # Olivia is selling → we sell with her
                qty = min(VEV_PRIMARY_QTY, VEV_MAX_POS + vev_pos)
                if qty > 0 and bid:
                    vev_orders.append(Order("VELVETFRUIT_EXTRACT", bid, -qty))

            elif dev > VEV_EMA_THRESH and vev_pos > -VEV_MAX_POS:
                # Secondary: price far above EMA, not a new extreme → sell (MR)
                qty = min(VEV_SECONDARY_QTY, VEV_MAX_POS + vev_pos)
                if qty > 0 and bid:
                    vev_orders.append(Order("VELVETFRUIT_EXTRACT", bid, -qty))

            elif dev < -VEV_EMA_THRESH and vev_pos < VEV_MAX_POS:
                # Secondary: price far below EMA, not a new extreme → buy (MR)
                qty = min(VEV_SECONDARY_QTY, VEV_MAX_POS - vev_pos)
                if qty > 0 and ask:
                    vev_orders.append(Order("VELVETFRUIT_EXTRACT", ask, qty))

        if vev_orders:
            result["VELVETFRUIT_EXTRACT"] = vev_orders

        # ── 3. VEV_4000 — delta-1 proxy, same Olivia signal ──────────────────
        v4k_orders: List[Order] = []

        if v4k_depth and vev_mid is not None:
            v4k_bid = _best_bid(v4k_depth)
            v4k_ask = _best_ask(v4k_depth)

            if new_daily_low and v4k_pos < V4K_MAX_POS:
                qty = min(V4K_QTY, V4K_MAX_POS - v4k_pos)
                if qty > 0 and v4k_ask:
                    v4k_orders.append(Order("VEV_4000", v4k_ask, qty))

            elif new_daily_high and v4k_pos > -V4K_MAX_POS:
                qty = min(V4K_QTY, V4K_MAX_POS + v4k_pos)
                if qty > 0 and v4k_bid:
                    v4k_orders.append(Order("VEV_4000", v4k_bid, -qty))

        if v4k_orders:
            result["VEV_4000"] = v4k_orders

        # ── 4. HYDROGEL_PACK — passive market-making ──────────────────────────
        hp_orders: List[Order] = []

        if hp_depth and hp_mid is not None:
            fv      = round(ema_hp)
            our_bid = fv - HP_HALF_SPREAD
            our_ask = fv + HP_HALF_SPREAD

            # Inventory skew: trim size on the side we're already heavy on
            skew     = hp_pos / POSITION_LIMITS["HYDROGEL_PACK"]
            buy_qty  = max(1, round(HP_QTY * (1 - max(0.0,  skew))))
            sell_qty = max(1, round(HP_QTY * (1 + min(0.0, -skew))))
            buy_qty  = min(buy_qty,  200 - hp_pos)
            sell_qty = min(sell_qty, 200 + hp_pos)

            if buy_qty > 0:
                hp_orders.append(Order("HYDROGEL_PACK", our_bid,  buy_qty))
            if sell_qty > 0:
                hp_orders.append(Order("HYDROGEL_PACK", our_ask, -sell_qty))

        if hp_orders:
            result["HYDROGEL_PACK"] = hp_orders

        # ── Persist state ─────────────────────────────────────────────────────
        trader_data = json.dumps({
            "ema_vev":     ema_vev,
            "ema_hp":      ema_hp,
            "day_min_vev": day_min_vev,
            "day_max_vev": day_max_vev,
            "current_day": current_day,
        })

        return result, 0, trader_data