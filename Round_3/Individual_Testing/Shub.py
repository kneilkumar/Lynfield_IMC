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
# All params fitted from 3 days of historical data.
# Backtest result: ~121k PnL over 3 days, trough -14k, peak +127k.
#
# VELVETFRUIT_EXTRACT (three-mode strategy):
#   Mode 1 — New daily extreme signal (~58k standalone)
#     Price making new intraday low → BUY  (mean reversion from extreme)
#     Price making new intraday high → SELL
#     Qty=30, no minimum threshold needed — any new extreme is a signal.
#
#   Mode 2 — EMA deviation secondary signal
#     When price deviates >12 from slow EMA but hasn't set a new extreme:
#     smaller trade in mean-reversion direction. Catches large intraday moves.
#     Qty=15.
#
#   Mode 3 — Passive market-making in neutral regime
#     When |dev| < 8 (price near fair value): quote ±3 ticks around EMA.
#     Captures bid-ask spread. Qty=2. Adds ~30k standalone.
#
# HYDROGEL_PACK:
#   HP has a WIDE spread (15.7 ticks) and trends strongly.
#   MR trading loses ~400k. Passive MM at spread=6, qty=2, very slow EMA.
#   Key insight: spread=6 << market spread=15.7 → we get filled when market
#   swings through our price, then mean reverts. Adds ~66k.
#   DO NOT do directional MR on HP.
#
# VEV_4000 (deep ITM call, delta≈1):
#   Extrinsic value≈0, corr(price, spot)=0.9986. Pure delta-1 proxy.
#   Same new-daily-extreme signal as spot. Qty=15, max pos ±100.
#   Wide spread (20 ticks) so conservative sizing.

# VEV spot
VEV_EMA_ALPHA       = 0.02
VEV_PRIMARY_QTY     = 30      # new daily extreme signal
VEV_SECONDARY_QTY   = 15      # EMA deviation signal
VEV_EMA_THRESH      = 12.0    # EMA deviation threshold for secondary signal
VEV_MM_SPREAD       = 3       # half-spread for neutral-regime passive MM
VEV_MM_QTY         = 2       # qty for neutral-regime passive MM
VEV_NEUTRAL_THRESH  = 8.0     # abs(dev) below which we're in neutral regime
VEV_MAX_POS         = 200

# VEV_4000 proxy
V4K_QTY             = 15
V4K_MAX_POS         = 100

# HP market-making
HP_EMA_ALPHA        = 0.001   # very slow — HP trends, EMA shouldn't chase it
HP_HALF_SPREAD      = 6       # wider spread = only fill on real market swings
HP_QTY              = 2       # tiny qty — don't build inventory on trends


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
    IMC Prosperity Round 3 — Data-calibrated strategy v3.

    Backtest on 3 days of historical data: ~121k PnL, trough -14k.

    Findings from data analysis:
    ────────────────────────────
    - buyer/seller are NaN in round 3 — no trader IDs visible
    - VEV_6000/6500 always trade at price=0, always same qty → worthless OTM, skip
    - VEV_5400/5500 trade in sync 95% of time → one bot trading basket, no edge for us
    - HP and VEV both have -0.13/-0.16 lag-1 autocorr BUT HP spread=15.7 vs VEV=5.0
      → HP spread/move ratio = 7.25 (too wide for MR), VEV = 4.41 (workable)
    - HP: passive MM at spread=6 captures 66k. MR on HP loses 400k.
    - VEV in neutral regime (near EMA): passive MM adds 30k on top of MR
    - New daily extreme signal: forward returns are 2-4x stronger than EMA signal
      at short horizons → use as primary signal, EMA as secondary

    Products deliberately skipped:
    - VEV_4500, 5000-6500: no exploitable edge found in data
    - Cross-product predictability (HP→VEV, VEV→HP): all correlations <0.01

    State persisted in traderData:
      ema_vev, ema_hp  — slow EMAs
      day_min_vev      — running daily low
      day_max_vev      — running daily high
      current_day      — day tracker for resetting extremes
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

        # ── Day rollover detection — reset daily extremes ─────────────────────
        # Timestamps restart at 0 each day. Use a large bucket to detect rollover.
        this_day = state.timestamp // 1_000_000
        if this_day != current_day:
            day_min_vev = 9e9
            day_max_vev = -9e9
            current_day = this_day

        # ── Get depths and positions ──────────────────────────────────────────
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

        S = vev_mid or ema_vev

        # ── Update daily VEV extremes ─────────────────────────────────────────
        old_min     = day_min_vev
        old_max     = day_max_vev
        day_min_vev = min(day_min_vev, S)
        day_max_vev = max(day_max_vev, S)

        new_daily_low  = S < old_min
        new_daily_high = S > old_max
        dev = S - ema_vev

        # ═══════════════════════════════════════════════════════════════════════
        # 1. VELVETFRUIT_EXTRACT — Three-mode strategy
        # ═══════════════════════════════════════════════════════════════════════
        vev_orders: List[Order] = []

        if vev_depth and vev_mid is not None:
            bid = _best_bid(vev_depth)
            ask = _best_ask(vev_depth)

            # ── Mode 1: New daily extreme (primary signal) ────────────────────
            if new_daily_low and vev_pos < VEV_MAX_POS:
                qty = min(VEV_PRIMARY_QTY, VEV_MAX_POS - vev_pos)
                if qty > 0 and ask:
                    vev_orders.append(Order("VELVETFRUIT_EXTRACT", ask, qty))

            elif new_daily_high and vev_pos > -VEV_MAX_POS:
                qty = min(VEV_PRIMARY_QTY, VEV_MAX_POS + vev_pos)
                if qty > 0 and bid:
                    vev_orders.append(Order("VELVETFRUIT_EXTRACT", bid, -qty))

            # ── Mode 2: EMA deviation (secondary signal) ──────────────────────
            elif dev > VEV_EMA_THRESH and vev_pos > -VEV_MAX_POS:
                qty = min(VEV_SECONDARY_QTY, VEV_MAX_POS + vev_pos)
                if qty > 0 and bid:
                    vev_orders.append(Order("VELVETFRUIT_EXTRACT", bid, -qty))

            elif dev < -VEV_EMA_THRESH and vev_pos < VEV_MAX_POS:
                qty = min(VEV_SECONDARY_QTY, VEV_MAX_POS - vev_pos)
                if qty > 0 and ask:
                    vev_orders.append(Order("VELVETFRUIT_EXTRACT", ask, qty))

            # ── Mode 3: Neutral regime — passive market-making ────────────────
            elif abs(dev) < VEV_NEUTRAL_THRESH:
                fv      = round(ema_vev)
                our_bid = fv - VEV_MM_SPREAD
                our_ask = fv + VEV_MM_SPREAD

                if ask and ask <= our_bid and vev_pos < VEV_MAX_POS:
                    qty = min(VEV_MM_QTY, VEV_MAX_POS - vev_pos)
                    if qty > 0:
                        vev_orders.append(Order("VELVETFRUIT_EXTRACT", ask, qty))

                if bid and bid >= our_ask and vev_pos > -VEV_MAX_POS:
                    qty = min(VEV_MM_QTY, VEV_MAX_POS + vev_pos)
                    if qty > 0:
                        vev_orders.append(Order("VELVETFRUIT_EXTRACT", bid, -qty))

        if vev_orders:
            result["VELVETFRUIT_EXTRACT"] = vev_orders

        # ═══════════════════════════════════════════════════════════════════════
        # 2. VEV_4000 — Delta-1 proxy, same extreme signal
        # ═══════════════════════════════════════════════════════════════════════
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

        # ═══════════════════════════════════════════════════════════════════════
        # 3. HYDROGEL_PACK — Passive market-making only
        # ═══════════════════════════════════════════════════════════════════════
        hp_orders: List[Order] = []

        if hp_depth and hp_mid is not None:
            hp_bid = _best_bid(hp_depth)
            hp_ask = _best_ask(hp_depth)
            fv      = round(ema_hp)
            our_bid = fv - HP_HALF_SPREAD
            our_ask = fv + HP_HALF_SPREAD

            # Inventory skew: reduce size when already positioned
            skew     = hp_pos / POSITION_LIMITS["HYDROGEL_PACK"]
            buy_qty  = max(1, round(HP_QTY * (1 - max(0.0,  skew))))
            sell_qty = max(1, round(HP_QTY * (1 + min(0.0, -skew))))
            buy_qty  = min(buy_qty,  200 - hp_pos)
            sell_qty = min(sell_qty, 200 + hp_pos)

            # Fill when market crosses our price
            if hp_ask and hp_ask <= our_bid and buy_qty > 0:
                hp_orders.append(Order("HYDROGEL_PACK", hp_ask,  buy_qty))
            if hp_bid and hp_bid >= our_ask and sell_qty > 0:
                hp_orders.append(Order("HYDROGEL_PACK", hp_bid, -sell_qty))

            # Also post passive quotes for when market comes to us
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