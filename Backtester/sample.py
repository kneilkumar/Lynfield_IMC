# Add code and refer to readme for instructions on how to run backtester.

from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Tuple, Optional
import json
import math
from collections import deque

# ─── Position limits ───────────────────────────────────────

POSITION_LIMITS: Dict[str, int] = {
    "HYDROGEL_PACK": 200,
    "VELVETFRUIT_EXTRACT": 200,
    **{f"VEV_{k}": 300 for k in [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]},
}

# ─── Parameters ────────────────────────────────────────────

VEV_EMA_ALPHA = 0.02
VEV_PRIMARY_QTY = 30
VEV_SECONDARY_QTY = 15
VEV_EMA_THRESH = 12.0
VEV_MM_SPREAD = 3
VEV_MM_QTY = 2
VEV_NEUTRAL_THRESH = 8.0
VEV_MAX_POS = 200

V4K_QTY = 15
V4K_MAX_POS = 100

HP_EMA_ALPHA = 0.001
HP_HALF_SPREAD = 6
HP_QTY = 2

# ─── Helpers ───────────────────────────────────────────────

def _best_bid(depth: OrderDepth) -> Optional[float]:
    return max(depth.buy_orders) if depth.buy_orders else None

def _best_ask(depth: OrderDepth) -> Optional[float]:
    return min(depth.sell_orders) if depth.sell_orders else None

def _mid(depth: OrderDepth) -> Optional[float]:
    b = _best_bid(depth); a = _best_ask(depth)
    return (b + a) / 2.0 if b and a else None

def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def bs_call(S, K, T, sigma):
    if sigma <= 0 or T <= 0:
        return max(S - K, 0)
    d1 = (math.log(S / K) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm_cdf(d1) - K * norm_cdf(d2)

# ─── Trader ────────────────────────────────────────────────

class Trader:

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:

        try:
            td = json.loads(state.traderData) if state.traderData else {}
        except:
            td = {}

        ema_vev = td.get("ema_vev", 5250.0)
        ema_hp = td.get("ema_hp", 9990.0)
        day_min = td.get("day_min", 1e9)
        day_max = td.get("day_max", -1e9)
        current_day = td.get("day", -1)

        returns_buffer = deque(td.get("returns_buffer", []), maxlen=200)
        last_price = td.get("last_price", None)

        result = {}

        # Day reset
        this_day = state.timestamp // 1_000_000
        if this_day != current_day:
            day_min = 1e9
            day_max = -1e9
            current_day = this_day

        vev_depth = state.order_depths.get("VELVETFRUIT_EXTRACT")
        hp_depth = state.order_depths.get("HYDROGEL_PACK")

        vev_mid = _mid(vev_depth) if vev_depth else None
        hp_mid = _mid(hp_depth) if hp_depth else None

        if vev_mid:
            ema_vev = VEV_EMA_ALPHA * vev_mid + (1 - VEV_EMA_ALPHA) * ema_vev

        if hp_mid:
            ema_hp = HP_EMA_ALPHA * hp_mid + (1 - HP_EMA_ALPHA) * ema_hp

        # ─── Rolling volatility ──────────────────────────────

        if vev_mid and last_price:
            ret = math.log(vev_mid / last_price)
            returns_buffer.append(ret)
        else:
            returns_buffer.append(0)

        last_price = vev_mid

        if len(returns_buffer) > 20:
            mean = sum(returns_buffer) / len(returns_buffer)
            var = sum((r - mean) ** 2 for r in returns_buffer) / len(returns_buffer)
            sigma = math.sqrt(var)
        else:
            sigma = 0.001

        T = 5 / 365

        # ─── VEV STRATEGY ────────────────────────────────────

        vev_pos = state.position.get("VELVETFRUIT_EXTRACT", 0)
        vev_orders = []

        if vev_mid:
            old_min, old_max = day_min, day_max
            day_min = min(day_min, vev_mid)
            day_max = max(day_max, vev_mid)

            new_low = vev_mid < old_min
            new_high = vev_mid > old_max
            dev = vev_mid - ema_vev

            bid = _best_bid(vev_depth)
            ask = _best_ask(vev_depth)

            if new_low and vev_pos < VEV_MAX_POS:
                vev_orders.append(Order("VELVETFRUIT_EXTRACT", ask, VEV_PRIMARY_QTY))

            elif new_high and vev_pos > -VEV_MAX_POS:
                vev_orders.append(Order("VELVETFRUIT_EXTRACT", bid, -VEV_PRIMARY_QTY))

            elif dev > VEV_EMA_THRESH:
                vev_orders.append(Order("VELVETFRUIT_EXTRACT", bid, -VEV_SECONDARY_QTY))

            elif dev < -VEV_EMA_THRESH:
                vev_orders.append(Order("VELVETFRUIT_EXTRACT", ask, VEV_SECONDARY_QTY))

        if vev_orders:
            result["VELVETFRUIT_EXTRACT"] = vev_orders

        # ─── HYDROGEL ────────────────────────────────────────

        hp_orders = []
        hp_pos = state.position.get("HYDROGEL_PACK", 0)

        if hp_mid:
            bid = _best_bid(hp_depth)
            ask = _best_ask(hp_depth)

            fv = round(ema_hp)
            our_bid = fv - HP_HALF_SPREAD
            our_ask = fv + HP_HALF_SPREAD

            if ask and hp_pos < 200:
                hp_orders.append(Order("HYDROGEL_PACK", our_bid, HP_QTY))

            if bid and hp_pos > -200:
                hp_orders.append(Order("HYDROGEL_PACK", our_ask, -HP_QTY))

        if hp_orders:
            result["HYDROGEL_PACK"] = hp_orders

        # ─── IV SCALPING ─────────────────────────────────────

        for strike in [5000, 5100, 5200]:

            product = f"VEV_{strike}"
            depth = state.order_depths.get(product)

            if not depth or not vev_mid:
                continue

            bid = _best_bid(depth)
            ask = _best_ask(depth)

            if not bid or not ask:
                continue

            mid = (bid + ask) / 2
            fair = bs_call(vev_mid, strike, T, sigma)

            edge = mid - fair
            pos = state.position.get(product, 0)
            limit = POSITION_LIMITS[product]

            orders = []

            if edge > 1.5 and pos > -limit:
                orders.append(Order(product, bid, -10))

            elif edge < -1.5 and pos < limit:
                orders.append(Order(product, ask, 10))

            if orders:
                result[product] = orders

        # ─── Save state ──────────────────────────────────────

        trader_data = json.dumps({
            "ema_vev": ema_vev,
            "ema_hp": ema_hp,
            "day_min": day_min,
            "day_max": day_max,
            "day": current_day,
            "returns_buffer": list(returns_buffer),
            "last_price": last_price
        })

        return result, 0, trader_data
