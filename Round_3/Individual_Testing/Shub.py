from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Tuple
import math
import json

# ─── Constants ───────────────────────────────────────────────────────────────

POSITION_LIMITS: Dict[str, int] = {
    "HYDROGEL_PACK": 200,
    "VELVETFRUIT_EXTRACT": 200,
    **{f"VEV_{k}": 300 for k in [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]},
}

# Fitted Black-Scholes sigma (price-units / sqrt(day)).
# Derived by minimising squared error across ~1800 historical snapshots
# spanning days 0-2 (TTE=8,7,6). Errors under ±3 units across all strikes.
BS_SIGMA = 64.0

# Round 3 starts at TTE = 5 days (per competition rules).
TTE = 5.0

# EMA smoothing factors (per tick; ~10,000 ticks per day)
HP_ALPHA  = 0.002
VEV_ALPHA = 0.005

# Market-making half-spreads (in price ticks)
HP_HALF_SPREAD  = 4
VEV_HALF_SPREAD = 2
OPT_HALF_SPREAD = 1

# Mispricing threshold for aggressive option taking
BS_EDGE = 3.0

# Base order quantities
HP_QTY  = 10
VEV_QTY = 10
OPT_QTY = 5


# ─── Math helpers (no scipy allowed) ─────────────────────────────────────────

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call(S: float, K: float, T: float, sigma: float) -> float:
    """
    Black-Scholes call price.
    T     : time to expiry in DAYS
    sigma : absolute vol in price-units / sqrt(day)
    """
    if T <= 0:
        return max(0.0, S - K)
    if S <= 0 or sigma <= 0:
        return max(0.0, S - K)
    vol = sigma / S          # convert to fractional vol
    try:
        d1 = (math.log(S / K) + 0.5 * vol * vol * T) / (vol * math.sqrt(T))
        d2 = d1 - vol * math.sqrt(T)
        return S * _norm_cdf(d1) - K * _norm_cdf(d2)
    except (ValueError, ZeroDivisionError):
        return max(0.0, S - K)


def bs_delta(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0:
        return 1.0 if S >= K else 0.0
    if sigma <= 0:
        return 1.0 if S >= K else 0.0
    vol = sigma / S
    try:
        d1 = (math.log(S / K) + 0.5 * vol * vol * T) / (vol * math.sqrt(T))
        return _norm_cdf(d1)
    except (ValueError, ZeroDivisionError):
        return 0.5


def _mid(depth: OrderDepth):
    if not depth.buy_orders or not depth.sell_orders:
        return None
    return (max(depth.buy_orders) + min(depth.sell_orders)) / 2.0


def _best_bid(depth: OrderDepth):
    return max(depth.buy_orders) if depth.buy_orders else None


def _best_ask(depth: OrderDepth):
    return min(depth.sell_orders) if depth.sell_orders else None


# ─── Trader ───────────────────────────────────────────────────────────────────

class Trader:
    """
    IMC Prosperity Round 3 — "Gloves Off"

    Products traded:
      • HYDROGEL_PACK         — market-making around EMA fair value
      • VELVETFRUIT_EXTRACT   — market-making around EMA fair value
      • VEV_4000 / VEV_4500   — deep ITM calls; treated as delta-1 (intrinsic only)
      • VEV_5000..VEV_5500    — near-ATM calls; Black-Scholes mispricing + MM
      • VEV_6000 / VEV_6500   — deep OTM, always 0.5 → skipped

    State (persisted via traderData JSON):
      hp_ema  : EMA of HYDROGEL_PACK mid price
      vev_ema : EMA of VELVETFRUIT_EXTRACT mid price (= spot S for BS)

    Black-Scholes calibration:
      sigma = 64 price-units/√day, TTE = 5 days at round start.
      Fitted from 3 days of historical data; max error ~3 units.
    """

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        # ── Load persisted EMAs ───────────────────────────────────────────────
        try:
            td = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            td = {}

        hp_ema  = float(td.get("hp_ema",  9990.0))
        vev_ema = float(td.get("vev_ema", 5250.0))

        result: Dict[str, List[Order]] = {}

        # ── Update EMAs ───────────────────────────────────────────────────────
        hp_depth  = state.order_depths.get("HYDROGEL_PACK")
        vev_depth = state.order_depths.get("VELVETFRUIT_EXTRACT")

        hp_mid  = _mid(hp_depth)  if hp_depth  else None
        vev_mid = _mid(vev_depth) if vev_depth else None

        if hp_mid is not None:
            hp_ema = HP_ALPHA * hp_mid + (1 - HP_ALPHA) * hp_ema
        if vev_mid is not None:
            vev_ema = VEV_ALPHA * vev_mid + (1 - VEV_ALPHA) * vev_ema

        S = vev_ema  # spot price for BS

        # ── HYDROGEL_PACK ─────────────────────────────────────────────────────
        if hp_depth:
            result["HYDROGEL_PACK"] = self._market_make(
                product="HYDROGEL_PACK",
                fv=hp_ema,
                half_spread=HP_HALF_SPREAD,
                base_qty=HP_QTY,
                pos=state.position.get("HYDROGEL_PACK", 0),
                limit=POSITION_LIMITS["HYDROGEL_PACK"],
                depth=hp_depth,
            )

        # ── VELVETFRUIT_EXTRACT ───────────────────────────────────────────────
        if vev_depth:
            result["VELVETFRUIT_EXTRACT"] = self._market_make(
                product="VELVETFRUIT_EXTRACT",
                fv=vev_ema,
                half_spread=VEV_HALF_SPREAD,
                base_qty=VEV_QTY,
                pos=state.position.get("VELVETFRUIT_EXTRACT", 0),
                limit=POSITION_LIMITS["VELVETFRUIT_EXTRACT"],
                depth=vev_depth,
            )

        # ── VEV Vouchers ──────────────────────────────────────────────────────
        for K in [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500]:
            product = f"VEV_{K}"
            depth   = state.order_depths.get(product)
            if depth is None:
                continue

            pos   = state.position.get(product, 0)
            limit = POSITION_LIMITS[product]
            fair  = bs_call(S, float(K), TTE, BS_SIGMA)

            # Deep ITM (4000, 4500): pure intrinsic, market-make like delta-1
            if K <= 4500:
                result[product] = self._market_make(
                    product=product,
                    fv=fair,
                    half_spread=OPT_HALF_SPREAD,
                    base_qty=OPT_QTY,
                    pos=pos,
                    limit=limit,
                    depth=depth,
                )
                continue

            # Near-ATM / OTM (5000–5500): BS mispricing edge + passive MM
            mkt = _mid(depth)
            if mkt is None:
                continue

            edge = fair - mkt
            orders: List[Order] = []

            if edge > BS_EDGE:
                # Market is underpricing → buy aggressively at best ask
                ask = _best_ask(depth)
                if ask is not None and ask < fair:
                    qty = min(OPT_QTY * 2, limit - pos)
                    if qty > 0:
                        orders.append(Order(product, ask, qty))

            elif edge < -BS_EDGE:
                # Market is overpricing → sell aggressively at best bid
                bid = _best_bid(depth)
                if bid is not None and bid > fair:
                    qty = min(OPT_QTY * 2, limit + pos)
                    if qty > 0:
                        orders.append(Order(product, bid, -qty))

            else:
                # Price near fair value → passive market-making
                orders = self._market_make(
                    product=product,
                    fv=fair,
                    half_spread=OPT_HALF_SPREAD,
                    base_qty=OPT_QTY,
                    pos=pos,
                    limit=limit,
                    depth=depth,
                )

            result[product] = orders

        # ── Persist EMAs ──────────────────────────────────────────────────────
        trader_data = json.dumps({"hp_ema": hp_ema, "vev_ema": vev_ema})

        return result, 0, trader_data

    # ── Market-making helper ──────────────────────────────────────────────────

    def _market_make(
        self,
        product: str,
        fv: float,
        half_spread: int,
        base_qty: int,
        pos: int,
        limit: int,
        depth: OrderDepth,
    ) -> List[Order]:
        """
        Quote a bid and ask around fair value `fv`.
        - Inventory skew: reduce size on the side where we're already exposed.
        - Aggressive take: lift/hit if market crosses our edge.
        """
        orders: List[Order] = []

        our_bid = round(fv) - half_spread
        our_ask = round(fv) + half_spread

        # Skew quantities based on current position
        skew     = pos / limit if limit else 0.0
        buy_qty  = max(1, round(base_qty * (1 - max(0.0,  skew))))
        sell_qty = max(1, round(base_qty * (1 + min(0.0, -skew))))

        can_buy  = limit - pos
        can_sell = limit + pos

        buy_qty  = min(buy_qty,  can_buy)
        sell_qty = min(sell_qty, can_sell)

        if buy_qty > 0:
            orders.append(Order(product, our_bid, buy_qty))
        if sell_qty > 0:
            orders.append(Order(product, our_ask, -sell_qty))

        # Aggressively take if someone is clearly on the wrong side of fair value
        ba = _best_ask(depth)
        bb = _best_bid(depth)

        if ba is not None and ba < round(fv):
            take = min(base_qty, can_buy - buy_qty)
            if take > 0:
                orders.append(Order(product, ba, take))

        if bb is not None and bb > round(fv):
            take = min(base_qty, can_sell - sell_qty)
            if take > 0:
                orders.append(Order(product, bb, -take))

        return orders