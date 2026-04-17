from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List

class Trader:

    POSITION_LIMIT = {
        "INTARIAN_PEPPER_ROOT": 80,
        "ASH_COATED_OSMIUM": 80,
    }

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        conversions = 0

        # =========================
        # LOOP THROUGH PRODUCTS
        # =========================
        for product in state.order_depths:

            order_depth: OrderDepth = state.order_depths[product]
            position = state.position.get(product, 0)
            limit = self.POSITION_LIMIT.get(product, 0)

            orders: List[Order] = []

            # ============================================================
            # ================= INTARIAN PEPPER ROOT =====================
            # ============================================================
            if product == "INTARIAN_PEPPER_ROOT":

                remaining = limit - position

                if remaining > 0 and order_depth.sell_orders:
                    for ask_price in sorted(order_depth.sell_orders.keys()):
                        ask_volume = abs(order_depth.sell_orders[ask_price])

                        volume = min(remaining, ask_volume)
                        if volume > 0:
                            orders.append(Order(product, ask_price, volume))
                            remaining -= volume

                        if remaining <= 0:
                            break

            # ============================================================
            # ================= ASH COATED OSMIUM ========================
            # ============================================================
            elif product == "ASH_COATED_OSMIUM":

                if not order_depth.buy_orders or not order_depth.sell_orders:
                    result[product] = orders
                    continue

                # ---------------- FAIR VALUE (WALL MID) ----------------
                min_bid = min(order_depth.buy_orders.keys())
                max_ask = max(order_depth.sell_orders.keys())
                fair_value = (min_bid + max_ask) / 2

                remaining_buy = limit - position
                remaining_sell = limit + position

                # =========================
                # 1. TAKE EDGE
                # =========================

                # BUY cheap
                for ask_price in sorted(order_depth.sell_orders.keys()):
                    ask_volume = abs(order_depth.sell_orders[ask_price])

                    if ask_price < fair_value and remaining_buy > 0:
                        volume = min(remaining_buy, ask_volume)
                        orders.append(Order(product, ask_price, volume))
                        remaining_buy -= volume
                    else:
                        break

                # SELL expensive
                for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
                    bid_volume = order_depth.buy_orders[bid_price]

                    if bid_price > fair_value and remaining_sell > 0:
                        volume = min(remaining_sell, bid_volume)
                        orders.append(Order(product, bid_price, -volume))
                        remaining_sell -= volume
                    else:
                        break

                # =========================
                # 2. MARKET MAKING (IMPROVED)
                # =========================

                # Base prices
                bid_price = int(fair_value - 1)
                ask_price = int(fair_value + 1)

                # Overbid best bid (stay below fair)
                for bp in sorted(order_depth.buy_orders.keys(), reverse=True):
                    if bp < fair_value:
                        bid_price = max(bid_price, bp + 1)
                        break

                # Undercut best ask (stay above fair)
                for sp in sorted(order_depth.sell_orders.keys()):
                    if sp > fair_value:
                        ask_price = min(ask_price, sp - 1)
                        break

                # =========================
                # 3. INVENTORY SKEW
                # =========================

                skew = int(position * 0.1)
                bid_price -= skew
                ask_price -= skew

                # =========================
                # 4. PLACE ORDERS
                # =========================

                if remaining_buy > 0:
                    orders.append(Order(product, bid_price, remaining_buy))

                if remaining_sell > 0:
                    orders.append(Order(product, ask_price, -remaining_sell))

            # =========================
            # SAVE ORDERS
            # =========================
            result[product] = orders

        return result, conversions, ""