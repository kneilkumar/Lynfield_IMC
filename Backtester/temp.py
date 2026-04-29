from datamodel import Order
import numpy as np


class Trader:

    POSITION_LIMITS = {
        "ASH_COATED_OSMIUM": 80,
    }

    FAIR_PRICE = 10000

    def run(self, state):
        results = {}
        conversions = 0

        for product in state.order_depths:
            if product != "ASH_COATED_OSMIUM":
                continue

            order_book = state.order_depths[product]
            position = state.position.get(product, 0)
            limit = self.POSITION_LIMITS[product]

            orders = self.trade_osmium(order_book, limit, position, product)
            results[product] = orders

        return results, conversions, "OSMIUM"

    # =========================
    # MAIN STRATEGY
    # =========================
    def trade_osmium(self, order_book, limit, position, product):

        orders = []

        buy_orders = order_book.buy_orders
        sell_orders = order_book.sell_orders

        if len(buy_orders) == 0 or len(sell_orders) == 0:
            return orders

        best_bid = max(buy_orders.keys())
        best_ask = min(sell_orders.keys())

        mid_price = (best_bid + best_ask) / 2

        # =========================
        # 1. AGGRESSIVE MEAN REVERSION
        # =========================
        for ask_price, ask_volume in sorted(sell_orders.items()):
            if ask_price < self.FAIR_PRICE:
                qty = min(-ask_volume, limit - position)
                if qty > 0:
                    orders.append(Order(product, ask_price, qty))
                    position += qty

        for bid_price, bid_volume in sorted(buy_orders.items(), reverse=True):
            if bid_price > self.FAIR_PRICE:
                qty = min(bid_volume, limit + position)
                if qty > 0:
                    orders.append(Order(product, bid_price, -qty))
                    position -= qty

        # =========================
        # 2. MARKET MAKING
        # =========================
        bid_quote = best_bid + 1
        ask_quote = best_ask - 1

        mm_size = 10

        if bid_quote >= self.FAIR_PRICE:
            bid_quote = None
        if ask_quote <= self.FAIR_PRICE:
            ask_quote = None

        if bid_quote is not None:
            buy_cap = limit - position
            if buy_cap > 0:
                qty = min(mm_size, buy_cap)
                orders.append(Order(product, bid_quote, qty))
                position += qty

        if ask_quote is not None:
            sell_cap = limit + position
            if sell_cap > 0:
                qty = min(mm_size, sell_cap)
                orders.append(Order(product, ask_quote, -qty))
                position -= qty

        # =========================
        # 3. INVENTORY CONTROL
        # =========================
        skew = int(0.7 * limit)

        if position > skew:
            orders.append(Order(product, self.FAIR_PRICE, -position))
            return orders

        if position < -skew:
            orders.append(Order(product, self.FAIR_PRICE, -position))
            return orders

        return orders