from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List
import jsonpickle


class Trader:
    POSITION_LIMITS = {
        "EMERALDS": 80,
        "TOMATOES": 80,
    }

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        conversions = 0

        if state.traderData:
            trader_data = jsonpickle.decode(state.traderData)
        else:
            trader_data = {
                "emerald_mids": [],
                "tomato_mids": []
            }

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            position = state.position.get(product, 0)
            limit = self.POSITION_LIMITS[product]

            if not order_depth.buy_orders or not order_depth.sell_orders:
                result[product] = orders
                continue

            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            mid_price = (best_bid + best_ask) / 2

            if product == "EMERALDS":
                trader_data["emerald_mids"].append(mid_price)
                if len(trader_data["emerald_mids"]) > 40:
                    trader_data["emerald_mids"].pop(0)

                short_window = trader_data["emerald_mids"][-8:]
                long_window = (
                    trader_data["emerald_mids"][-20:]
                    if len(trader_data["emerald_mids"]) >= 20
                    else trader_data["emerald_mids"]
                )

                short_fair = sum(short_window) / len(short_window)
                long_fair = sum(long_window) / len(long_window)
                fair_value = 0.7 * short_fair + 0.3 * long_fair

                orders += self.trade_emeralds_blended(
                    product, order_depth, position, limit, fair_value
                )

            elif product == "TOMATOES":
                trader_data["tomato_mids"].append(mid_price)
                if len(trader_data["tomato_mids"]) > 40:
                    trader_data["tomato_mids"].pop(0)

                short_window = trader_data["tomato_mids"][-8:]
                long_window = (
                    trader_data["tomato_mids"][-20:]
                    if len(trader_data["tomato_mids"]) >= 20
                    else trader_data["tomato_mids"]
                )

                short_fair = sum(short_window) / len(short_window)
                long_fair = sum(long_window) / len(long_window)
                fair_value = 0.7 * short_fair + 0.3 * long_fair

                orders += self.trade_tomatoes_57761(
                    product, order_depth, position, limit, fair_value
                )

            result[product] = orders

        traderData = jsonpickle.encode(trader_data)
        return result, conversions, traderData

    def trade_emeralds_blended(self, product, order_depth, position, limit, fair_value):
        orders = []

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())

        # Aggressive taking around blended fair
        for ask in sorted(order_depth.sell_orders.keys()):
            ask_vol = -order_depth.sell_orders[ask]
            if ask <= fair_value:
                qty = min(ask_vol, limit - position)
                if qty > 0:
                    orders.append(Order(product, ask, qty))
                    position += qty

        for bid in sorted(order_depth.buy_orders.keys(), reverse=True):
            bid_vol = order_depth.buy_orders[bid]
            if bid >= fair_value:
                qty = min(bid_vol, position + limit)
                if qty > 0:
                    orders.append(Order(product, bid, -qty))
                    position -= qty

        # Passive quoting around blended fair
        bid_quote = int(fair_value - 2)
        ask_quote = int(fair_value + 2)

        # Inventory skew
        if position > 60:
            bid_quote = int(fair_value - 4)
            ask_quote = int(fair_value + 1)
        elif position > 30:
            bid_quote = int(fair_value - 3)
            ask_quote = int(fair_value + 1)
        elif position < -60:
            bid_quote = int(fair_value - 1)
            ask_quote = int(fair_value + 4)
        elif position < -30:
            bid_quote = int(fair_value - 1)
            ask_quote = int(fair_value + 3)

        # Keep quotes sensible relative to current market
        bid_quote = min(bid_quote, best_bid + 1)
        ask_quote = max(ask_quote, best_ask - 1)

        buy_qty = min(18, limit - position)
        sell_qty = min(18, position + limit)

        if buy_qty > 0:
            orders.append(Order(product, bid_quote, buy_qty))
        if sell_qty > 0:
            orders.append(Order(product, ask_quote, -sell_qty))

        return orders

    def trade_tomatoes_57761(self, product, order_depth, position, limit, fair_value):
        orders = []

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())

        take_edge = 2.0

        for ask in sorted(order_depth.sell_orders.keys()):
            ask_vol = -order_depth.sell_orders[ask]
            if ask < fair_value - take_edge:
                qty = min(ask_vol, limit - position)
                if qty > 0:
                    orders.append(Order(product, ask, qty))
                    position += qty

        for bid in sorted(order_depth.buy_orders.keys(), reverse=True):
            bid_vol = order_depth.buy_orders[bid]
            if bid > fair_value + take_edge:
                qty = min(bid_vol, position + limit)
                if qty > 0:
                    orders.append(Order(product, bid, -qty))
                    position -= qty

        bid_quote = int(fair_value - 2)
        ask_quote = int(fair_value + 2)

        if position > 60:
            bid_quote = int(fair_value - 4)
            ask_quote = int(fair_value + 1)
        elif position > 30:
            bid_quote = int(fair_value - 3)
            ask_quote = int(fair_value + 1)
        elif position < -60:
            bid_quote = int(fair_value - 1)
            ask_quote = int(fair_value + 4)
        elif position < -30:
            bid_quote = int(fair_value - 1)
            ask_quote = int(fair_value + 3)

        bid_quote = min(bid_quote, best_bid + 1)
        ask_quote = max(ask_quote, best_ask - 1)

        buy_qty = min(18, limit - position)
        sell_qty = min(18, position + limit)

        if buy_qty > 0:
            orders.append(Order(product, bid_quote, buy_qty))
        if sell_qty > 0:
            orders.append(Order(product, ask_quote, -sell_qty))

        return orders
    
    ##version 7

    