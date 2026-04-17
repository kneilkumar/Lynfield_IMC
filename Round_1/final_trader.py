###########################################################
###########################################################


#THIS IS OUR FINAL SUBMISSION ONLY CHANGE WITH TEAM APPROVAL
#FOR INDIVIDUAL TESTING PLEASE SEE INDIVIDUAL CODE FOLDER

from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List
import numpy as np


class Trader:

    POSITION_LIMITS = {
        "ASH_COATED_OSMIUM": 80,
        "INTARIAN_PEPPER_ROOT": 80,
    }

    FAIR_PRICE = 10000

    # =========================
    # MAIN ENTRY
    # =========================
    def run(self, state: TradingState):
        results: Dict[str, List[Order]] = {}
        conversions = 0

        for product in state.order_depths:
            position = state.position.get(product, 0)
            limit = self.POSITION_LIMITS.get(product, 0)

            order_book = state.order_depths[product]

            # route to correct strategy WITHOUT changing logic
            if product == "ASH_COATED_OSMIUM":
                orders = self.trade_osmium(order_book, limit, position, product)

            elif product == "INTARIAN_PEPPER_ROOT":
                orders = self.trade_pepper(order_book, limit, position, product)

            else:
                continue

            results[product] = orders

        return results, conversions, ""

    # =========================
    # PEPPER STRATEGY (UNCHANGED)
    # =========================
    def trade_pepper(self, order_depth, limit, position, product):

        orders = []

        remaining = limit - position

        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders)
            best_ask = min(order_depth.sell_orders)
            mid_price = (best_bid + best_ask) / 2
        else:
            mid_price = None

        if remaining > 0 and order_depth.sell_orders and mid_price is not None:

            for ask_price in sorted(order_depth.sell_orders.keys()):
                ask_volume = abs(order_depth.sell_orders[ask_price])

                if ask_price > mid_price + 15:
                    continue

                volume = min(remaining, ask_volume)

                if volume > 0:
                    orders.append(Order(product, ask_price, volume))
                    remaining -= volume

                if remaining <= 0:
                    break

        return orders

    # =========================
    # OSMIUM STRATEGY (UNCHANGED)
    # =========================
    def trade_osmium(self, order_book, limit, position, product):

        orders = []

        buy_orders = order_book.buy_orders
        sell_orders = order_book.sell_orders

        if len(buy_orders) == 0 or len(sell_orders) == 0:
            return orders

        best_bid = max(buy_orders.keys())
        best_ask = min(sell_orders.keys())

        # mid_price = (best_bid + best_ask) // 2

      
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

        
        skew = int(0.7 * limit)

        if position > skew:
            orders.append(Order(product, self.FAIR_PRICE, -position))
            return orders

        if position < -skew:
            orders.append(Order(product, self.FAIR_PRICE, -position))
            return orders

        return orders