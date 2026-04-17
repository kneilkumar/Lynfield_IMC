###########################################################
###########################################################


#THIS IS OUR FINAL SUBMISSION ONLY CHANGE WITH TEAM APPROVAL
#FOR INDIVIDUAL TESTING PLEASE SEE INDIVIDUAL CODE FOLDER

from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List

class Trader:

    POSITION_LIMIT = 80
    THRESHOLD = 15  # tweakable

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        conversions = 0

        product = "INTARIAN_PEPPER_ROOT"
        orders: List[Order] = []

        if product not in state.order_depths:
            return {}, conversions, ""

        order_depth: OrderDepth = state.order_depths[product]
        position = state.position.get(product, 0)

        remaining = self.POSITION_LIMIT - position

        # Compute mid price
        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders)
            best_ask = min(order_depth.sell_orders)
            mid_price = (best_bid + best_ask) / 2
        else:
            mid_price = None

        # Sweep with filter
        if remaining > 0 and order_depth.sell_orders and mid_price is not None:

            for ask_price in sorted(order_depth.sell_orders.keys()):
                ask_volume = abs(order_depth.sell_orders[ask_price])

                # FILTER: skip extreme overpriced levels
                if ask_price > mid_price + self.THRESHOLD:
                    continue

                volume = min(remaining, ask_volume)

                if volume > 0:
                    orders.append(Order(product, ask_price, volume))
                    remaining -= volume

                if remaining <= 0:
                    break

        result[product] = orders
        return result, conversions, ""