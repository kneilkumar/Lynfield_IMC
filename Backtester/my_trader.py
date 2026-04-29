from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Tuple, Optional
import json
import math

POSITION_LIMITS = {
    "HYDROGEL_PACK": 200,
    "VELVETFRUIT_EXTRACT": 200,
    **{f"VEV_{k}": 300 for k in [4000,4500,5000,5100,5200,5300,5400,5500,6000,6500]},
}

# ================= HELPERS =================
def _best_bid(depth): return max(depth.buy_orders) if depth.buy_orders else None
def _best_ask(depth): return min(depth.sell_orders) if depth.sell_orders else None
def _mid(depth):
    b=_best_bid(depth); a=_best_ask(depth)
    return (b+a)/2 if b and a else None

def _norm_cdf(x): return 0.5*(1+math.erf(x/math.sqrt(2)))

def _bs_call(S,K,T,sigma):
    if T<=0 or sigma<=0: return max(S-K,0)
    d1=(math.log(S/K)+0.5*sigma*sigma*T)/(sigma*math.sqrt(T))
    d2=d1-sigma*math.sqrt(T)
    return S*_norm_cdf(d1)-K*_norm_cdf(d2)

def _implied_vol(price,S,K,T,sigma0=0.3):
    intrinsic=max(S-K,0)
    if price<=intrinsic+1e-4: return None
    sigma=sigma0
    for _ in range(50):
        d1=(math.log(S/K)+0.5*sigma*sigma*T)/(sigma*math.sqrt(T))
        v=S*math.exp(-0.5*d1*d1)/math.sqrt(2*math.pi)*math.sqrt(T)
        if v<1e-8: break
        sigma -= (_bs_call(S,K,T,sigma)-price)/v
        sigma=max(sigma,1e-6)
    return sigma

def _fit_smile(ms,ivs):
    n=len(ms)
    if n<3: return None
    S00=n; S10=sum(ms); S20=sum(m*m for m in ms)
    S30=sum(m**3 for m in ms); S40=sum(m**4 for m in ms)
    T0=sum(ivs)
    T1=sum(ms[i]*ivs[i] for i in range(n))
    T2=sum(ms[i]**2*ivs[i] for i in range(n))

    A=[[S00,S10,S20],[S10,S20,S30],[S20,S30,S40]]
    b=[T0,T1,T2]

    for i in range(3):
        for j in range(i+1,3):
            f=A[j][i]/A[i][i]
            for k in range(3): A[j][k]-=f*A[i][k]
            b[j]-=f*b[i]

    x=[0,0,0]
    for i in reversed(range(3)):
        x[i]=(b[i]-sum(A[i][j]*x[j] for j in range(i+1,3)))/A[i][i]

    return (x[0],x[1],x[2])

# ================= TRADER =================
class Trader:

    def run(self,state:TradingState):

        td=json.loads(state.traderData) if state.traderData else {}

        ema_vev=td.get("ema_vev",5247)
        ema_hp=td.get("ema_hp",9994)
        smile_hist=td.get("smile_hist",[])

        result={}

        # ===== MARKET =====
        vev_depth=state.order_depths.get("VELVETFRUIT_EXTRACT")
        hp_depth=state.order_depths.get("HYDROGEL_PACK")

        vev_mid=_mid(vev_depth) if vev_depth else None
        hp_mid=_mid(hp_depth) if hp_depth else None

        if vev_mid: ema_vev=0.005*vev_mid+0.995*ema_vev
        if hp_mid: ema_hp=0.001*hp_mid+0.999*ema_hp

        S=vev_mid if vev_mid else ema_vev
        T=4/252

        # ================= VELVETFRUIT (UNCHANGED) =================
        if vev_depth and vev_mid:
            bid=_best_bid(vev_depth)
            ask=_best_ask(vev_depth)
            pos=state.position.get("VELVETFRUIT_EXTRACT",0)

            dev=vev_mid-ema_vev

            orders=[]

            if dev>12 and bid:
                orders.append(Order("VELVETFRUIT_EXTRACT",bid,-15))
            elif dev<-12 and ask:
                orders.append(Order("VELVETFRUIT_EXTRACT",ask,15))

            if abs(dev)<8:
                fv=round(ema_vev)
                if pos<200:
                    orders.append(Order("VELVETFRUIT_EXTRACT",fv-3,2))
                if pos>-200:
                    orders.append(Order("VELVETFRUIT_EXTRACT",fv+3,-2))

            if orders: result["VELVETFRUIT_EXTRACT"]=orders

        # ================= HYDROGEL (UNCHANGED) =================
        if hp_depth and hp_mid:
            bid=_best_bid(hp_depth)
            ask=_best_ask(hp_depth)
            pos=state.position.get("HYDROGEL_PACK",0)

            fv=round(ema_hp)

            orders=[]
            if ask and ask<=fv-6:
                orders.append(Order("HYDROGEL_PACK",ask,3))
            if bid and bid>=fv+6:
                orders.append(Order("HYDROGEL_PACK",bid,-3))

            orders.append(Order("HYDROGEL_PACK",fv-6,3))
            orders.append(Order("HYDROGEL_PACK",fv+6,-3))

            result["HYDROGEL_PACK"]=orders

        # ================= VEV_4000 (UNCHANGED) =================
        depth=state.order_depths.get("VEV_4000")
        if depth and vev_mid:
            bid=_best_bid(depth)
            ask=_best_ask(depth)
            pos=state.position.get("VEV_4000",0)

            mid=(bid+ask)/2
            orders=[]

            orders.append(Order("VEV_4000",int(mid)-8,3))
            orders.append(Order("VEV_4000",int(mid)+8,-3))

            result["VEV_4000"]=orders

        # ================= OPTIONS (OUR STRATEGY) =================
        OPT_STRIKES=[5200,5300,5400,5500]

        opt_data=[]

        for K in OPT_STRIKES:
            depth=state.order_depths.get(f"VEV_{K}")
            if not depth: continue

            b=_best_bid(depth)
            a=_best_ask(depth)
            if not b or not a: continue

            mid=(b+a)/2
            iv=_implied_vol(mid,S,K,T)
            if iv is None: continue

            m=K-S
            opt_data.append((K,m,iv,mid,b,a))

            smile_hist.append((m,iv))

        if len(smile_hist)>500:
            smile_hist=smile_hist[-500:]

        if len(smile_hist)>=30:
            ms=[x[0] for x in smile_hist]
            ivs=[x[1] for x in smile_hist]
            coeffs=_fit_smile(ms,ivs)

            if coeffs:
                c,b_coef,a_coef=coeffs

                for K,m,iv,mid,b_price,a_price in opt_data:

                    smile_iv=max(a_coef*m*m+b_coef*m+c,0.05)
                    fair=_bs_call(S,K,T,smile_iv)

                    dev=mid-fair
                    pos=state.position.get(f"VEV_{K}",0)

                    orders=[]

                    if dev>1 and pos>-250:
                        orders.append(Order(f"VEV_{K}",b_price,-20))
                    elif dev<-1 and pos<250:
                        orders.append(Order(f"VEV_{K}",a_price,20))

                    if abs(dev)<0.5:
                        if pos>0:
                            orders.append(Order(f"VEV_{K}",b_price,-min(20,pos)))
                        elif pos<0:
                            orders.append(Order(f"VEV_{K}",a_price,min(20,-pos)))

                    if orders:
                        result.setdefault(f"VEV_{K}",[]).extend(orders)

        # ================= DEAD VOUCHERS =================
        for p in ["VEV_6000","VEV_6500"]:
            depth=state.order_depths.get(p)
            if not depth: continue

            orders=[]
            for price,qty in depth.sell_orders.items():
                if price==0:
                    orders.append(Order(p,0,-qty))
            for price,qty in depth.buy_orders.items():
                if price==1:
                    orders.append(Order(p,1,-qty))

            if orders: result[p]=orders

        td["ema_vev"]=ema_vev
        td["ema_hp"]=ema_hp
        td["smile_hist"]=smile_hist

        return result,0,json.dumps(td)