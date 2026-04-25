import numpy

r_values = list(range(670, 925,5))

bids = list(range(670, 925, 5))

best_ev = -1
best_b1 = None

for b1 in bids:
    ev = 0
    for r in r_values:
        if b1 >= r:
            ev += (920 - b1)

    ev /= len(r_values)

    if ev > best_ev:
        best_ev = ev
        best_b1 = b1
print(best_b1, best_ev)

