# -------------------------------
# PARAMETERS
# -------------------------------

# Reserve prices (discrete)
r_values = list(range(670, 925, 5))  # 670 to 920 inclusive

# Bid search space
bids = list(range(670, 921))  # step = 1

# Assumed average b2 values
avg_values = list(range(750, 905, 5))

# -------------------------------
# FUNCTION TO COMPUTE EV
# -------------------------------

def compute_ev(b1, b2, avg_b2):
    total = 0

    for r in r_values:

        # Case 1: b1 trade
        if r < b1:
            profit = 920 - b1

        # Case 2: b2 trade
        elif b1 <= r < b2:
            profit = 920 - b2

            # Apply penalty if below or equal avg
            if b2 <= avg_b2:
                penalty = ((920 - avg_b2) / (920 - b2)) ** 3
                profit *= penalty

        # Case 3: no trade
        else:
            profit = 0

        total += profit

    return total / len(r_values)


# -------------------------------
# MAIN OPTIMISATION LOOP
# -------------------------------

results = []

for avg_b2 in avg_values:

    best_ev = -1
    best_pair = None

    for b1 in bids:
        for b2 in bids:

            if b2 <= b1:
                continue

            ev = compute_ev(b1, b2, avg_b2)

            if ev > best_ev:
                best_ev = ev
                best_pair = (b1, b2)

    results.append((avg_b2, best_pair, best_ev))


# -------------------------------
# OUTPUT RESULTS
# -------------------------------

print("Results by avg_b2:\n")

for avg_b2, (b1, b2), ev in results:
    print(f"avg_b2 = {avg_b2} → best (b1, b2) = ({b1}, {b2}), EV = {ev:.4f}")