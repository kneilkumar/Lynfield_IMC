import numpy as np

best_pnl = -1e9
best_x = None
best_y = None

# Loop through all possible integer allocations
for x in range(0, 101):  # Research %
    for y in range(0, 101 - x):  # Scale % (ensures x + y <= 100)

        # Compute Research output
        R = 200000 * np.log(1 + x) / np.log(101)

        # Compute Scale multiplier
        S = 7 * y / 100

        # Compute Budget used
        budget = 50000 * (x + y) / 100

        # Speed = 0 → multiplier = 0.1
        pnl = 0.1 * R * S - budget

        # Track best result
        if pnl > best_pnl:
            best_pnl = pnl
            best_x = x
            best_y = y

# Print result
print("Optimal Allocation (Speed = 0):")
print(f"Research: {best_x}%")
print(f"Scale: {best_y}%")
print(f"Speed: 0%")
print(f"Max PnL: {best_pnl:.2f}")