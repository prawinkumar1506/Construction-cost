import pandas as pd

a = pd.read_csv("submission_lb_matching.csv")
b = pd.read_csv("submission_ultimate_location_time.csv")

TARGET = "construction_cost_per_m2_usd"

# weighted average
a[TARGET] = 0.6 * a[TARGET] + 0.4 * b[TARGET]

a.to_csv("submission_ensemble.csv", index=False)

print("Saved submission_ensemble.csv")
