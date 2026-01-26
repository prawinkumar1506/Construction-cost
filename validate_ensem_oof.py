import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

TARGET = "construction_cost_per_m2_usd"

train = pd.read_csv("datasets/train_tabular.csv")
y = train[TARGET].values

oof_a = np.load("oof_lb_matching.npy")
oof_b = np.load("oof_ultimate.npy")

# individual scores
mae_a = mean_absolute_error(y, oof_a)
mae_b = mean_absolute_error(y, oof_b)

# ensemble
oof_ens = 0.6 * oof_a + 0.4 * oof_b
mae_ens = mean_absolute_error(y, oof_ens)

print("==============================")
print("MAE Model A (lb_matching):", mae_a)
print("MAE Model B (ultimate):   ", mae_b)
print("MAE Ensemble (0.6A+0.4B): ", mae_ens)
print("==============================")
