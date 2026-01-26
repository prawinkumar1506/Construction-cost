"""
META-ENSEMBLE: COMBINE EXISTING BEST + EXTERNAL
Strategy: Blend your proven 0.2317 submissions with external data predictions
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

print("="*80)
print("META-ENSEMBLE: EXISTING BEST + EXTERNAL DATA")
print("Target: Beat 0.2317 by intelligent blending")
print("="*80)

# =========================
# LOAD EXISTING SUBMISSIONS
# =========================
print("\n[STEP 1/3] Loading existing submissions...")

try:
    # Your best submissions
    sub_ensemble = pd.read_csv("submission_ensemble.csv")  # 0.2317
    sub_blend = pd.read_csv("submission_blend_70_30.csv")  # 0.2320
    
    # External data submissions
    sub_hybrid = pd.read_csv("submission_hybrid_external.csv")  # New
    
    print(f"   Loaded ensemble: {sub_ensemble.shape}")
    print(f"   Loaded blend: {sub_blend.shape}")
    print(f"   Loaded hybrid: {sub_hybrid.shape}")
    
except Exception as e:
    print(f"   ERROR: {e}")
    print("   Make sure all submission files exist!")
    exit()

# =========================
# LOAD OOF PREDICTIONS FOR CV VALIDATION
# =========================
print("\n[STEP 2/3] Loading OOF predictions for validation...")

try:
    # Load training data for CV validation
    train = pd.read_csv("datasets/train_tabular.csv")
    y_true = train["construction_cost_per_m2_usd"]
    
    # Load OOF predictions (if available)
    try:
        oof_hybrid = np.load("oof_hybrid_external.npy")
        print(f"   Loaded hybrid OOF: {oof_hybrid.shape}")
        
        # Calculate CV MAE
        cv_hybrid = mean_absolute_error(y_true, oof_hybrid)
        print(f"   Hybrid CV MAE: {cv_hybrid:.4f}")
        
    except:
        print("   No OOF available, using test predictions only")
        oof_hybrid = None
        
except Exception as e:
    print(f"   Warning: {e}")
    oof_hybrid = None

# =========================
# META-ENSEMBLE BLENDING
# =========================
print("\n[STEP 3/3] Creating meta-ensemble...")

# Extract predictions
pred_ensemble = sub_ensemble["construction_cost_per_m2_usd"].values
pred_blend = sub_blend["construction_cost_per_m2_usd"].values
pred_hybrid = sub_hybrid["construction_cost_per_m2_usd"].values

print(f"   Ensemble stats: mean={pred_ensemble.mean():.0f}, std={pred_ensemble.std():.0f}")
print(f"   Blend stats: mean={pred_blend.mean():.0f}, std={pred_blend.std():.0f}")
print(f"   Hybrid stats: mean={pred_hybrid.mean():.0f}, std={pred_hybrid.std():.0f}")

# Strategy 1: Conservative blend (mostly existing best)
pred_conservative = 0.7 * pred_ensemble + 0.2 * pred_blend + 0.1 * pred_hybrid

# Strategy 2: Balanced blend
pred_balanced = 0.5 * pred_ensemble + 0.3 * pred_blend + 0.2 * pred_hybrid

# Strategy 3: Aggressive blend (more external data)
pred_aggressive = 0.4 * pred_ensemble + 0.3 * pred_blend + 0.3 * pred_hybrid

# Strategy 4: Best two only (ignore external if it's bad)
pred_best_two = 0.6 * pred_ensemble + 0.4 * pred_blend

# =========================
# CREATE SUBMISSIONS
# =========================
print("\n[CREATING] Meta-ensemble submissions...")

# Get data_id from any submission
data_id = sub_ensemble["data_id"]

# Create all meta-ensemble submissions
submissions = {
    "submission_meta_conservative.csv": pred_conservative,
    "submission_meta_balanced.csv": pred_balanced,
    "submission_meta_aggressive.csv": pred_aggressive,
    "submission_meta_best_two.csv": pred_best_two
}

for filename, predictions in submissions.items():
    # Clip predictions
    predictions_clipped = np.clip(predictions, 50, 6000)
    
    # Create submission
    submission = pd.DataFrame({
        "data_id": data_id,
        "construction_cost_per_m2_usd": predictions_clipped
    })
    
    submission.to_csv(filename, index=False)
    
    print(f"   Saved: {filename}")
    print(f"      Mean: {predictions_clipped.mean():.0f}")
    print(f"      Range: [{predictions_clipped.min():.0f}, {predictions_clipped.max():.0f}]")

# =========================
# RECOMMENDATIONS
# =========================
print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)

print("\nSubmission Priority Order:")
print("1. submission_meta_conservative.csv  (70% ensemble + 20% blend + 10% external)")
print("2. submission_meta_best_two.csv      (60% ensemble + 40% blend, NO external)")
print("3. submission_meta_balanced.csv      (50% ensemble + 30% blend + 20% external)")
print("4. submission_meta_aggressive.csv    (40% ensemble + 30% blend + 30% external)")

print("\nStrategy:")
print("- Conservative: Safest bet, minimal risk")
print("- Best Two: Ignore external data completely")
print("- Balanced: Moderate external data inclusion")
print("- Aggressive: Maximum external data (risky)")

print("\nExpected Performance:")
print("- Conservative/Best Two: Should beat or match 0.2317")
print("- Balanced: Might improve slightly")
print("- Aggressive: Higher risk, could be worse")

print("\n" + "="*80)
print("META-ENSEMBLE COMPLETE")
print("="*80)