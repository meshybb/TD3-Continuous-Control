


Hyperparameters (vs. paper)

| Parameter | Paper | This code | Note |
|---|---|---|---|
| LR (actor/critic) | 1e-3 | 3e-4 | changed |
| Hidden dim | 400/300 | 256/256 | changed |
| Batch size | 100 | 256 | changed |
| τ (soft update) | 0.005 | 0.005 | same |
| Policy noise σ | 0.2 | 0.2 | same |
| Noise clip c | 0.5 | 0.5 | same |
| Policy delay d | 2 | 2 | same |
| Exploration noise | 0.1 | 0.1 | same |
| Discount γ | 0.99 | 0.99 | same |
| Start timesteps | 10,000 | 1,000 (InvertedPendulum) / 10,000 (HalfCheetah) | changed for InvertedPendulum |
| Max timesteps | 1,000,000 | 1,000,000 (InvPend) / 3,000,000 (HalfCheetah) | changed for HalfCheetah |
