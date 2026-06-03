# The hyperparameters we used: 

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
| Start timesteps | 10,000 | 5,000 (InvPend) / 10,000 (HalfCheetah) | changed for InvPend |
| Max timesteps | 1,000,000 | 1,000,000 (InvPend) / 3,000,000 (HalfCheetah) | changed for HalfCheetah |


---- 

# Results: 

```bash
========================================================================
 --- Analyzing 3 runs for InvertedPendulum-v4---
 Seed 55: Final Reward = 1000.0
 Seed 42: Final Reward = 1000.0
 Seed 68: Final Reward = 1000.0

 --- Summary ---
  Final rewards: ['1000.0', '1000.0', '1000.0']
  Mean ± Std   : 1000.00 ± 0.00
========================================================================
========================================================================
--- Analyzing 3 runs for HalfCheetah-v4 ---
Seed 55: Final Reward = 11445.77
Seed 42: Final Reward = 12602.26
Seed 68: Final Reward = 13182.54

--- Summary ---
Final rewards: [11445.77, 12602.26, 13182.54]
Mean ± Std   : 12410.19 ± 721.92
========================================================================
```
For creating vidoes of the trained policy, we used the following models of each environment:

- InvertedPendulum: results/InvertedPendulum-v4_run1_seed42.pt
- HalfCheetah: results/HalfCheetah-v4_run1_seed68_best.pt

(creatin the video: python record_video.py --env HalfCheetah-v4 --ckpt results/HalfCheetah-v4_seed68_best.pt
                    python record_video.py --env InvertedPendulum-v4 --ckpt results/InvertedPendulum-v4_run1_seed42.pt