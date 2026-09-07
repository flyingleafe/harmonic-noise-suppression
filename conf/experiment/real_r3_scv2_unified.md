# real_r3_scv2_unified

Fresh R3 SCV2 rerun with unchanged batch size 128 and PIT-MSE training objective. Validation runs every 500 optimizer updates over the complete frozen real/static/stochastic panel. Log-smoothed any-subset progress drives LR reduction and saturation stopping; subset-specific stable-best checkpoints are retained.
