# conf/online_mix/ — Online-Mixing & Noise-Gen Policy YAMLs

Durable policy files describing **source pools and mixing rules** for
online-mixed training streams. These are **not** Hydra config-group entries
(nothing does `override /online_mix: ...`) — they are plain data files read at
runtime by `data_processing.frame_datasets.OnlineMixFrameDataset.from_yaml`,
referenced from a `conf/data/*.yaml` entry's `path:` field (or, for the
historical HPC/noise-gen variants, cited in `conf/experiment/*` comments and
`REPLICATION.md`).

Relocated here from the former top-level `configs/` directory (2026-07-03) when
`configs/` was folded into the Hydra tree.

| File | Loaded by | Purpose |
|---|---|---|
| `rp_{all,dregon,md2}.yaml`, `rp_*_labelaug.yaml` | `conf/data/rp_real.yaml` (`data.train.params.path`, set per experiment) | Paper real-data protocol streams (`docs/experiments/paper-real-protocol.md`): DREGON room 2 / FLY125 / both + silence at 16.7 % of chunks, 2 s, LibriSpeech per channel, severity ramps in global sample ids (16 chunks = 1 step); `_labelaug` adds `noise_rate_warp`. Validation `rp-real-valid-v1`. |
| `online_mix_v4_michaels_train_no_room1.yaml` | `conf/data/online_mix_v4_michaels.yaml` | V4-michaels train stream (DREGON `in_flight_noise` minus room1 + FLY125 + LibriSpeech) |
| `online_mix_v4_michaels_train_no_room1_no_aug.yaml` | `conf/data/online_mix_v4_michaels_no_aug.yaml` | Same, augmentation off (A/B baseline arm) |
| `online_mix_generated_augment_example.yaml` | `conf/data/online_mix_generated_augment.yaml` | Adds a `kind: generated` noise source (trained `PositionalHarmonicNoiseGen`) — E4 |
| `online_mix_v4_michaels_train_no_room1_gpfs.yaml` | (HPC variant; `conf/experiment/c10_arch_sweep_online.yaml` comment) | Same policy with `/gpfs` scratch paths |
| `noise_gen_online_dregon_michaels.yaml` | (historical; the current E2 data config approximates it offline) | Original per-frame online slicer policy for noise-gen — see `REPLICATION.md` § E2/E3 |
| `noise_gen_online_dregon_michaels_swapped.yaml` | (historical; E3's swapped split) | Swapped-split variant of the above |

Policy schema and the `kind:` source types are documented in
`src/data_processing/AGENTS.md` § "Online Mixing for RPS Prediction".
