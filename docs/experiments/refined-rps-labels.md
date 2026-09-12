# Refined rotor-speed labels: regime gating, full coverage, publication

**Status**: closed 2026-09-12. `DREGON-frames@261b09971c8a` (recipe_version 2)
and `michaels-frames@8e9d149560dd` (recipe_version 3) carry `rps_refined`.

## Why

The raw rotor-speed label is a tachometer (`motors_measured`, Michael's `rps`)
or a command track (`motors_command`), and it carries a scale-like error of
0.3–0.8 % of rate (`telemetry-fitness.md` § 6d). The F_VK/L-BFGS refiner
(`scripts/refine_dregon_rps.py`) corrects it against the recording's own comb.
Two defects blocked using it as a published label.

## Defect 1 — standby was refined, and should not be

Measured on the committed sidecars: DREGON `free-flight_nosource_room1` was
corrected by −0.8 to −2.0 rev/s throughout standby, and FLY125 carried a
+1.1 rev/s excursion on one rotor exactly at the ramp.

The refiner's own objective says those corrections do not earn their keep. Comb
fitness, telemetry vs refined labels, same audio:

| regime | FLY125 | room1 |
|---|---:|---:|
| standby | **−0.0021** | **−0.0788** |
| ramp | +0.0098 | +0.0140 |
| cruise | +0.0021 | +0.0169 |

A standby shaft is nearly static, its comb weak and dense; the correction found
there is fitted to noise.

**Policy** (`data_processing/rps_gating.py`, `GatePolicy`): standby (slowest
rotor < 45 rev/s) keeps the telemetry EXACTLY; settled cruise (≥ 65 rev/s)
takes the refinement EXACTLY; the ramp blends the *correction*,
`r = r_tel + w·(r_ref − r_tel)`, so both end points are exact by construction.
`w` is a smoothstep in the slowest rotor's rate, held down for `settle_s = 1 s`
after the rig leaves standby. The regime is read from the **slowest** rotor: a
window with one rotor still idling is not cruise.

Gating lives in the producer (`refine_worker` skips non-cruise windows,
`stitch` gates before writing); `scripts/fix_refined_labels.py` re-gates
existing sidecars. Sidecars keep the ungated stitch as `r_refined_raw` plus
`gate_weight` / `gate_policy`.

### Four gate bugs found by measuring, not by reading

1. **Envelope keyed on cruise entry was discontinuous.** Just below the
   threshold the rate term is already ≈1; at the first cruising frame the
   elapsed time is 0, so the minimum dropped `w` to 0 and ramped it back — a
   step in the middle of the merge. Keyed on **standby exit** both factors
   vanish together. Measured: "drops while the rate rises" = 0 on all five
   recordings.
2. **A global max-step continuity check is blind.** These recordings ramp at up
   to 80 rev/s per frame, which swamps any gate artefact. The check now
   separates the weight's step in **smooth** stretches (≤ 0.069, bound 0.182)
   from steps where the telemetry itself jumps across the gate band (room1 goes
   67.3 → 0.0 rev/s in one frame at motor cut-off).
3. **Settle blanked mid-flight starts.** A recording that begins out of standby
   has no ramp to settle after; it now starts settled. Caught by the existing
   stitch test, which read 80.0 (telemetry) where the refinement said 79.0.
4. **Anchoring.** A sidecar's `ft` is relative to the frame's **audio**
   `t_start` and already includes `t0_offset_s`, so anchoring on the telemetry
   track double-shifts it — 0.01 s on Michael's rig, 5.48 s on room1, whose
   clock is absolute unix seconds.

## Defect 2 — room2 could not be refined at all

`load_published_noise_sources` was pinned to `motors_measured` and skips a frame
lacking the requested track. DREGON room1 carries the tachometer; the five
`*_room2` flights publish **only** `motors_command`. The refiner therefore
reported success on one room1 flight and silently dropped every room2
recording. With a per-frame preference chain (`RPS_KEY_PREFERENCE`,
`resolve_rps_key`) the loader yields 6 recordings instead of 1.

## Coverage

12 sidecars, all gate invariants measured (standby bit-identical to telemetry,
settled cruise bit-identical to the raw refinement):

| recording | reference key | standby | correction rms (rev/s) |
|---|---|---:|---:|
| FLY124 | `rps` | 38.4 % | 0.130 |
| FLY125 | `rps` | 12.2 % | 0.102 |
| free-flight_nosource_room1 | `motors_measured` | 7.0 % | 0.452 |
| free-flight_speech-high_room1 | `motors_measured` | 9.7 % | 0.262 |
| free-flight_speech-low_room1 | `motors_measured` | 9.6 % | 0.475 |
| free-flight_whitenoise-high_room1 | `motors_measured` | 9.8 % | 0.236 |
| free-flight_whitenoise-low_room1 | `motors_measured` | 11.5 % | 0.373 |
| free-flight_nosource_room2 | `motors_command` | 4.3 % | 0.279 |
| hovering_nosource_room2 | `motors_command` | 9.1 % | 0.266 |
| rectangle_nosource_room2 | `motors_command` | 8.6 % | 0.498 |
| spinning_nosource_room2 | `motors_command` | 10.0 % | 0.156 |
| updown_nosource_room2 | `motors_command` | 7.9 % | 0.316 |

Compute: `uni-cpu`, one job per recording. The first batch died with
`BrokenProcessPool` — four refiner workers at ~12 GB each (k_max 40, 4
channels) against a 32 GB request. Resubmitted with `--jobs 1` and 64 GB, all
succeeded. A Kaggle GPU smoke run succeeded in 2 m 45 s wall but Kaggle buffers
logs until completion and the `--outputs` glob collected nothing, so **no
per-unit GPU timing was obtained**; CPU was sufficient (11.2 s for the k_max 10
single-channel smoke window).

## Publication

`rps_refined` is an **additional** series; the raw track is untouched beside it.

* Attachment happens in the `source_frames` **derivation**, not in the
  builders, which stay raw — a builder cannot state which label bytes a
  published dataset contains.
* Each spec's `gen["refined_labels"]` records the track name, the gate policy,
  and a `{recording_id: 16-hex SHA-256 prefix}` manifest scoped to that
  source's own recordings. `_verify_refined_labels` checks it at generation, so
  a re-refinement or a changed gate mints a new derivation identity instead of
  letting dload memoize stale labels.
* Recordings without a sidecar publish their reference track under the refined
  name, so the field is never partially defined. The 262 DREGON bench runs have
  no rotor label and no track.

Verified by reading the **published** versions back: all 10 DREGON recordings
with telemetry match their sidecars with the first stamp at each recording's own
`t0_offset` (+2.52 s … +6.50 s); FLY124 (4, 3801) and FLY125 (4, 5553) at
+0.0100 s. Sizes: DREGON 2.7 GiB / 19 shards / 272 samples, Michael's
422.9 MiB / 2 shards.

## Open

* The sidecars are git artifacts, not a dload dataset. A consumer that wants
  refined labels without the frames still reads
  `src/data_processing/refined_labels/`.
* `decomp-frames-v1/v2` still pin `rps_override_dir` for room1 only; they were
  not re-derived here.
* The GPU path (`TRACKING_DEVICE=cuda`, F_VK supports it) remains unbenchmarked.
* Gate thresholds (45 / 65 rev/s, `settle_s = 1 s`) are a policy chosen from the
  measured regime structure of these two rigs, not fitted.
