# R3 DREGON label registration: do the labels register the real comb lines?

`noise-v2-dregon-registration/1`, git `01e1464bc0c5b424c3d65abca4e690f6ae6acc82`. Read-only; laptop CPU.

Question (R3 plan, `#model-r3` round table): the DREGON frozen-comb fit puts the comb at `comb_gain_db = -3.232` dB while the band-level match the HPPNet gate needs is `+21.8` dB — a 25.03 dB disagreement. If the label carrier does not sit on the real lines the likelihood never saw the comb and the quiet comb is an artefact; if it does, the quiet comb is honest and the gate disagrees with the spectrum.

## What the label does inside a 4 s score window

| window | mean carriers (rev/s) | within-window drift (rev/s) | sorted gaps |
|---|---|---|---|
| `free-flight` | 84.38, 76.69, 81.65, 78.91 | 3.49, 4.06, 3.93, 4.34 | 2.22, 2.74, 2.74 |
| `hovering` | 85.07, 77.07, 82.10, 79.35 | 8.92, 16.02, 10.90, 14.75 | 2.28, 2.75, 2.98 |
| `updown` | 82.78, 75.62, 80.86, 77.90 | 19.93, 14.47, 20.11, 15.83 | 2.28, 2.96, 1.92 |
| `rectangle` | 84.13, 76.65, 81.89, 79.67 | 25.00, 24.99, 25.00, 25.00 | 3.02, 2.23, 2.23 |
| `spinning` | 84.62, 76.35, 82.28, 79.55 | 20.92, 15.52, 18.56, 18.09 | 3.20, 2.73, 2.34 |

The two published tracks agree: the `rps_refined` and `motors_command` window means differ by at most 0.0438 rev/s, so no verdict below depends on which one is called 'the label'.

## Separability (d)

| analysis | bin (Hz) | frames | k separable from k | k unusable above (drift) | orders used |
|---|---:|---:|---:|---:|---:|
| `window_4s` | 0.25 | 1 | 1 | 29 | 29-39 |
| `frame_1.024s` | 0.97656 | 6 | 1 | 40 | 40-55 |
| `frame_2048` | 7.8125 | 122 | 8 | 72 | 65-85 |

## Registration per window x rotor

`n>6 dB` counts orders whose line clears the local median floor by 6 dB. The peak-over-median estimator is biased upward by the search window and by the eight-microphone average, so every count is given beside the SAME read taken half an order off every line (`@null`), where the model puts nothing. The decision statistic is the paired sign test between the two, `z`.

### `frame_2048`

| window | rotor | k used | delta* (rev/s) | S gain % | S gain % @null | n>6 dB @label | @delta* | @null | z @label | z @delta* | mean excess (dB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `free-flight` | 0 | 85 | -1.246 | 2.22 | 0.37 | 0 | 0 | 1 | 0.76 | 0.33 | -0.054 |
| `free-flight` | 1 | 65 | -1.496 | 5.33 | 0.8 | 0 | 0 | 0 | -1.12 | -0.37 | -0.26 |
| `free-flight` | 2 | 71 | -1.114 | 3.24 | 0.0 | 0 | 1 | 0 | -0.36 | 0.59 | -0.042 |
| `free-flight` | 3 | 67 | -0.646 | 1.71 | 2.86 | 0 | 0 | 0 | 0.86 | 0.61 | 0.051 |
| `hovering` | 0 | 75 | -1.5 | 4.38 | 0.27 | 0 | 1 | 0 | -1.73 | 0.81 | -0.161 |
| `hovering` | 1 | 47 | -1.48 | 3.75 | 0.34 | 0 | 0 | 0 | -1.02 | -0.44 | -0.242 |
| `hovering` | 2 | 43 | -1.238 | 1.99 | 0.17 | 0 | 0 | 0 | 0.46 | 0.46 | -0.023 |
| `hovering` | 3 | 62 | -0.798 | 1.67 | 2.11 | 0 | 1 | 0 | -0.25 | -0.51 | 0.082 |
| `updown` | 0 | 56 | -1.212 | 2.75 | 0.21 | 0 | 0 | 0 | 0.0 | 0.8 | -0.046 |
| `updown` | 1 | 44 | -1.442 | 1.64 | 0.54 | 0 | 0 | 0 | -1.51 | -0.3 | -0.101 |
| `updown` | 2 | 35 | -0.67 | 0.48 | 2.32 | 0 | 0 | 0 | 1.18 | 1.86 | 0.105 |
| `updown` | 3 | 57 | -0.27 | 0.48 | 3.71 | 0 | 0 | 0 | -0.66 | -0.93 | 0.101 |
| `rectangle` | 0 | 80 | -1.5 | 3.63 | 0.33 | 1 | 1 | 1 | 0.0 | -0.22 | -0.138 |
| `rectangle` | 1 | 64 | -1.418 | 3.24 | 0.32 | 0 | 2 | 1 | 0.75 | 0.25 | -0.129 |
| `rectangle` | 2 | 53 | -0.902 | 1.83 | 1.56 | 3 | 3 | 1 | 0.69 | 1.24 | 0.091 |
| `rectangle` | 3 | 57 | -0.186 | 0.95 | 4.07 | 2 | 2 | 2 | 1.46 | 0.66 | 0.213 |
| `spinning` | 0 | 83 | -1.262 | 1.23 | 0.32 | 0 | 0 | 1 | -0.33 | 0.33 | -0.045 |
| `spinning` | 1 | 55 | -1.5 | 3.9 | 0.68 | 0 | 0 | 0 | -0.67 | -1.21 | -0.18 |
| `spinning` | 2 | 55 | 0.408 | 0.62 | 1.36 | 1 | 1 | 0 | -1.75 | 0.4 | -0.013 |
| `spinning` | 3 | 58 | 0.15 | 0.31 | 3.52 | 0 | 0 | 1 | 1.05 | 1.31 | 0.158 |

### `order_tracked_4s`

| window | rotor | k used | delta* (rev/s) | S gain % | S gain % @null | n>6 dB @label | @delta* | @null | z @label | z @delta* | mean excess (dB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `free-flight` | 0 | 91 | 0.526 | 6.99 | 11.41 | 5 | 6 | 3 | 0.73 | 2.2 | 0.246 |
| `free-flight` | 1 | 101 | 1.288 | 13.31 | 5.55 | 3 | 6 | 8 | -0.9 | 1.49 | -0.218 |
| `free-flight` | 2 | 95 | -1.434 | 17.06 | 5.97 | 3 | 9 | 1 | -0.31 | 3.39 | 0.157 |
| `free-flight` | 3 | 98 | -0.366 | 2.94 | 8.25 | 4 | 8 | 2 | 1.41 | 2.63 | 0.414 |
| `hovering` | 0 | 91 | -1.158 | 11.24 | 1.64 | 4 | 7 | 3 | 0.1 | 1.36 | -0.055 |
| `hovering` | 1 | 100 | 1.348 | 15.41 | 2.09 | 1 | 4 | 5 | -2.0 | 0.0 | -0.387 |
| `hovering` | 2 | 94 | -0.936 | 7.14 | 6.73 | 5 | 7 | 1 | -0.21 | 1.44 | 0.194 |
| `hovering` | 3 | 97 | -0.684 | 8.35 | 16.19 | 7 | 10 | 1 | 2.34 | 3.15 | 0.593 |
| `updown` | 0 | 93 | -1.434 | 6.38 | 3.37 | 3 | 3 | 2 | -1.35 | 0.73 | -0.039 |
| `updown` | 1 | 102 | 0.836 | 4.15 | 7.72 | 1 | 3 | 6 | 0.2 | 1.39 | -0.023 |
| `updown` | 2 | 95 | -0.18 | 6.19 | 2.75 | 1 | 3 | 5 | 1.74 | 1.74 | 0.036 |
| `updown` | 3 | 99 | -0.186 | 1.94 | 6.82 | 2 | 2 | 2 | 1.91 | 2.31 | 0.254 |
| `rectangle` | 0 | 92 | -1.44 | 6.96 | 3.4 | 9 | 17 | 14 | -0.42 | -0.21 | -0.021 |
| `rectangle` | 1 | 101 | -0.456 | 5.88 | 3.48 | 11 | 15 | 16 | -0.1 | 0.3 | 0.023 |
| `rectangle` | 2 | 94 | -1.17 | 23.78 | 7.96 | 8 | 22 | 12 | -0.62 | 2.68 | 0.011 |
| `rectangle` | 3 | 97 | -0.572 | 7.72 | 10.51 | 17 | 21 | 7 | 2.34 | 4.16 | 0.654 |
| `spinning` | 0 | 91 | -0.936 | 13.21 | 2.48 | 4 | 6 | 3 | -0.52 | 1.99 | -0.041 |
| `spinning` | 1 | 101 | -1.304 | 13.98 | 3.57 | 1 | 10 | 8 | -1.69 | -0.1 | -0.409 |
| `spinning` | 2 | 94 | 0.768 | 16.41 | 6.8 | 4 | 9 | 2 | -1.44 | 2.06 | -0.044 |
| `spinning` | 3 | 97 | 0.1 | 3.03 | 10.3 | 12 | 12 | 3 | 3.15 | 3.55 | 0.522 |

One order does stand above the null: `k = 1`, the shaft rate itself, at a mean label-vs-null SNR of `free-flight` 13.5 vs 4.0; `hovering` 15.4 vs 3.6; `updown` 5.7 vs 4.3; `rectangle` 18.5 vs 3.9; `spinning` 12.6 vs 3.8 dB. It decides nothing about registration and is included in the sums above only because the sign test weights every order equally: at `k = 1` the search window spans a carrier offset of +-4.5 rev/s, three times the whole grid, so that order cannot localise `delta` at all; and its line sits at 76-85 Hz where the floor's own steep low-frequency slope biases a +-30 Hz local median downward, while the null reads it at 115-128 Hz on the flat part. Whether it is a real shaft line or a slope artefact, it carries no comb: one order of ~95, and the frozen comb's power is in `k >= 2`.

## Verdict

Taken on the order-tracked 4 s read, where one bin at `k = 2` is ~0.10 rev/s. At the likelihood's `frame_2048` one bin at `k = 2` is 3.9 rev/s — wider than the whole search — so the `|delta*| < 1 bin` half of the test cannot discriminate there; its `S` gain and line counts are in the table above.

| window | verdict | max \|delta*\| (rev/s) | in k=2 bins | max S gain % | max z @label | max z @delta* | rotors with a comb |
|---|---|---:|---:|---:|---:|---:|---:|
| `free-flight` | **NO_COMB** (4/4 rotors) | 1.434 | 11.47 | 17.06 | 1.41 | 3.39 | 0/4 |
| `hovering` | **NO_COMB** (4/4 rotors) | 1.348 | 10.78 | 15.41 | 2.34 | 3.15 | 0/4 |
| `updown` | **NO_COMB** (4/4 rotors) | 1.434 | 11.47 | 6.38 | 1.91 | 2.31 | 0/4 |
| `rectangle` | **NO_COMB** (4/4 rotors) | 1.440 | 11.52 | 23.78 | 2.34 | 4.16 | 0/4 |
| `spinning` | **REGISTERED** (1/4 rotors) | 0.100 | 0.80 | 3.03 | 3.15 | 3.55 | 1/4 |

Bars: a comb is DETECTED at the label at `z >= 3.0` and at `delta*` at `z >= 5.0` (the second is selection-corrected — see the self-test). REGISTERED needs a detection at the label with `|delta*|` inside a `k = 2` bin and an `S` gain under 10 %; MISREGISTERED needs a detection off the label; NO_COMB is returned when no offset in +-1.5 rev/s finds a comb at all, which is a different implication for the comb level and so is not folded into MISREGISTERED by an argmax chasing noise.

## Self-test: the sensitivity behind the null

A null result is worth only the sensitivity behind it, so the whole check was re-run on `free-flight_nosource_room2` plus a SYNTHETIC comb of the model's own form, planted on the label at descending band levels (dB re that window's own 30-7900 Hz power; 0 dB is what the gate's band-level-matched arm renders) and once at a deliberate +0.4 rev/s offset. `passed = True`.

| injected comb level (dB) | planted delta (rev/s) | verdict | rotors detected | max \|delta* error\| (rev/s) | median z @label | median z @delta* | median mean excess (dB) |
|---:|---:|---|---:|---:|---:|---:|---:|
| +0 | +0.0 | REGISTERED | 4/4 | 0.052 | 9.80 | 9.80 | 11.35 |
| -6 | +0.0 | REGISTERED | 4/4 | 0.032 | 9.70 | 9.70 | 10.45 |
| -12 | +0.0 | REGISTERED | 4/4 | 0.048 | 9.39 | 9.39 | 8.55 |
| -18 | +0.0 | REGISTERED | 4/4 | 0.026 | 8.80 | 8.80 | 5.34 |
| -24 | +0.0 | REGISTERED | 4/4 | 0.048 | 5.44 | 5.64 | 2.31 |
| -30 | +0.0 | REGISTERED | 1/4 | 0.804 | 2.19 | 2.39 | 0.85 |
| -36 | +0.0 | NO_COMB | 0/4 | 1.434 | 1.17 | 2.72 | 0.31 |
| +0 | +0.4 | MISREGISTERED | 0/4 | 0.054 | 0.15 | 9.11 | -0.14 |

So the check finds a comb planted ON the label down to -24.0 dB on every rotor (and on one rotor at -30 dB), calls it REGISTERED with `delta*` recovered to 0.052 rev/s; loses it at -36 dB WITHOUT a false MISREGISTERED (`no_false_positive = True`); and calls the +0.4 rev/s-offset comb MISREGISTERED with `delta*` recovered to 0.054 rev/s, `z @delta*` 8.3-9.5 and `z @label` 0.31. The real windows show neither signature.

## What it implies for the comb level

* `exact re-evaluation`: not cheap: needs a spectrum.py flight forward pass per grid point, and spectrum/model are being rewritten for R3 concurrently.
* `proxy`: 10 log10 sum_k line-excess(delta*) / sum_k line-excess(0).
* proxy recovery from re-registering at `delta*`: frame median 0.18 dB (max 0.5); order-tracked median 0.03 dB (max 1.44).
* comb level the spectra support, read off the self-test calibration: `comb_level_db = None`, `upper bound = -36.0` dB re the window's own band power, from a measured median excess of 0.017 dB and median z of -0.15.

**Conclusion.** Over the 40 rotor reads (5 windows x 4 rotors x 2 label tracks) the check returns 0 MISREGISTERED, 1 REGISTERED and 39 NO_COMB. Nothing is misregistered: where a comb is detectable at all it sits ON the label, and where it is not, no offset in +-1.5 rev/s finds one. So the first branch of the question holds — **the likelihood's quiet comb is honest**. Re-registering would buy 0.03 dB (median, order-tracked; 0.18 dB on the likelihood's own frame), not 25.03 dB. The resolved comb in these windows is at most -36.0 dB of the window's own band power, so the band-level-matched arm the gate likes (+21.8 dB, comb band power = the real clip's band power) is louder than ANY comb the spectrum contains by at least 36 dB. The 25.03 dB disagreement is therefore not a registration bug and not a likelihood bug: the gate's scorer needs a comb the DREGON room-2 cruise spectrum does not have, and R3's frozen-comb `c` cannot serve both.

## Figures

* `results/noise_v2/rounds/round3/registration/score_curves.png`
* `results/noise_v2/rounds/round3/registration/line_snr.png`
* `results/noise_v2/rounds/round3/registration/labels.png`
