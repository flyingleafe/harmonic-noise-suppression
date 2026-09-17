# Noise model v2, round 1 — supports

Every support is one observation window at 16 kHz in the frozen periodogram
convention `|rfft(x*w)|^2 / sum(w^2)` (periodic Hann, one-sided,
`revised_eval.window_periodogram`): bench = ONE frame over the whole segment,
flight = NFFT 2048 / hop 512. Numbers below are read from `index.json`, which is
written by `scripts/noise_v2_supports.py build`.

Sets present: bench-points, dregon-bench, dregon-floor, michaels-cruise. 
Total supports cached: 0.

## bench-points

0 of 135 specs built (0.0 s, git `955532017a5c`).

FAILURES: all 135 requested supports are unavailable: uni-cpu retry could not access dload manifests: botocore.exceptions.NoCredentialsError: Unable to locate credentials (initial submission failed before execution because PYTHONPATH=src was absent). Individual support names are in `index.json`.

No support material was resolved. The requested count is recorded above; segments, carrier estimates, and residual statistics are unavailable.

## dregon-bench

0 of 21 specs built (0.0 s, git `955532017a5c`).

FAILURES: all 21 requested supports are unavailable: uni-cpu retry could not access dload manifests: botocore.exceptions.NoCredentialsError: Unable to locate credentials (initial submission failed before execution because PYTHONPATH=src was absent). Individual support names are in `index.json`.

No support material was resolved. The requested count is recorded above; segments, carrier estimates, and residual statistics are unavailable.

## dregon-floor

0 of 10 specs built (0.0 s, git `955532017a5c`).

FAILURES: all 10 requested supports are unavailable: uni-cpu retry could not access dload manifests: botocore.exceptions.NoCredentialsError: Unable to locate credentials (initial submission failed before execution because PYTHONPATH=src was absent). Individual support names are in `index.json`.

No support material was resolved. The requested count is recorded above; segments, carrier estimates, and residual statistics are unavailable.

## michaels-cruise

0 of 13 specs built (0.0 s, git `955532017a5c`).

FAILURES: all 13 requested supports are unavailable: uni-cpu retry could not access dload manifests: botocore.exceptions.NoCredentialsError: Unable to locate credentials (initial submission failed before execution because PYTHONPATH=src was absent). Individual support names are in `index.json`.

No support material was resolved. The requested count is recorded above; segments, carrier estimates, and residual statistics are unavailable.

## Bytes

* `bench-points`: 0.0 MB of `.npz` (power as float32)
* `dregon-bench`: 0.0 MB of `.npz` (power as float32)
* `dregon-floor`: 0.0 MB of `.npz` (power as float32)
* `michaels-cruise`: 0.0 MB of `.npz` (power as float32)

The `.npz` caches are NOT committed (`results/**` is gitignored and they are hundreds of MB); `index.json` and this file are. Any consumer regenerates a cache with `python scripts/noise_v2_supports.py build --set <set>`.
