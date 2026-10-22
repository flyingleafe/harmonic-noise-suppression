# Noise model v4 — pending paper edits (round 5)

Edits for `writing/papers/2026-08_wrapup/src/main.tex` (§ rotor-noise priors,
Table `tab:priors`, Algorithm `alg:fit-bench`), held here so that they do not
conflict with the edits on the paper's main branch. Apply them by hand once
that branch is merged. Line numbers are those of the paper at submodule commit
`9da03ef`; the anchors are the quoted text.

They follow the round-5 prior changes in `src/experiments/noise_model/v4_priors.py`
(commits `94050d5f`, the hard caps and one `c_W`; and the `b(f) >= 0`
truncation). The fit results these priors produce are still `\wip`.

## 1. Pedestal width prior paragraph (≈ l. 2439–2451)

Replace from "broad skirt on a weak line costs more than it buys. Its scale is
$c_W^{\mathrm{even}}$ …" through the `% source:` comment with:

```latex
broad skirt on a weak line costs more than it buys. Its scale $c_W$ is set so that the 97.5th
percentile of the measured widths of the blade-pass orders sits at the 97.5th percentile of the
half-normal, $2.24\,c_W$. The widths come from the phase coherence of all 20 bench recordings at
lags from 2.5 ms to 2 s, with the shaft removed and the band window, the floor and the tails
of the neighbouring orders in the forward model. Over 387 blade-pass orders this gives
$c_W = 6.9$ Hz. The median width grows with the order, from 1.0--1.6 Hz at $k \le 20$ to 8.7 Hz
at $k > 40$. The 94 other orders measure wider ($c_W = 30.7$ Hz). Under that scale, however, the
bench fit bought skirts of 100--140 Hz on them for 6--10 nats each and let them take the floor's
cells. A half-normal tail is too cheap against $1.5\times10^7$ cells. So every order takes the
blade-pass scale, and two hard caps bound the pedestal: $W_{\mathrm P,k} \le 24$ Hz, half the
slowest bench line spacing, so that a skirt never reaches the neighbouring lines; and
$\sigma^2_{\mathrm P,k} \le 3$, the 97.5th percentile of the measured variances (2.8) rounded
up, so that a line keeps at least 5\% of its power in the carrier.
% source: results/noise_v4/phase_width/{summary.json,findings.md} (omnirun nv4-a0-phasewidth-ff6078),
% parity c_W 6.939 / 30.678; even median W by group 4-8 1.00, 9-20 1.61, 41+ 8.70 Hz;
% s2P p97.5 2.81; slowest bench carrier 48.4 rev/s (motor_Motor2_50);
% round-4 runaway: results/noise_v4/bench/restarts/dregon__kappa0__s3.json (docs/experiments/noise-model-v4-bench.md, round 4)
```

## 2. Floor exponent truncation (new paragraph after the floor-push paragraph, ≈ l. 2457)

Insert after "… (Appendix~\ref{sec:app-checks}). \wip{the value of $\kappa$ from
the planted bench ladder.}" and before "On the bench the shaft follows …":

```latex
The floor exponent~\eqref{eq:floorexp} is truncated to $b(f) \ge 0$ over the band, 30 Hz to
7.9 kHz: the broadband floor does not get quieter as the rotor speeds up. Since $b$ is linear in
$\log f$, the two band edges carry the constraint. Without it the fits extrapolated the measured
slope below 300 Hz to $b(31\,\mathrm{Hz}) = -3.5$ in flight and $-6.9$ on the bench. Such a floor
grows as the rotors slow down, and at standby it rose 15 dB over the data at 23--47 Hz.
% source: results/noise_v4/flight/md2_fly125__flight_v4.json (b0 1.55, b1 0.85),
% results/noise_v4/bench/restarts/dregon__kappa0__s3.json (b0 0.58, b1 1.24); FLY124 @16+8 s
% standby: expected periodogram vs Welch, docs/experiments/noise-model-v4-bench.md round 5
```

## 3. Table `tab:priors` rows

```latex
$b_0$, $b_1$ & scalar & $N(3, 1)$; $N(0.5, 0.25)$ per octave; truncated to $b(f) \ge 0$ on the
  band & B, F \\
```

```latex
$\sigma^2_{\mathrm P,k}$ (rad$^2$) & order & $\mathrm{LN}(1, 1.5)$ truncated at 3 & B; prior median in
  flight \\
$W_{\mathrm P,k}$ (Hz) & order & $\mathrm{HN}(c_W)$ truncated at 24 Hz & B; prior median in flight \\
```

(replacing the `$b_0$, $b_1$` row and the two rows with `$c_W^{\mathrm{even}}$` /
`$c_W^{\mathrm{odd}}$`).

## 4. Algorithm `alg:fit-bench`, first step

Replace "set $c_W^{\mathrm{even}}$, $c_W^{\mathrm{odd}}$ from
$W_{\mathrm P}$~\eqref{eq:width}\;" with:

```latex
  phase increment variance; set $c_W$ from the blade-pass $W_{\mathrm P}$~\eqref{eq:width}\;
```
