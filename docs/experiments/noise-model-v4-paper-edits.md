# Noise model v4 — pending paper edits (rounds 5–6)

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

## 5. The value of κ (≈ l. 2592)

Replace "\wip{the value of $\kappa$ from the planted bench ladder.}" with:

```latex
No value of $\kappa$ meets that rule on the DREGON ladder. On a planted twin of the bench fit,
$\kappa = 0$ recovers the floor above 500 Hz to 1.5 dB rms (2.4 dB at most) and to $+2.5$ dB on
average below it; $\kappa = 100$ and $1000$ lift the floor below 500 Hz by $+16.5$ and $+19.5$ dB,
and $\kappa = 10^4$ diverges. Below 500 Hz the bench floor and the bench wind explain the same
cells, and the push only moves the floor up. We therefore fit with $\kappa = 0$.
% source: results/noise_v4/bench/planted/dregon__kappa0__r5__s2__floor2500__kappa{0,100}__r1.json;
% docs/experiments/noise-model-v4-bench.md, round 5
```

The same value goes into the check-outcome `\wip` at ≈ l. 2855 (§ 7 below).

## 6. Fitted bench laws (≈ l. 1642)

Replace the `\wip{The fitted exponents …}` with (the four-rotor prediction is not
run yet; keep that half as `\wip`):

```latex
The converged bench fit has $\alpha^{\mathrm a} = 3.92$, $\alpha^{\mathrm m} = -0.11$,
$b_0 = 4.04$ and $b_1 = -1.15$ per octave, a phase-pedestal variance of 1.0 (median over
orders) with a median width of 3.8 Hz at $k \le 20$, and a shaft with
$\sigma_\nu(80\revps) = 0.73$ rad/s and $\lambda_{\mathrm{ref}} = 52$ s$^{-1}$
($a_\sigma = 2.11$, $b_\lambda = 4.44$).
% source: results/noise_v4/bench/restarts/dregon__kappa0__r5__s2__c7.json params
\wip{the four-rotor prediction against its 1.5 dB and 2 dB bars.}
```

## 7. Check outcomes (≈ l. 2855)

Replace the `\wip{The outcome of checks …}` with (DREGON half; MD2 half below):

```latex
On the planted twin of the final bench fit ($\kappa = 0$), $\alpha^{\mathrm a}$ is recovered to
0.04, $\varepsilon$ to 4\,\% and $\lambda_{\mathrm{ref}}$ to 5\,\%. The line profiles $\ge 10$ dB
over the floor are recovered to 0.35 dB (median; 88\,\% within 1 dB) and the pedestal widths
$\ge 15$ dB over the floor to 18\,\% (median; 71\,\% within 30\,\%). $b_0$ is off by 1.0 and the
floor fails its 1 dB rule where the truth floor lies far under the lines. The final fit is a
minimum: one direction of curvature $-0.28$ is left, along which no step lowers the
objective. The likelihood weight of the bench fit by block jackknife is
$\eta = 1.5\times10^{-3}$ at 200 Hz blocks and $1.3\times10^{-3}$ with one block per recording.
On the planted twin, where the model is exact, $\eta = 3.3\times10^{-2}$ and $3.1\times10^{-2}$.
The Whittle likelihood is therefore overconfident by a factor of about 30 even without model
error, and the model error of the real recordings adds a factor of about 22. The twin is a minimum
in the same sense: 6 directions of curvature down to $-0.31$ are left, along which the best step
gains less than the fit's convergence tolerance ($10^{-4}$ nats per cell).
% source: results/noise_v4/bench/planted/dregon__kappa0__r5__s2__c6__kappa0__r1.json;
% results/noise_v4/bench/eta/dregon__kappa0__r5__s2__c7.json;
% results/noise_v4/bench/eta/dregon__kappa0__r5__s2__c6__kappa0__r1.fit__c3.json
```

## 8. Flight fit of the Matrice 100 (after the "Flight fit." paragraph, ≈ l. 1645)

```latex
One joint fit over both regimes converges but fails the standby gate (PIT 27.8\revps{}), so
the Matrice 100 is fitted per regime, as before, and the renderer joins the two fits across the
ramp. Two changes make the regime fits converge to sensible values. The phase pedestal is fitted
per order: held at its prior median it gives every line 63\,\% of its power in the skirt, and the
valleys between the cruise lines are 10 dB deeper than that. The motor-line exponent is bounded,
$\lvert\alpha^{\mathrm m}\rvert \le 4$: a regime spans only 1.4--1.9$\times$ in speed, and the
free exponent ran to $-40.8$ and $+17.5$, which the ramp then extrapolates.
% source: docs/experiments/noise-model-v4-bench.md, round 6; src/experiments/noise_model/v4_priors.py
```

Table `tab:priors`: the `$\sigma^2_{\mathrm P,k}$` and `$W_{\mathrm P,k}$` rows say
"B; prior median in flight" — change to "B, F". The `$\alpha^{\mathrm m}$` row gains "F:
truncated at $\lvert\alpha^{\mathrm m}\rvert \le 4$".

## 9. Scores (≈ l. 1710 and l. 2866)

Matrice 100 (converged per-regime pair, FLY124, four render seeds):

```latex
On the Matrice 100 the per-regime ratio is 0.481 (bar 1.05; the earlier flight fit 0.515), with
PIT 0.359, 3.309 and 0.700\revps{} on standby, ramp and cruise, and a cruise long-term-spectrum
deviation of 1.516 dB (bar 1.332 dB, earlier fit 1.541 dB).
% source: omnirun nv4-score-md2-r6c3-449bbb parity/arm_michaels_v4.json
```

DREGON: the bench lines in flight fail the gate (best converged flight stage
15.8\revps{} against the bar 2.188 and the earlier fit's 1.680). Do not write a
DREGON score sentence until a DREGON fit passes; the reasons are in
`docs/explainers/noise-model-v4-dregon-gap.qmd` § 5.
