"""Throwaway: does a per-mic transfer explain the Michael's LTAS proxy gap?

Renders the r4a per-regime arm on the two frozen cruise supports exactly as
noise_v2_round_score does (absolute level, seed 2001), then EQs the SYNTHETIC
mic 0 by mic 0's FLY125 deviation from the mic mean (mic_gains.json
levels_db.full, cruise windows, 24 third-octave bands, log-f interpolation) and
re-scores ltas_deviation_db(real, arm, mic=0). Prints per-band deviations.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import noise_v2_round_score as SC  # noqa: E402

from experiments.noise_model import gates as GT  # noqa: E402
from experiments.stochastic_fit import revised_eval as RE  # noqa: E402

fits = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/noise_v3/fits_r4a")
render_mod, _ = SC._v2_modules()
arm = SC.v2_regime_arm(
    "michaels",
    render_mod=render_mod,
    cruise_path=fits / "michaels_fly125_cruise__flight_v3.json",
    standby_path=fits / "michaels_fly125_standby__flight_v3.json",
)

rec = json.loads(Path("results/noise_v2/mic_gains/mic_gains.json").read_text())["rigs"]["michaels"]
full = np.asarray(rec["levels_db"]["full"], dtype=np.float64)  # (W, M, B)
names = [w["name"] for w in rec["windows"]]
cruise_w = [i for i, n in enumerate(names) if "+8_" in n]
dev = full[cruise_w] - full[cruise_w].mean(axis=1, keepdims=True)  # (Wc, M, B)
transfer = dev.mean(axis=0)  # (M, B)
band_hz = np.asarray(rec["band_centres_hz"], dtype=np.float64)
print("mic 0 transfer (dB) on the 24 bands:", transfer[0].round(1))


def eq(audio: np.ndarray, sr: int) -> np.ndarray:
    out = audio.copy()
    f = np.fft.rfftfreq(audio.shape[-1], 1.0 / sr)
    lf = np.log(np.maximum(f, band_hz[0]))
    for m in range(audio.shape[0]):
        h_db = np.interp(lf, np.log(band_hz), transfer[m])
        out[m] = np.fft.irfft(np.fft.rfft(audio[m]) * 10.0 ** (h_db / 20.0), n=audio.shape[-1])
    return out


for support in GT.MICHAELS_CRUISE_SUPPORTS:
    clip = RE.load_window(
        support.window,
        dataset=GT.DATASET["michaels"],
        version=None,
        channels=None,
        rps_key=GT.RAW_RPS_KEY["michaels"],
    )
    real = np.asarray(clip.audio, dtype=np.float64)[:8]
    ref = np.atleast_2d(np.asarray(clip.rps, dtype=np.float64))
    for seed in (2001, 2002, 2003, 2004):
        audio = arm.render(ref, regime="cruise", n_mics=8, seed=seed)[:8]
        for label, a in (("as rendered", audio), ("mic-0 EQ", eq(audio, int(clip.sr)))):
            r = RE.ltas_deviation_db(real, a, mic=0)
            d = np.asarray(r["bands_arm_db"]) - np.asarray(r["bands_real_db"])
            print(
                f"{support.key} s{seed} {label:12s} ltas_abs {r['mean_abs_db']:.3f}  offset "
                f"{r['level_offset_db']:+.2f}  per band {np.round(d, 2)}"
            )
