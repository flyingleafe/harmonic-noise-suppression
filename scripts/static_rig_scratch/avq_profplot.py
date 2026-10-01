import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

keys = ["S1_seq1", "S1_seq2", "S1_seq3"]
fig, axes = plt.subplots(3, 4, figsize=(22, 12), sharey="row")
for r, key in enumerate(keys):
    z = np.load(f"/tmp/avq_refine_{key}.npz")
    for c, f in enumerate("ABCD"):
        ax = axes[r, c]
        pw, fl, s1 = z[f"{f}_pw"], z[f"{f}_fl"], z[f"{f}_s1"]
        K = pw.shape[1]
        kk = np.arange(1, K + 1)
        coll = np.zeros((len(s1), K), bool)
        for g in "ABCD":
            if g == f:
                continue
            so = z[f"{g}_s1"][:, None]
            fk = kk[None, :] * s1[:, None]
            coll |= np.abs(fk - np.round(fk / so) * so) < 3.0
        lvl = 10 * np.log10(np.maximum(pw, 1e-30)).mean(axis=2)  # mic-mean dB per block
        flv = 10 * np.log10(np.maximum(fl, 1e-30)).mean(axis=2)
        snr = 10 * np.log10(np.maximum(pw / fl - 1, 1e-3))
        snr_med = np.array(
            [
                np.nanmedian(np.median(snr[:, k, :], axis=1)[~coll[:, k]])
                if (~coll[:, k]).sum() > 5
                else np.nan
                for k in range(K)
            ]
        )
        lvl_med = np.array(
            [
                np.nanmedian(lvl[~coll[:, k], k]) if (~coll[:, k]).sum() > 5 else np.nan
                for k in range(K)
            ]
        )
        p10 = np.array(
            [
                np.nanpercentile(lvl[~coll[:, k], k], 10) if (~coll[:, k]).sum() > 5 else np.nan
                for k in range(K)
            ]
        )
        p90 = np.array(
            [
                np.nanpercentile(lvl[~coll[:, k], k], 90) if (~coll[:, k]).sum() > 5 else np.nan
                for k in range(K)
            ]
        )
        fl_med = np.nanmedian(flv, axis=0)
        ok = np.isfinite(lvl_med) & (kk * s1.mean() <= 10000)
        det = ok & (snr_med > 6)
        ax.plot(kk[ok], fl_med[ok], "k--", lw=0.8, label="noise in 3 Hz ENBW (20th-pct floor)")
        ax.vlines(kk[ok], p10[ok], p90[ok], color="0.8", lw=1)
        ax.plot(
            kk[ok & ~det],
            lvl_med[ok & ~det],
            "o",
            ms=3,
            mfc="none",
            color="C0",
            label="line power, median over blocks (line/noise <= 6 dB)",
        )
        ax.plot(kk[det], lvl_med[det], "o", ms=4, color="C3", label="line/noise > 6 dB")
        for k in kk[det]:
            if k > 8:
                ax.annotate(
                    str(k),
                    (k, lvl_med[k - 1]),
                    fontsize=7,
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                )
        ax.set_title(
            f"{key} rotor {f}  s={s1.mean():.1f} rev/s  (f = k*s; k=42 -> {42 * s1.mean():.0f} Hz)"
        )
        ax.set_xlim(0, 130)
        ax.grid(alpha=0.3)
        if c == 0:
            ax.set_ylabel("dB (mean-square power, arbitrary ref), mic-mean")
        if r == 2:
            ax.set_xlabel("order k")
axes[0, 0].legend(fontsize=7, loc="upper right")
fig.suptitle(
    "AVQ harmonic profiles along the block-refined tracks (0.5 s blocks, 8 mics); grey bars = p10-p90 over blocks; collisions with other rotors' harmonics removed"
)
fig.tight_layout()
fig.savefig("/tmp/avq_profiles.png", dpi=100)
