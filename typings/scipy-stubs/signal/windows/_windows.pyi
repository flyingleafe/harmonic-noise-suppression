# Return types for scipy's window functions, so that pyright stops inferring them.
#
# scipy ships no types for `scipy.signal`, so pyright infers a return type from
# the library source. For `get_window` that means the union over all 26 window
# functions, and it costs about four minutes of checking per pyright process:
# 357 s for a file whose only statement is `get_window("hann", 8)`, and the
# same for one call of anything built on it — welch, csd, coherence,
# periodogram, spectrogram, firwin, resample_poly (pyright 1.1.407, scipy
# 1.17.0, 2026-11-20). 61 files of this repository call one of those. The
# pre-commit hook paid it per process and every language server paid it again
# whenever its type cache was dropped. With the returns declared here the same
# probes check in 1-8 s, and the types inferred for welch and the rest are the
# ones they had before.
#
# Signatures follow scipy 1.17.0. Arguments are deliberately `Any`: the point
# is the return types, and no call that type-checked before may stop doing so.
#
# `py.typed` at the package root says "partial" (PEP 561): pyright overlays
# this package on the installed scipy and takes every other module from the
# library. pyright 1.1.407 does that overlay by redirecting EVERY `.pyi` path
# under scipy/ to this package, whether or not the file exists here, which
# hides the stubs scipy itself ships (`scipy.special` would lose its ufuncs,
# `scipy.spatial.distance` its `pdist`). Each of them is therefore linked back
# from the same relative path here; `tests/test_typings.py` fails when a scipy
# upgrade adds one that is not.

from typing import Any

import numpy.typing as npt

__all__ = [
    "barthann",
    "bartlett",
    "blackman",
    "blackmanharris",
    "bohman",
    "boxcar",
    "chebwin",
    "cosine",
    "dpss",
    "exponential",
    "flattop",
    "gaussian",
    "general_cosine",
    "general_gaussian",
    "general_hamming",
    "get_window",
    "hamming",
    "hann",
    "kaiser",
    "kaiser_bessel_derived",
    "lanczos",
    "nuttall",
    "parzen",
    "taylor",
    "triang",
    "tukey",
]

type _Array = npt.NDArray[Any]

def boxcar(M: Any, sym: Any = True, *, xp: Any = None, device: Any = None) -> _Array: ...
def triang(M: Any, sym: Any = True, *, xp: Any = None, device: Any = None) -> _Array: ...
def parzen(M: Any, sym: Any = True, *, xp: Any = None, device: Any = None) -> _Array: ...
def bohman(M: Any, sym: Any = True, *, xp: Any = None, device: Any = None) -> _Array: ...
def blackman(M: Any, sym: Any = True, *, xp: Any = None, device: Any = None) -> _Array: ...
def nuttall(M: Any, sym: Any = True, *, xp: Any = None, device: Any = None) -> _Array: ...
def blackmanharris(M: Any, sym: Any = True, *, xp: Any = None, device: Any = None) -> _Array: ...
def flattop(M: Any, sym: Any = True, *, xp: Any = None, device: Any = None) -> _Array: ...
def bartlett(M: Any, sym: Any = True, *, xp: Any = None, device: Any = None) -> _Array: ...
def barthann(M: Any, sym: Any = True, *, xp: Any = None, device: Any = None) -> _Array: ...
def hamming(M: Any, sym: Any = True, *, xp: Any = None, device: Any = None) -> _Array: ...
def kaiser(M: Any, beta: Any, sym: Any = True, *, xp: Any = None, device: Any = None) -> _Array: ...
def kaiser_bessel_derived(
    M: Any, beta: Any, *, sym: Any = True, xp: Any = None, device: Any = None
) -> _Array: ...
def gaussian(
    M: Any, std: Any, sym: Any = True, *, xp: Any = None, device: Any = None
) -> _Array: ...
def general_cosine(M: Any, a: Any, sym: Any = True) -> _Array: ...
def general_gaussian(
    M: Any, p: Any, sig: Any, sym: Any = True, *, xp: Any = None, device: Any = None
) -> _Array: ...
def general_hamming(
    M: Any, alpha: Any, sym: Any = True, *, xp: Any = None, device: Any = None
) -> _Array: ...
def chebwin(M: Any, at: Any, sym: Any = True, *, xp: Any = None, device: Any = None) -> _Array: ...
def cosine(M: Any, sym: Any = True, *, xp: Any = None, device: Any = None) -> _Array: ...
def hann(M: Any, sym: Any = True, *, xp: Any = None, device: Any = None) -> _Array: ...
def exponential(
    M: Any,
    center: Any = None,
    tau: Any = 1.0,
    sym: Any = True,
    *,
    xp: Any = None,
    device: Any = None,
) -> _Array: ...
def tukey(
    M: Any, alpha: Any = 0.5, sym: Any = True, *, xp: Any = None, device: Any = None
) -> _Array: ...
def taylor(
    M: Any,
    nbar: Any = 4,
    sll: Any = 30,
    norm: Any = True,
    sym: Any = True,
    *,
    xp: Any = None,
    device: Any = None,
) -> _Array: ...

# `return_ratios=True` returns `(windows, ratios)`.
def dpss(
    M: Any,
    NW: Any,
    Kmax: Any = None,
    sym: Any = True,
    norm: Any = None,
    return_ratios: Any = False,
    *,
    xp: Any = None,
    device: Any = None,
) -> Any: ...
def get_window(
    window: Any, Nx: Any, fftbins: Any = True, *, xp: Any = None, device: Any = None
) -> _Array: ...
def lanczos(M: Any, *, sym: Any = True, xp: Any = None, device: Any = None) -> _Array: ...
