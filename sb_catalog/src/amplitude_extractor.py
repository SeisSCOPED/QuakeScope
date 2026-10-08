import logging
import sys
import threading
import time
import warnings
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Optional

import numpy as np
import obspy
from .sb_lite import sbu  # QS_SBLITE: seisbench.util under off

logger = logging.getLogger("amplitude_extractor")

# IASPEI-standard Wood-Anderson: T0 = 0.8 s, damping h = 0.7, static gain 2080,
# applied to ground VELOCITY (one zero at the origin). The poles are
# -h*w0 +- i*w0*sqrt(1-h^2) with w0 = 2*pi/0.8.
#
# These were previously [-6.283 +- 4.7124j], which is h = 0.8 - Richter's
# original damping, but paired with IASPEI's 2080 gain rather than Richter's
# 2800, i.e. a mix of two conventions. Switching to the IASPEI pair raises
# amplitudes by a near-uniform 0.033 ML (measured on five CI stations,
# station-to-station spread 0.010 ML) and makes this catalog comparable to
# other ML catalogs, including Denolle-Lab/cascadia_obs_ensemble.
WOOD_ANDERSON = {
    "poles": [-5.49779 - 5.60886j, -5.49779 + 5.60886j],
    "zeros": [0 + 0j],
    "gain": 1.0,
    "sensitivity": 2080,
}

#: Amplitude implementations. "obspy" runs Stream.slice/detrend/remove_response/
#: simulate per strong pick (the seed path). "fast" runs the same arithmetic in
#: numpy with the loop-invariant spectra (evalresp response, pre_filt taper,
#: water-level inverse, WA response, time tapers) memoised per station-day,
#: and falls back to "obspy" per pick for anything it does not mirror.
#: "fused" is "fast" with the velocity prep fused per trace (detrend into a
#: fresh array, taper only the edge samples, obspy's zero-phase highpass as
#: separate forward/backward sosfilt tasks on a small thread pool, sensitivity
#: divided into the scalar peaks); the picker also overlaps its two passes.
AMP_IMPLS = ("fast", "obspy", "fused")
#: Where the fast/fused passes take their scipy.signal and obspy.signal functions from.
#: "scipy" imports those packages (scipy.signal's __init__, and obspy.signal's, which pulls in
#: PPSD and matplotlib). "slim" uses the verbatim copies in slim_signal / slim_evalresp, scipy's
#: own sosfilt kernel loaded from its extension file and a table of the Butterworth highpass
#: sections, so neither package is executed (pick_workload QS_STARTUP2=v2; fused arm only). A slim miss (a corner
#: outside the table, a float32 detrend, the kernel or libevresp failing to load) uses scipy or
#: obspy for that call, and is counted in `slim_counters`.
AMP_SIGNALS = ("scipy", "slim")


class _Fallback(Exception):
    """The fast path cannot reproduce obspy for this pick; use obspy."""


def _import_fused_modules() -> None:
    """Import what the fused arm's two passes import lazily, once."""
    import scipy.fft  # noqa: F401
    import scipy.signal  # noqa: F401
    from obspy.core.inventory import PolynomialResponseStage  # noqa: F401
    from obspy.core.util.base import _get_function_from_entry_point
    from obspy.signal import invsim, util  # noqa: F401
    _get_function_from_entry_point("taper", "cosine")


def _import_slim_modules() -> None:
    """Import what the passes need on the "slim" signal arm: no scipy.signal, no obspy.signal."""
    import scipy.fft  # noqa: F401
    from obspy.core.inventory import PolynomialResponseStage  # noqa: F401

    from . import slim_evalresp, slim_signal  # noqa: F401
    slim_signal.load()


def _linear_detrend(data: np.ndarray) -> np.ndarray:
    """What ``Trace.detrend("linear")`` returns (scipy's least-squares line),
    in closed form: the same fitted line without building an (npts, 2) design
    matrix. float32 keeps scipy, which computes in float32 as obspy does."""
    if data.dtype == np.float32:
        import scipy.signal
        return scipy.signal.detrend(data, type="linear")
    out = np.array(data, dtype=np.float64)          # always a fresh array
    n = out.shape[-1]
    xc = np.arange(n, dtype=np.float64) - (n - 1) / 2.0
    ym = out.mean(axis=-1, keepdims=True)
    slope = (out @ xc)[..., None] / float(xc @ xc)
    out -= ym
    if out.ndim == 1:
        xc *= slope[0]                  # day-long trace: no extra temporary
        out -= xc
    else:
        out -= slope * xc
    return out


def _slice_like_obspy(traces, starttime, endtime) -> list:
    """``Stream.slice(starttime, endtime)`` (nearest_sample, no padding, empty
    traces dropped) as (trace, data view, new starttime) without copying stats
    or data. Mirrors Stream.slice -> Trace.trim -> _ltrim/_rtrim step by step,
    with the same UTCDateTime arithmetic, so every sample index agrees."""
    from obspy.core.compatibility import round_away

    if not traces:
        return []
    first = traces[0].stats
    delta = round_away((starttime - first.starttime) * first.sampling_rate)
    starttime = first.starttime + delta * first.delta
    delta = round_away((endtime - first.endtime) * first.sampling_rate)
    endtime = first.endtime + delta * first.delta
    if starttime > endtime:
        raise _Fallback("startime is larger than endtime")
    out = []
    for tr in traces:
        stats = tr.stats
        data = tr.data
        sr = stats.sampling_rate
        dt = stats.delta
        start = stats.starttime
        # _ltrim
        d = int(round_away((starttime - start) * sr))
        if d > 0:
            start = start + d * dt
            # obspy moves starttime first, so this endtime is the new start
            # plus the untrimmed length.
            if starttime > _endtime(start, len(data), sr):
                data = data[:0]
            else:
                data = data[d:]
        npts = len(data)
        # _rtrim
        d = int(round_away((endtime - start) * sr) - npts + 1)
        if d < 0:
            if endtime < start:
                data = data[:0]
            else:
                total = len(data) - abs(d)
                if endtime == start:
                    total = 1
                data = data[:total]
        if len(data) == 0:
            continue
        out.append((tr, data, start))
    return out


def _endtime(start, npts: int, sampling_rate: float):
    """Stats.endtime for a trace of ``npts`` samples starting at ``start``."""
    delta = 1.0 / float(sampling_rate)
    timediff = 0 if npts == 0 else float(npts - 1) * delta
    return obspy.UTCDateTime(ns=start.ns + int(round(timediff * 1e9)))


class AmplitudeExtractor:
    """
    Extracts trace amplitudes from a set of picks.
    The model extracts the average peak amplitude over available components listed.

    :param time_before: Time before pick in seconds to include in search window for peak.
    :param time_after: Time after pick in seconds to include in search window for peak.
    The response is removed once per station over the whole stream, and each
    pick's window is then sliced from the result - not one deconvolution per
    pick, which is what this did before and which cost ~6,700 deconvolutions on
    a single busy station-day.

    :param time_before: Time before pick in seconds to include in search window for peak.
    :param time_after: Time after pick in seconds to include in search window for peak.
    :param slack: Unused since the deconvolution was hoisted out of the per-pick
                  loop; kept so existing call sites keep working.
    :param response_removal_args: Additional arguments for removing the response.
                                  Passed directly to obspy's `remove_response` function.
                                  By default, uses `{"water_level": 20, "pre_filt": [0.1, 0.2, 40, 45]}`.
    :param components: Components to take into account.
    :param parallel: Ignored. Parallelism belongs at the worker-process level;
                     joblib nested inside each process oversubscribed the box.
    :param raw_highpass: Corner frequency in Hz of the high-pass filter applied
                         before measuring the raw amplitude. Suppresses microseism
                         and long-period noise that would mask small events.
                         Set to None to measure on unfiltered counts.
    :param taper_seconds: Length of the taper applied at each end of a trace
                          before deconvolution. Expressed in seconds rather than
                          as a fraction because obspy's 5% default is 72 minutes
                          on a day-long trace, which would null every pick near
                          a day boundary.
    """

    def __init__(
        self,
        time_before: float = 3,
        time_after: float = 10,
        slack: float = 10,
        response_removal_args: Optional[dict] = None,
        components: str = "NE12",
        parallel: bool = False,
        raw_highpass: Optional[float] = 1.0,
        taper_seconds: float = 60.0,
        wa_min_conf: float = 0.5,
        wa_window_seconds: float = 70.0,
        vel_highpass: float = 0.2,
        impl: str = "obspy",
        workers: Optional[int] = None,
        signal: str = "scipy",
    ):
        if impl not in AMP_IMPLS:
            raise ValueError(f"impl must be one of {AMP_IMPLS}, not {impl!r}")
        if signal not in AMP_SIGNALS:
            raise ValueError(f"signal must be one of {AMP_SIGNALS}, not {signal!r}")
        self.impl = impl
        #: The signal arm asked for, and the one in use: "slim" falls back to "scipy" (counted)
        #: if its modules cannot be loaded.
        self.signal = signal
        self._slim = signal == "slim" and impl == "fused"
        #: Slim arm: SOS table hits and misses, float32 detrends sent to scipy, load failures,
        #: and evalresp calls answered by the vendored copy.
        self.slim_counters = {"sos_hits": 0, "sos_misses": 0, "detrend_f32": 0,
                              "load_failures": 0, "evalresp": 0}
        # Fast-path counters, cumulative over the extractor's life: spectrum
        # cache hits/misses, evalresp evaluations, per-pick obspy fallbacks.
        self.wa_counters = defaultdict(int)
        # Fused velocity counters: station passes, traces filtered, masked
        # stations sent back to the obspy prep.
        self.vel_counters = defaultdict(int)
        # Size of the fused arm's filter pool; None = the process's torch
        # intra-op thread count (the worker's core budget), resolved lazily.
        self._workers = workers
        self._pool: Optional[ThreadPoolExecutor] = None
        #: (time, from, to) per filter-pool resize (set_workers; tail widening).
        self.width_events: list[tuple[float, int, int]] = []
        # The fused arm runs both passes on two threads at once, and their lazy
        # imports race: a first import of scipy from two threads can see a
        # partially initialised module. So the fused modules are imported once,
        # by whichever comes first: warm_up() (in place), start_warm_up() (on a
        # thread of its own), or the first pass. Both passes wait for it.
        self._warm_up: Optional[Future] = None
        self._warm_up_lock = threading.Lock()
        #: "eager", "background" or "first pass": who started the fused imports.
        self.warm_up_mode: Optional[str] = None
        self.time_before = time_before
        self.time_after = time_after
        self.slack = slack
        self.components = components
        self.raw_highpass = raw_highpass
        self.taper_seconds = taper_seconds
        # Only picks at or above this confidence get a Wood-Anderson amplitude.
        # The deconvolution is the dominant cost in the whole pipeline and the
        # marginal end of the catalogue is not where magnitudes are wanted: the
        # picking threshold is 0.2, and 0.5 keeps ~16% of picks.
        self.wa_min_conf = wa_min_conf
        # Total length of the window deconvolved around each qualifying pick.
        # Must comfortably exceed time_before + time_after plus both tapers:
        # at 70 s the 15% taper is 10.5 s a side, leaving a 49 s core for a
        # 13 s measurement window.
        self.wa_window_seconds = wa_window_seconds
        self.vel_highpass = vel_highpass
        if parallel:
            # Retained for call-site compatibility. The per-pick work is now a
            # slice and a maximum, so there is nothing worth a joblib worker -
            # and nesting Parallel(n_jobs=-1) inside each worker process
            # oversubscribed every core on the box.
            logger.debug("`parallel` is ignored; parallelism belongs at the process level")

        if response_removal_args is None:
            self.response_removal_args = {
                "water_level": 20,
                "pre_filt": [0.1, 0.2, 40, 45],
            }
        else:
            self.response_removal_args = response_removal_args

        # The pre_filt low corner no longer has to fit inside a 33 s window: the
        # deconvolution now runs on whatever span the stream covers, typically a
        # full day. The old guard checked `time_before + time_after + 2 * slack`
        # and would fire spuriously here. What remains worth checking is that the
        # *measurement* window can hold the periods being measured, since a
        # 13 s window cannot represent a 50 s period however well the trace was
        # deconvolved.
        pre_filt = self.response_removal_args.get("pre_filt")
        if pre_filt:
            measurement = self.time_before + self.time_after
            longest_period = 1.0 / pre_filt[0]
            if measurement < longest_period:
                warnings.warn(
                    f"Measurement window is {measurement:g} s but pre_filt passes "
                    f"periods up to {longest_period:g} s; the peak will be taken "
                    f"over less than one cycle. Raise time_before/time_after or "
                    f"the pre_filt low corner.",
                    stacklevel=2,
                )

    def warm_up(self) -> None:
        """Import the fused arm's modules now, in this thread (a no-op for the
        other arms, or once started)."""
        self._start_warm_up("eager")
        if self._warm_up is not None:
            self._warm_up.result()

    def start_warm_up(self) -> None:
        """Import the fused arm's modules on a thread of their own, once (a no-op
        for the other arms, or once started).

        The lean picker calls this as its first classify starts: scipy.signal
        and obspy.signal (with matplotlib) are about 0.7 s of imports that no
        pass needs until that classify's picks exist, and the forward runs
        mostly without the GIL.
        """
        self._start_warm_up("background")

    def _start_warm_up(self, mode: str) -> None:
        if self.impl != "fused":
            return
        with self._warm_up_lock:
            if self._warm_up is not None:
                return
            self._warm_up = future = Future()
            self.warm_up_mode = mode

        def run():
            try:
                if self._slim:
                    try:
                        _import_slim_modules()
                    except (ImportError, OSError, AttributeError) as exc:
                        # Not silent: count it, say it, and run the scipy/obspy imports.
                        self.slim_counters["load_failures"] += 1
                        self._slim = False
                        logger.warning(f"slim signal modules failed to load ({exc!r}); "
                                       f"using scipy.signal/obspy.signal")
                        _import_fused_modules()
                else:
                    _import_fused_modules()
            except BaseException as exc:
                future.set_exception(exc)
            else:
                from .profiling import timeline
                timeline.mark("amp_imports_done")
                future.set_result(None)

        if mode == "background":
            threading.Thread(target=run, name="amp-imports", daemon=True).start()
        else:
            run()

    def _await_warm_up(self) -> None:
        """Block until the fused imports are done, running them here if nobody started them."""
        if self.impl == "fused":
            self._start_warm_up("first pass")
            self._warm_up.result()

    @property
    def signal_in_use(self) -> str:
        """"slim" or "scipy": the arm the passes actually take."""
        return "slim" if self._slim else "scipy"

    def _detrend(self, data: np.ndarray) -> np.ndarray:
        """`_linear_detrend`, counting the float32 case the slim arm leaves to scipy.signal."""
        if self._slim and data.dtype == np.float32:
            self.slim_counters["detrend_f32"] += 1
        return _linear_detrend(data)

    @property
    def workers(self) -> int:
        """Filter-pool size for the fused velocity pass."""
        if self._workers is None:
            torch = sys.modules.get("torch")
            self._workers = max(1, torch.get_num_threads()) if torch else 1
        return self._workers

    def set_workers(self, workers: int) -> None:
        """Resize the fused arm's filter pool to `workers` (tail widening).

        Called between station-days, from the thread that starts the passes,
        never while a pass is running: the old pool is shut down without
        waiting (it is idle; its threads exit) and the next pass creates one
        of the new size. A no-op when the size is unchanged.
        """
        workers = max(1, int(workers))
        if workers == self.workers:
            return
        self.width_events.append((time.time(), self.workers, workers))
        self._workers = workers
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None

    def _filter_pool(self) -> ThreadPoolExecutor:
        """One pool per extractor, created on first use; see close()."""
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=self.workers,
                                            thread_name_prefix="amp-vel")
        return self._pool

    def close(self) -> None:
        """Shut the fused arm's filter pool down (a no-op for other arms)."""
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    def _stations_with_picks(self, picks: sbu.PickList) -> dict[str, list[int]]:
        """Pick indices grouped by NET.STA, so the response is removed once per
        station rather than once per pick."""
        groups = defaultdict(list)
        for i, pick in enumerate(picks):
            parts = pick.trace_id.split(".")
            groups[f"{parts[0]}.{parts[1]}"].append(i)
        return groups

    def _taper_fraction(self, stream: obspy.Stream) -> float:
        """Taper a fixed number of seconds, not a fixed fraction.

        obspy tapers 5% by default, which on a day-long trace is 72 minutes at
        each end and would silently null the amplitude of every pick near a day
        boundary. Scale the fraction so the taper is always about
        ``taper_seconds`` long.
        """
        longest = max((tr.stats.endtime - tr.stats.starttime for tr in stream),
                      default=0.0)
        if longest <= 0:
            return 0.05
        return float(min(0.05, self.taper_seconds / longest))

    @staticmethod
    def _taper_length(stream: obspy.Stream, fraction: float) -> float:
        """Seconds tapered at each end, for the longest trace. Used to reject
        picks that fall inside the taper rather than measure a suppressed peak."""
        longest = max((tr.stats.endtime - tr.stats.starttime for tr in stream),
                      default=0.0)
        return float(fraction * longest)

    @staticmethod
    def _fit_to_nyquist(stream, pre_filt=None, highpass=None):
        """Clamp filter corners below the Nyquist of the SLOWEST trace present.

        The corners here were chosen for 100 Hz data, which is what the picker
        sees - `downsample_to_target` brings anything faster down to it. But
        response removal runs on the trace as recorded, and a stream can hold a
        channel far below that: NN and SN station-days on 2026-09-07 carried a
        6 Hz record beside 100 Hz ones. obspy then raises

            ValueError: Selected corner frequency is above Nyquist.

        which cost the station-day. Scaling the corners keeps the measurement
        on the band the data can actually support instead of discarding it.
        Returns (pre_filt, highpass), either possibly adjusted, or None for a
        stream with nothing usable.
        """
        rates = [tr.stats.sampling_rate for tr in stream
                 if tr.stats.sampling_rate and tr.stats.npts]
        return AmplitudeExtractor._fit_rates_to_nyquist(rates, pre_filt, highpass)

    @staticmethod
    def _fit_rates_to_nyquist(rates, pre_filt=None, highpass=None):
        """`_fit_to_nyquist` on the sampling rates of the traces present."""
        if not rates:
            return pre_filt, highpass
        nyq = min(rates) / 2.0
        # Stay clear of the corner itself; a filter AT Nyquist is unstable.
        ceiling = nyq * 0.9
        if pre_filt is not None and pre_filt[3] > ceiling:
            lo1, lo2 = pre_filt[0], pre_filt[1]
            hi2 = ceiling
            hi1 = max(lo2 * 1.5, hi2 * 0.9)
            pre_filt = [lo1, lo2, hi1, hi2] if hi1 < hi2 else None
        if highpass is not None and highpass >= ceiling:
            highpass = None          # nothing sensible left to high-pass
        return pre_filt, highpass

    def extract_amplitudes(
        self, stream: obspy.Stream, picks: sbu.PickList, inventory: obspy.Inventory
    ) -> list[float]:
        """
        Extract Wood-Anderson amplitudes from the horizontal components.
        Returns NaN for every pick below `wa_min_conf` or where no amplitude
        could be determined.

        The response is removed on a short window around each qualifying pick,
        not on the day-long trace. Deconvolving the whole day cost ~14 s per
        station-day-channel and was **71-89% of total campaign wall clock**
        (measured, `docs/rerun_2026/22_amplitude_profile.md`) - five times the
        cost of everything else in the pipeline combined.

        A short window is sound here specifically because Wood-Anderson is
        window-insensitive: measured against the day-long deconvolution on
        CI.BAR 2010.166, the ratio is median 1.0000, p10 0.999, p90 1.001, 97%
        within 5%. `docs/amplitude_conventions.md` records why this holds for WA
        and *not* for displacement, where the same substitution shifts
        amplitudes by 12x. Do not reuse this window for a longer-period measure.

        Cost scales with pick count rather than samples: ~20 ms per pick against
        a flat ~14 s per station-day, so short windows win below ~680 picks.
        Gating at `wa_min_conf` keeps the count well under that - the median
        station-day has 120 picks at conf >= 0.5 against 940 at all confidences.
        """
        self._await_warm_up()
        if len(picks) == 0:
            return []                       # never deconvolve a day with no picks

        stream = stream.select(channel=f"*[{self.components}]")
        amplitudes: list[float] = [np.nan] * len(picks)
        half = self.wa_window_seconds / 2.0

        for station, indices in self._stations_with_picks(picks).items():
            net, sta = station.split(".")
            strong = [i for i in indices
                      if float(picks[i].peak_value) >= self.wa_min_conf]
            if not strong:
                continue
            sub_inv = inventory.select(network=net, station=sta)
            station_stream = stream.select(network=net, station=sta)
            if len(station_stream) == 0:
                continue

            spectra: dict = {}              # per station-day; dropped on return
            for i in strong:
                if self.impl in ("fast", "fused"):
                    try:
                        amplitudes[i] = self._wa_pick_fast(
                            station_stream, sub_inv, picks[i], half, spectra)
                        continue
                    except _Fallback:
                        self.wa_counters["fallbacks"] += 1
                amplitudes[i] = self._wa_pick_obspy(
                    station_stream, sub_inv, picks[i], half)

        return amplitudes

    # A fixed fraction is right here: every window is the same length, so 15%
    # is a fixed number of seconds, and the guard in _peak_from_arrays rejects
    # any pick whose measurement window reaches into it rather than measuring a
    # suppressed peak.
    _WA_TAPER_FRACTION = 0.15

    def _wa_pick_obspy(self, station_stream, sub_inv, pick, half) -> float:
        """One strong pick's Wood-Anderson amplitude, the obspy way."""
        t = pick.peak_time
        window = station_stream.slice(t - half, t + half).copy()
        if len(window) == 0:
            return np.nan
        window.detrend("linear")
        fraction = self._WA_TAPER_FRACTION
        args = dict(self.response_removal_args)
        pf, _ = self._fit_to_nyquist(window, args.get("pre_filt"))
        if args.get("pre_filt") is not None:
            if pf is None:
                return np.nan               # no band left below Nyquist
            args["pre_filt"] = pf
        try:
            window.remove_response(
                sub_inv,
                taper_fraction=fraction,
                **args,
            )
        except Exception:                   # no response information
            return np.nan
        window.simulate(paz_simulate=WOOD_ANDERSON)
        arrays = self._as_arrays(
            window, self._taper_length(window, fraction))
        return self._peak_from_arrays(arrays, pick, mean=True)

    def _wa_pick_fast(self, station_stream, sub_inv, pick, half, spectra) -> float:
        """`_wa_pick_obspy` in numpy, step for step:

        remove_response: float64, demean, SAC cosine taper (15%), rfft at
        obspy's nfft, x pre_filt taper, x water-level-inverted evalresp VEL
        response, Nyquist bin made real, irfft, trim. simulate(WA): demean,
        5% half-cosine taper, rfft, x WA paz response, Nyquist bin real,
        irfft, trim, simple detrend, x sensitivity.

        Everything that depends only on (response, delta, npts, nfft, pre_filt)
        is computed once per station-day in ``spectra``. Traces of one window
        with the same length and rate are transformed together. Raises
        _Fallback for anything this does not mirror exactly.
        """
        import scipy.fft
        from obspy.core.inventory import PolynomialResponseStage
        slim = self._slim
        if slim:
            from . import slim_evalresp
            from .slim_signal import (_npts2nfft, cosine_sac_taper, cosine_taper,
                                      invert_spectrum, paz_to_freq_resp)
        else:
            from obspy.signal.invsim import (cosine_sac_taper, cosine_taper,
                                             invert_spectrum, paz_to_freq_resp)
            from obspy.signal.util import _npts2nfft

        t = pick.peak_time
        window = _slice_like_obspy(station_stream.traces, t - half, t + half)
        if not window:
            return np.nan
        if any(np.ma.isMaskedArray(data) for _, data, _ in window):
            raise _Fallback("masked data")
        fraction = self._WA_TAPER_FRACTION
        args = self.response_removal_args
        if set(args) - {"water_level", "pre_filt"}:
            raise _Fallback("evalresp kwargs")
        water_level = args.get("water_level", 60)
        pre_filt = args.get("pre_filt")
        rates = [tr.stats.sampling_rate for tr, _, _ in window
                 if tr.stats.sampling_rate]
        pf, _ = self._fit_rates_to_nyquist(rates, pre_filt)
        if pre_filt is not None:
            if pf is None:
                return np.nan
            pre_filt = pf

        # Stream.remove_response stops at the first trace without a response,
        # and the pick is then NaN.
        responses = []
        for tr, _, start in window:
            try:
                response = sub_inv.get_response(tr.id, start)
            except Exception:
                return np.nan
            if (not response.response_stages
                    or isinstance(response.response_stages[0],
                                  PolynomialResponseStage)):
                raise _Fallback("polynomial response")
            responses.append(response)

        groups: dict = defaultdict(list)
        for k, (tr, data, _) in enumerate(window):
            groups[(len(data), tr.stats.delta, data.dtype)].append(k)

        out: list = [None] * len(window)
        # Any error in this arithmetic (e.g. a window too short for the
        # tapers) goes back to obspy, which decides NaN or raises as before.
        try:
            for (npts, delta, _), rows in groups.items():
                nfft = _npts2nfft(npts)
                tkey = ("time", npts)
                tapers = spectra.get(tkey)
                if tapers is None:
                    tapers = (cosine_taper(npts, fraction, sactaper=True,
                                           halfcosine=False),
                              cosine_taper(npts, 0.05))
                    spectra[tkey] = tapers
                wkey = ("wa", delta, nfft)
                wa = spectra.get(wkey)
                if wa is None:
                    wa = paz_to_freq_resp(WOOD_ANDERSON["poles"],
                                          WOOD_ANDERSON["zeros"],
                                          WOOD_ANDERSON["gain"], delta, nfft)
                    spectra[wkey] = wa
                fd_taper, inverse = [], []
                for k in rows:
                    rkey = (id(responses[k]), delta, npts, nfft,
                            tuple(pre_filt) if pre_filt else None, water_level)
                    entry = spectra.get(rkey)
                    if entry is None:
                        self.wa_counters["misses"] += 1
                        self.wa_counters["evalresp"] += 1
                        try:
                            if slim:
                                self.slim_counters["evalresp"] += 1
                                freq_response, freqs = slim_evalresp.get_evalresp_response(
                                    responses[k], delta, nfft, output="VEL")
                            else:
                                freq_response, freqs = responses[k].get_evalresp_response(
                                    delta, nfft, output="VEL")
                        except Exception:
                            raise _Fallback("evalresp")
                        taper = (cosine_sac_taper(freqs, flimit=pre_filt)
                                 if pre_filt else None)
                        if water_level is None:
                            freq_response[0] = 0.0
                            freq_response[1:] = 1.0 / freq_response[1:]
                        else:
                            invert_spectrum(freq_response, water_level)
                        # Hold the Response so its id() cannot be reused.
                        entry = (responses[k], taper, freq_response)
                        spectra[rkey] = entry
                    else:
                        self.wa_counters["hits"] += 1
                    fd_taper.append(entry[1])
                    inverse.append(entry[2])

                # detrend("linear") on the window, then remove_response.
                data = self._detrend(np.stack([window[k][1] for k in rows]))
                data = np.asarray(data, dtype=np.float64)
                data -= data.mean(axis=-1, keepdims=True)
                data *= tapers[0]
                spec = scipy.fft.rfft(data, n=nfft, axis=-1)
                if pre_filt:
                    spec *= np.stack(fd_taper)
                spec *= np.stack(inverse)
                spec[:, -1] = np.abs(spec[:, -1]) + 0.0j
                data = scipy.fft.irfft(spec, n=nfft, axis=-1)[:, :npts]
                # simulate(paz_simulate=WOOD_ANDERSON), simulate_seismometer defaults.
                data -= data.mean(axis=-1, keepdims=True)
                data *= tapers[1]
                spec = scipy.fft.rfft(data, n=nfft, axis=-1)
                spec *= wa
                spec[:, -1] = np.abs(spec[:, -1]) + 0.0j
                data = scipy.fft.irfft(spec, n=nfft, axis=-1)[:, :npts]
                x1 = data[:, :1]
                data -= x1 + np.arange(npts) * (data[:, -1:] - x1) / float(npts - 1)
                data *= WOOD_ANDERSON["sensitivity"]
                for row, k in enumerate(rows):
                    out[k] = data[row]
        except _Fallback:
            raise
        except Exception as exc:
            raise _Fallback(repr(exc)) from exc

        # _as_arrays(window, _taper_length(window, fraction))
        ends = [_endtime(start, len(data), tr.stats.sampling_rate)
                for tr, data, start in window]
        longest = max(end - start for end, (_, _, start) in zip(ends, window))
        taper_len = float(fraction * longest)
        arrays = []
        for (tr, _, start), end, data in zip(window, ends, out):
            s = start.timestamp
            arrays.append((tr.id[-1], data, s, float(tr.stats.sampling_rate),
                           s + taper_len, end.timestamp - taper_len))
        return self._peak_from_arrays(arrays, pick, mean=True)

    def extract_velocity_amplitudes(
        self, stream: obspy.Stream, picks: sbu.PickList, inventory: obspy.Inventory
    ) -> list[float]:
        """
        Peak ground velocity near each pick, in m/s, for **every** pick.

        This is the cheap amplitude that replaces raw counts. Rather than
        deconvolving the full response, it divides by the scalar instrument
        sensitivity and high-passes at `vel_highpass` Hz - one filter and one
        divide over the day, no FFT of the transfer function. Measured at 0.95 s
        per station-day-channel against 13.5 s for the day-long deconvolution,
        a 14x saving, and unlike raw counts the number is physical and
        comparable between instruments.

        The scalar gain is flat-response only, so this is valid in the band
        where the instrument is flat to velocity and NOT below `vel_highpass`.
        It is a triage and QC amplitude; `amp` remains the calibrated one.
        Returns NaN where no response or no data is available.
        """
        self._await_warm_up()
        if len(picks) == 0:
            return []

        amplitudes: list[float] = [np.nan] * len(picks)

        for station, indices in self._stations_with_picks(picks).items():
            net, sta = station.split(".")
            selected = stream.select(network=net, station=sta)
            if len(selected) == 0:
                continue
            if self.impl == "fused":
                if not any(np.ma.isMaskedArray(tr.data) for tr in selected):
                    arrays = self._fused_velocity_arrays(
                        selected, inventory.select(network=net, station=sta))
                    for i in indices:
                        amplitudes[i] = self._scaled_peak_from_arrays(
                            arrays, picks[i])
                    continue
                self.vel_counters["masked_fallbacks"] += 1
            if self.impl == "fast" and not any(
                    np.ma.isMaskedArray(tr.data) for tr in selected):
                # detrend("linear") in closed form, straight into a new array:
                # scipy's least-squares solve over a day-long trace was ~2/3 of
                # this pass, and the int32 copy before it was thrown away.
                prepared = obspy.Stream([
                    obspy.Trace(data=_linear_detrend(tr.data) if len(tr.data)
                                else tr.data.copy(), header=tr.stats.copy())
                    for tr in selected])
            else:
                prepared = selected.copy()
                prepared.detrend("linear")
            fraction = self._taper_fraction(prepared)
            prepared.taper(max_percentage=fraction, type="cosine")
            _, hp = self._fit_to_nyquist(prepared, highpass=self.vel_highpass)
            if hp is not None:
                prepared.filter("highpass", freq=hp, corners=4, zerophase=True)
            # Per-trace sensitivity: components of one station can differ, and a
            # station's response changes across epochs.
            sub_inv = inventory.select(network=net, station=sta)
            kept = obspy.Stream()
            for tr in prepared:
                try:
                    sens = (sub_inv
                            .get_response(tr.id, tr.stats.starttime)
                            .instrument_sensitivity.value)
                except Exception:
                    continue                # no response for this component
                if not sens:
                    continue
                tr.data = tr.data / float(sens)
                kept += tr
            if len(kept) == 0:
                continue
            arrays = self._as_arrays(kept, self._taper_length(kept, fraction))
            for i in indices:
                amplitudes[i] = self._peak_from_arrays(arrays, picks[i], mean=False)

        return amplitudes

    def _fused_velocity_arrays(self, selected: obspy.Stream,
                               sub_inv: obspy.Inventory) -> list[tuple]:
        """The fast arm's velocity prep for one station's unmasked traces,
        without the day-long copies it does not need, as `_as_arrays` rows
        plus each trace's sensitivity.

        Per trace, bit for bit what the fast arm computes: `_linear_detrend`
        into a fresh array, `Trace.taper(max_percentage, "cosine")` applied to
        the edge samples only (the window is exactly 1.0 in between), and
        obspy's `highpass(zerophase=True)`: the same iirfilter sos, sosfilt,
        then sosfilt of the flipped result flipped back. The two sosfilt
        passes run as separate tasks on the filter pool (sosfilt drops the
        GIL), so a third channel's forward pass overlaps the first two
        channels' backward passes. The divide by sensitivity is left to the
        scalar peak (`_scaled_peak_from_arrays`), which is exact because
        rounding a quotient is monotonic in the dividend.
        """
        if self._slim:
            from . import slim_signal

            cosine = slim_signal.cosine_taper
            sosfilt = slim_signal.sosfilt
            counters = self.slim_counters

            def highpass_sos(wn):
                sos = slim_signal.butter_highpass_sos(wn)
                if sos is not None:
                    counters["sos_hits"] += 1
                    return sos
                counters["sos_misses"] += 1
                from scipy.signal import iirfilter
                return iirfilter(4, wn, rp=None, rs=None, btype="highpass",
                                 ftype="butter", output="sos")
        else:
            from obspy.core.util.base import _get_function_from_entry_point
            from scipy.signal import iirfilter, sosfilt

            cosine = _get_function_from_entry_point("taper", "cosine")

            def highpass_sos(wn):
                return iirfilter(4, wn, rp=None, rs=None, btype="highpass",
                                 ftype="butter", output="sos")

        fraction = self._taper_fraction(selected)
        _, hp = self._fit_to_nyquist(selected, highpass=self.vel_highpass)
        # Traces without a usable sensitivity are dropped as the fast arm
        # drops them, but before any filtering.
        kept = []
        for tr in selected:
            try:
                sens = (sub_inv
                        .get_response(tr.id, tr.stats.starttime)
                        .instrument_sensitivity.value)
            except Exception:
                continue                    # no response for this component
            if not sens:
                continue
            kept.append((tr, float(sens)))
        if not kept:
            return []

        def forward(tr):
            data = (self._detrend(tr.data) if len(tr.data)
                    else tr.data.copy())
            npts = len(data)
            if npts:
                # Trace.taper: half-length, taper_sides, then data *= taper.
                wlen = min(int(fraction * npts), int(npts / 2))
                sides = cosine(2 * wlen if 2 * wlen == npts else 2 * wlen + 1,
                               p=1.0)
                if wlen:
                    data[:wlen] *= sides[:wlen]
                    data[npts - wlen:] *= sides[len(sides) - wlen:]
            if hp is None or not npts:
                return data, None
            fe = 0.5 * tr.stats.sampling_rate
            sos = highpass_sos(hp / fe)
            return np.flip(sosfilt(sos, data, axis=-1), axis=-1), sos

        def backward(first, sos):
            return np.flip(sosfilt(sos, first, axis=-1), axis=-1)

        pool = self._filter_pool()
        out: list = [None] * len(kept)
        pending = {pool.submit(forward, tr): k for k, (tr, _) in enumerate(kept)}
        second = {}
        for fut in as_completed(pending):
            k = pending[fut]
            data, sos = fut.result()
            if sos is None:
                out[k] = data
            else:
                second[pool.submit(backward, data, sos)] = k
                self.vel_counters["traces_filtered"] += 1
        for fut, k in second.items():
            out[k] = fut.result()
        self.vel_counters["stations"] += 1

        # _as_arrays(kept, _taper_length(kept, fraction)), plus sensitivity.
        taper_len = self._taper_length([tr for tr, _ in kept], fraction)
        arrays = []
        for (tr, sens), data in zip(kept, out):
            if len(data) == 0:
                continue
            start = tr.stats.starttime.timestamp
            arrays.append((tr.id[-1], data, start, float(tr.stats.sampling_rate),
                           start + taper_len,
                           tr.stats.endtime.timestamp - taper_len, sens))
        return arrays

    def _scaled_peak_from_arrays(self, arrays: list[tuple], pick: sbu.Pick):
        """`_peak_from_arrays(mean=False)` on data that is still in counts:
        each trace's window peak is divided by its sensitivity as a scalar.
        max|x / s| == max|x| / |s| exactly, in the data's own dtype."""
        t0 = pick.peak_time.timestamp - self.time_before
        t1 = pick.peak_time.timestamp + self.time_after

        peaks: dict[str, float] = {}
        for comp, data, start, rate, core_start, core_end, sens in arrays:
            if t0 < core_start or t1 > core_end:
                continue
            i0 = int(round((t0 - start) * rate))
            i1 = int(round((t1 - start) * rate)) + 1
            if i0 < 0 or i1 > len(data) or i1 <= i0:
                continue
            val = float(np.max(np.abs(data[i0:i1])) / abs(sens))
            peaks[comp] = max(peaks.get(comp, 0.0), val)

        if not peaks:
            return np.nan
        return float(np.max(list(peaks.values())))

    def extract_raw_amplitudes(
        self, stream: obspy.Stream, picks: sbu.PickList
    ) -> list[float]:
        """
        Extract peak raw amplitudes around each pick, P and S alike, in a
        window of time_before/time_after seconds around the pick peak.

        The data is kept in raw counts (no response removal), but high-passed at
        raw_highpass Hz (default 1 Hz). As above, the filter is applied once per
        station over the whole stream rather than once per pick, and the
        per-pick work is a slice and a maximum. All components are considered
        and the maximum over components is returned. Returns NaN for every pick
        where no data is available.
        """
        if len(picks) == 0:
            return []

        amplitudes: list[float] = [np.nan] * len(picks)

        for station, indices in self._stations_with_picks(picks).items():
            net, sta = station.split(".")
            prepared = stream.select(network=net, station=sta).copy()
            if len(prepared) == 0:
                continue
            prepared.detrend("linear")
            taper_len = 0.0
            if self.raw_highpass is not None:
                fraction = self._taper_fraction(prepared)
                prepared.taper(max_percentage=fraction, type="cosine")
                _, hp = self._fit_to_nyquist(prepared, highpass=self.raw_highpass)
                if hp is not None:
                    prepared.filter("highpass", freq=hp, corners=4,
                                    zerophase=True)
                taper_len = self._taper_length(prepared, fraction)
            arrays = self._as_arrays(prepared, taper_len)

            for i in indices:
                amplitudes[i] = self._peak_from_arrays(arrays, picks[i], mean=False)

        return amplitudes

    @staticmethod
    def _as_arrays(prepared: obspy.Stream, taper_len: float) -> list[tuple]:
        """Flatten a stream to plain arrays once, so the per-pick path is numpy
        only.

        Profiling showed amplitude extraction was still the largest stage after
        the deconvolution was hoisted - 3.4-4.1 ms per pick for what should be a
        slice and a maximum. The cost was obspy: `Stream.slice` per pick, and a
        `select()` scan per trace per pick to find the taper bounds. Both are
        loop-invariant, so they are done once here.
        """
        out = []
        for tr in prepared:
            if len(tr.data) == 0:
                continue
            start = tr.stats.starttime.timestamp
            out.append((
                tr.id[-1],                       # component
                tr.data,
                start,
                float(tr.stats.sampling_rate),
                start + taper_len,               # first trustworthy sample
                tr.stats.endtime.timestamp - taper_len,
            ))
        return out

    def _peak_from_arrays(self, arrays: list[tuple], pick: sbu.Pick, mean: bool):
        """Peak amplitude in one pick's window. Pure index arithmetic."""
        t0 = pick.peak_time.timestamp - self.time_before
        t1 = pick.peak_time.timestamp + self.time_after

        peaks: dict[str, float] = {}
        for comp, data, start, rate, core_start, core_end in arrays:
            # Reject rather than measure inside a taper: the signal is driven
            # smoothly to zero there, so a measurement is wrong, not imprecise.
            if t0 < core_start or t1 > core_end:
                continue
            i0 = int(round((t0 - start) * rate))
            i1 = int(round((t1 - start) * rate)) + 1
            if i0 < 0 or i1 > len(data) or i1 <= i0:
                continue
            val = float(np.max(np.abs(data[i0:i1])))
            peaks[comp] = max(peaks.get(comp, 0.0), val)

        if not peaks:
            return np.nan
        values = list(peaks.values())
        return float(np.mean(values) if mean else np.max(values))

    def _peak_in_window(
        self,
        prepared: obspy.Stream,
        pick: sbu.Pick,
        mean: bool = True,
        taper_len: float = 0.0,
    ):
        """Peak amplitude in one pick's measurement window, cut from an already
        detrended, deconvolved (or filtered) stream.

        Picks whose window overlaps a taper are returned as NaN rather than
        measured. The taper drives the signal smoothly to zero, so measuring
        inside it yields an amplitude that is not merely imprecise but wrong -
        by up to 200x for a pick at a trace edge. Missing is honest; suppressed
        is not. This costs about 0.1% of picks on a day-long trace, at the day
        boundaries and beside gaps.
        """
        t_start = pick.peak_time - self.time_before
        t_end = pick.peak_time + self.time_after
        window = prepared.slice(t_start, t_end)

        if len(window) == 0:
            return np.nan

        # Extract peak
        component_peaks = defaultdict(lambda: 0)
        for trace in window:
            if len(trace.data) == 0:
                # A slice across a gap yields empty traces; np.max would raise.
                continue
            source = prepared.select(id=trace.id)
            if taper_len > 0 and source:
                core_start = min(tr.stats.starttime for tr in source) + taper_len
                core_end = max(tr.stats.endtime for tr in source) - taper_len
                if t_start < core_start or t_end > core_end:
                    continue
            val = np.max(
                np.abs(trace.data)
            )  # detrended and deconvolved over the whole stream beforehand
            component_peaks[trace.id[-1]] = max(component_peaks[trace.id[-1]], val)

        if not component_peaks:
            return np.nan

        if mean:
            return np.nanmean(list(component_peaks.values()))
        else:
            return np.nanmax(list(component_peaks.values()))
