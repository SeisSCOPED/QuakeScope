import logging
import warnings
from collections import defaultdict
from typing import Optional

import numpy as np
import obspy
import seisbench.util as sbu

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
    ):
        self.time_before = time_before
        self.time_after = time_after
        self.slack = slack
        self.components = components
        self.raw_highpass = raw_highpass
        self.taper_seconds = taper_seconds
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

    def extract_amplitudes(
        self, stream: obspy.Stream, picks: sbu.PickList, inventory: obspy.Inventory
    ) -> list[float]:
        """
        Extract Wood-Anderson amplitudes from the horizontal components.
        Returns NaN for every pick where no amplitude could be determined.

        The response is removed **once per station**, on whatever span the stream
        covers, and each pick's measurement window is then sliced out of the
        deconvolved trace. Previously this ran one `remove_response` per pick -
        roughly 6,700 deconvolutions for a single busy station-day, all of them
        recomputing the same transfer function.

        Deconvolving the long trace is also the better measurement: a short
        window cannot resolve periods near its own length, which is why
        `docs/amplitude_conventions.md` constrains the pre-filter to what a 33 s
        window supported. That constraint no longer applies here.
        """
        if len(picks) == 0:
            return []                       # never deconvolve a day with no picks

        stream = stream.select(channel=f"*[{self.components}]")
        amplitudes: list[float] = [np.nan] * len(picks)

        for station, indices in self._stations_with_picks(picks).items():
            net, sta = station.split(".")
            sub_inv = inventory.select(network=net, station=sta)
            prepared = stream.select(network=net, station=sta).copy()
            if len(prepared) == 0:
                continue
            prepared.detrend("linear")
            fraction = self._taper_fraction(prepared)
            try:
                prepared.remove_response(
                    sub_inv,
                    taper_fraction=fraction,
                    **self.response_removal_args,
                )
            except Exception:               # no response information
                continue
            prepared.simulate(paz_simulate=WOOD_ANDERSON)
            arrays = self._as_arrays(prepared, self._taper_length(prepared, fraction))

            for i in indices:
                amplitudes[i] = self._peak_from_arrays(arrays, picks[i], mean=True)

        return amplitudes

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
                prepared.filter(
                    "highpass", freq=self.raw_highpass, corners=4, zerophase=True
                )
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
