import warnings
from collections import defaultdict
from typing import Optional

import numpy as np
import obspy
import seisbench.util as sbu
from joblib import Parallel, delayed

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
    :param slack: Additional time in seconds included for removing/simulating response and detrending.
    :param response_removal_args: Additional arguments for removing the response.
                                  Passed directly to obspy's `remove_response` function.
                                  By default, uses `{"water_level": 20, "pre_filt": [0.1, 0.2, 40, 45]}`.
                                  The low corner must be a period the deconvolved
                                  window can resolve - see the note below.
    :param components: Components to take into account.
    :param raw_highpass: Corner frequency in Hz of the high-pass filter applied
                         before measuring the raw amplitude. Suppresses microseism
                         and long-period noise that would mask small events.
                         Set to None to measure on unfiltered counts.
    """

    def __init__(
        self,
        time_before: float = 3,
        time_after: float = 10,
        slack: float = 10,
        response_removal_args: Optional[dict] = None,
        components: str = "NE12",
        parallel: bool = True,
        raw_highpass: Optional[float] = 1.0,
    ):
        self.time_before = time_before
        self.time_after = time_after
        self.slack = slack
        self.components = components
        self.parallel = parallel
        self.raw_highpass = raw_highpass

        if response_removal_args is None:
            self.response_removal_args = {
                "water_level": 20,
                "pre_filt": [0.1, 0.2, 40, 45],
            }
        else:
            self.response_removal_args = response_removal_args

        # The deconvolved window is time_before + time_after + 2 * slack seconds.
        # A pre_filt low corner of f Hz asks the deconvolution to resolve a 1/f
        # second period; below ~3 cycles that returns ringing rather than signal.
        # This is harmless for the Wood-Anderson amplitude itself, because the
        # WA simulation is a sharp bandpass near 1 Hz that discards the
        # long-period noise - it is a trap only if this class is ever reused to
        # measure a displacement (Mw) amplitude, where the error reached 12x in
        # the equivalent Cascadia code. Warn rather than fail.
        pre_filt = self.response_removal_args.get("pre_filt")
        if pre_filt:
            window = self.time_before + self.time_after + 2 * self.slack
            longest_period = 1.0 / pre_filt[0]
            if window < 3 * longest_period:
                warnings.warn(
                    f"Deconvolution window is {window:g} s but pre_filt starts at "
                    f"{pre_filt[0]:g} Hz ({longest_period:g} s period); at least "
                    f"{3 * longest_period:g} s is needed. Raise `slack` or the "
                    f"pre_filt low corner.",
                    stacklevel=2,
                )

    def extract_amplitudes(
        self, stream: obspy.Stream, picks: sbu.PickList, inventory: obspy.Inventory
    ) -> list[float]:
        """
        Extract Wood-Anderson amplitudes from the horizontal components.
        Returns NaN for every pick where no amplitude could be determined.
        """
        stream = stream.select(channel=f"*[{self.components}]")

        amplitudes = []
        for pick in picks:
            # Extract right part of data to reduce unnecessary pickling
            net = pick.trace_id.split(".")[0]
            sta = pick.trace_id.split(".")[1]
            sub_inv = inventory.select(network=net, station=sta)

            large_window = (
                stream.select(network=net, station=sta)
                .slice(
                    pick.peak_time - self.time_before - self.slack,
                    pick.peak_time + self.time_after + self.slack,
                )
                .copy()
            )

            if self.parallel:
                amplitudes.append(
                    delayed(self._extract_single_amplitude)(
                        large_window, pick, sub_inv, mean=True
                    )
                )
            else:
                amplitudes.append(
                    self._extract_single_amplitude(
                        large_window, pick, sub_inv, mean=True
                    )
                )

        if self.parallel:
            amplitudes = Parallel(n_jobs=-1)(amplitudes)

        return amplitudes

    def extract_raw_amplitudes(
        self, stream: obspy.Stream, picks: sbu.PickList
    ) -> list[float]:
        """
        Extract peak raw amplitudes around each pick, P and S alike, in a
        window of time_before/time_after seconds around the pick peak.
        The data is kept in raw counts (no response removal), but high-passed
        at raw_highpass Hz (default 1 Hz). The filter is applied on a window with slack
        seconds of margin on both sides. All components are considered and the maximum
        over components is returned. Returns NaN for every pick where no data is available.
        """
        amplitudes = []
        for pick in picks:
            net = pick.trace_id.split(".")[0]
            sta = pick.trace_id.split(".")[1]
            large_window = (
                stream.select(network=net, station=sta)
                .slice(
                    pick.peak_time - self.time_before - self.slack,
                    pick.peak_time + self.time_after + self.slack,
                )
                .copy()
            )

            if self.parallel:
                amplitudes.append(
                    delayed(self._extract_single_amplitude)(
                        large_window, pick, None, mean=False
                    )
                )
            else:
                amplitudes.append(
                    self._extract_single_amplitude(large_window, pick, None, mean=False)
                )

        if self.parallel:
            amplitudes = Parallel(n_jobs=-1)(amplitudes)

        return amplitudes

    def _extract_single_amplitude(
        self,
        large_window: obspy.Stream,
        pick: sbu.Pick,
        sub_inv: obspy.Inventory,
        mean: bool = True,
    ):
        # normalize window
        large_window.detrend("linear")
        if sub_inv is not None:
            # Remove response and simulate Wood-Anderson
            try:
                large_window.remove_response(sub_inv, **self.response_removal_args)
            except Exception:  # No response information
                return np.nan
            large_window.simulate(paz_simulate=WOOD_ANDERSON)
        elif self.raw_highpass is not None:
            # Use raw counts and high-pass filter
            large_window.taper(max_percentage=0.05, type="cosine")
            large_window.filter(
                "highpass", freq=self.raw_highpass, corners=4, zerophase=True
            )
        else:
            pass

        # Slice window
        window = large_window.slice(
            pick.peak_time - self.time_before,
            pick.peak_time + self.time_after,
        )

        if len(window) == 0:
            return np.nan

        # Extract peak
        component_peaks = defaultdict(lambda: 0)
        for trace in window:
            val = np.max(
                np.abs(trace.data)
            )  # Has been detrended with comparison to larger window before
            component_peaks[trace.id[-1]] = max(component_peaks[trace.id[-1]], val)

        if mean:
            return np.nanmean(list(component_peaks.values()))
        else:
            return np.nanmax(list(component_peaks.values()))
