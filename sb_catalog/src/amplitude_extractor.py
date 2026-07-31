from collections import defaultdict
from typing import Optional

import numpy as np
import obspy
import seisbench.util as sbu
from joblib import Parallel, delayed

WOOD_ANDERSON = {
    "poles": [-6.283 + 4.7124j, -6.283 - 4.7124j],
    "zeros": [0 + 0j],
    "gain": 1.0,
    "sensitivity": 2080,
}


class AmplitudeExtractor:
    """
    Extracts WoodAnderson amplitudes from a set of picks.
    The model extracts the average peak amplitude over all available components listed.
    By default, all horizontal components will be included.

    :param time_before: Time before pick in seconds to include in search window for peak.
    :param time_after: Time after pick in seconds to include in search window for peak.
    :param slack: Additional time in seconds included for removing/simulating response and detrending.
    :param response_removal_args: Additional arguments for removing the response.
                                  Passed directly to obspy's `remove_response` function.
                                  By default, uses `{"water_level": 20, "pre_filt": [0.02, 0.05, 40, 45]}`.
    :param components: Components to take into account.
    :param raw_time_before: Time before pick in seconds for the raw amplitude window.
    :param raw_time_after: Time after pick in seconds for the raw amplitude window.
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
        raw_time_before: float = 2,
        raw_time_after: float = 2,
        raw_highpass: Optional[float] = 1.0,
    ):
        self.time_before = time_before
        self.time_after = time_after
        self.slack = slack
        self.components = components
        self.parallel = parallel
        self.raw_time_before = raw_time_before
        self.raw_time_after = raw_time_after
        self.raw_highpass = raw_highpass

        if response_removal_args is None:
            self.response_removal_args = {
                "water_level": 20,
                "pre_filt": [0.02, 0.05, 40, 45],
            }
        else:
            self.response_removal_args = response_removal_args

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
            sub = stream.select(network=net, station=sta)
            sub_inv = inventory.select(network=net, station=sta)

            large_window = sub.slice(
                pick.peak_time - self.time_before - self.slack,
                pick.peak_time + self.time_after + self.slack,
            ).copy()

            if self.parallel:
                amplitudes.append(
                    delayed(self._extract_single_amplitude)(large_window, pick, sub_inv)
                )
            else:
                amplitudes.append(
                    self._extract_single_amplitude(large_window, pick, sub_inv)
                )

        if self.parallel:
            amplitudes = Parallel(n_jobs=-1)(amplitudes)

        return amplitudes

    def extract_raw_amplitudes(
        self, stream: obspy.Stream, picks: sbu.PickList
    ) -> list[float]:
        """
        Extract peak raw amplitudes around each pick, P and S alike, in a
        window of raw_time_before/raw_time_after seconds around the pick peak.
        The data is kept in raw counts (no response removal), but high-passed
        at raw_highpass Hz (default 1 Hz) so microseism and long-period noise
        do not mask small events. The filter is applied on a window with slack
        seconds of margin on both sides, so edge effects stay outside the
        measurement window. All components are considered and the maximum
        over components is returned.
        Returns NaN for every pick where no data is available.
        """
        amplitudes = []
        for pick in picks:
            net = pick.trace_id.split(".")[0]
            sta = pick.trace_id.split(".")[1]
            large_window = (
                stream.select(network=net, station=sta)
                .slice(
                    pick.peak_time - self.raw_time_before - self.slack,
                    pick.peak_time + self.raw_time_after + self.slack,
                )
                .copy()
            )

            if self.raw_highpass is not None:
                large_window.detrend("linear")
                large_window.taper(max_percentage=0.05, type="cosine")
                large_window.filter(
                    "highpass", freq=self.raw_highpass, corners=4, zerophase=True
                )

            window = large_window.slice(
                pick.peak_time - self.raw_time_before,
                pick.peak_time + self.raw_time_after,
            )

            peak = np.nan
            for trace in window:
                if len(trace.data) == 0:
                    continue
                val = np.max(np.abs(trace.data - np.mean(trace.data)))
                peak = val if np.isnan(peak) else max(peak, val)
            amplitudes.append(peak)

        return amplitudes

    def _extract_single_amplitude(
        self, large_window: obspy.Stream, pick: sbu.Pick, sub_inv: obspy.Inventory
    ):
        # normalize window
        large_window.detrend("linear")
        try:
            large_window.remove_response(sub_inv, **self.response_removal_args)
        except:  # No response information
            return np.nan

        large_window.simulate(paz_simulate=WOOD_ANDERSON)

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

        return np.nanmean(list(component_peaks.values()))
